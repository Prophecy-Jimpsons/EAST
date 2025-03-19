import sys
import os
import torch
import json
import argparse
import time
import logging
from datetime import datetime

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.model_utils import load_model_and_tokenizer, save_model, safe_cleanup
from src.utils.memory_utils import check_system_resources, clean_gpu_memory
from src.sparsity.static_pruning import implement_static_pruning, count_zero_params
from src.sparsity.cyclic_sparsity import apply_cyclic_sparsity, compute_cyclic_sparsity
from src.evaluation.inference import test_model_inference, evaluate_model_performance

# Configure logging
def setup_logging(output_dir):
    """Set up logging to both console and file"""
    os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)
    log_file = os.path.join(output_dir, "logs", f"east_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logging.info(f"Logging initialized. Log file: {log_file}")
    return log_file

def parse_args():
    parser = argparse.ArgumentParser(description="Train with EAST methodology")
    parser.add_argument("--model_name", type=str, default="deepseek-ai/deepseek-coder-1b-base", 
                        help="Model name or path")
    parser.add_argument("--output_dir", type=str, default="outputs/models/east_1b", 
                        help="Output directory")
    parser.add_argument("--sparsity_config", type=str, default="configs/sparsity_config.json", 
                        help="Path to sparsity configuration")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate after training")
    parser.add_argument("--save_model", action="store_true", help="Save model after training")
    parser.add_argument("--log_level", type=str, default="INFO", 
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Logging level")
    return parser.parse_args()

def log_memory_stats(stage_name):
    """Log detailed memory statistics at a specific stage"""
    logging.info(f"Memory stats at {stage_name}:")
    mem_stats = check_system_resources()
    
    # Explicitly log GPU utilization percentage if available
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            device = torch.cuda.get_device_properties(i)
            utilization = torch.cuda.memory_allocated(i) / device.total_memory * 100
            logging.info(f"  GPU {i} utilization: {utilization:.2f}%")
    
    return mem_stats

def time_operation(operation_name, func, *args, **kwargs):
    """Execute a function while timing it and logging the duration"""
    logging.info(f"Starting {operation_name}...")
    start_time = time.time()
    
    try:
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        logging.info(f"Completed {operation_name} in {elapsed:.2f} seconds")
        return result
    except Exception as e:
        elapsed = time.time() - start_time
        logging.error(f"Failed {operation_name} after {elapsed:.2f} seconds: {str(e)}")
        raise

def main():
    # Parse arguments
    args = parse_args()
    
    # Create output directory and set up logging
    os.makedirs(args.output_dir, exist_ok=True)
    log_file = setup_logging(args.output_dir)
    
    # Log script start and configuration
    logging.info("=" * 50)
    logging.info(f"EAST Training Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info(f"PyTorch version: {torch.__version__}")
    logging.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logging.info(f"CUDA version: {torch.version.cuda}")
        logging.info(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Log initial system state
    logging.info("Initial system state:")
    initial_mem = log_memory_stats("initialization")
    
    # Create metadata dictionary to track execution
    metadata = {
        "start_time": datetime.now().isoformat(),
        "model_name": args.model_name,
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "command_args": vars(args),
        "stages": {}
    }
    
    try:
        # Load sparsity configuration
        logging.info(f"Loading sparsity configuration from {args.sparsity_config}")
        try:
            with open(args.sparsity_config, 'r') as f:
                sparsity_config = json.load(f)
                logging.info(f"Successfully loaded sparsity configuration")
                logging.debug(f"Configuration: {json.dumps(sparsity_config, indent=2)}")
        except Exception as e:
            logging.warning(f"Error loading sparsity config: {e}")
            logging.warning("Using default configuration")
            sparsity_config = {
                "static_pruning": {
                    "sparsity": 0.8,
                    "chunk_size": 1000000
                },
                "cyclic_sparsity": {
                    "min_sparsity": 0.8,
                    "max_sparsity": 0.95,
                    "cycle": 5
                },
                "dynamic_relu": {
                    "enabled": True,
                    "start_epoch": 0,
                    "end_epoch": 75
                },
                "weight_sharing": {
                    "enabled": True,
                    "share_ratio": 0.5
                }
            }
            logging.debug(f"Default configuration: {json.dumps(sparsity_config, indent=2)}")
        
        # Step 1: Load model and tokenizer
        logging.info("Step 1: Loading model and tokenizer")
        stage_start = time.time()
        model, tokenizer = time_operation("model loading", 
                                         load_model_and_tokenizer, 
                                         args.model_name)
        
        if model is None or tokenizer is None:
            logging.error("Failed to load model or tokenizer. Exiting.")
            return
        
        metadata["stages"]["model_loading"] = {
            "duration_seconds": time.time() - stage_start,
            "memory_after": log_memory_stats("after model loading")
        }
        
        # Step 2: Apply DyReLU phasing (first component of EAST)
        if sparsity_config.get("dynamic_relu", {}).get("enabled", False):
            logging.info("Step 2: Applying DyReLU phasing")
            stage_start = time.time()
            
            try:
                from src.sparsity.dynamic_relu import apply_dyrelu_phasing
                model = time_operation("DyReLU phasing",
                                      apply_dyrelu_phasing,
                                      model, 
                                      current_epoch=0,
                                      start_epoch=sparsity_config["dynamic_relu"].get("start_epoch", 0),
                                      end_epoch=sparsity_config["dynamic_relu"].get("end_epoch", 75))
                
                metadata["stages"]["dyrelu_phasing"] = {
                    "duration_seconds": time.time() - stage_start,
                    "memory_after": log_memory_stats("after DyReLU phasing")
                }
            except ImportError as e:
                logging.error(f"Failed to import DyReLU module: {e}")
                logging.warning("Skipping DyReLU phasing")
            except Exception as e:
                logging.error(f"Error during DyReLU phasing: {e}")
                logging.warning("Continuing without DyReLU phasing")
        
        
        # Step 3: Apply weight sharing (second component of EAST)
        if sparsity_config.get("weight_sharing", {}).get("enabled", False):
            logging.info("Step 3: Applying weight sharing")
            stage_start = time.time()
            
            try:
                from src.sparsity.weight_sharing import apply_weight_sharing
                model = time_operation("weight sharing",
                                    apply_weight_sharing,
                                    model,
                                    share_blocks=True,
                                    share_ratio=sparsity_config["weight_sharing"].get("share_ratio", 0.5))
                
                metadata["stages"]["weight_sharing"] = {
                    "duration_seconds": time.time() - stage_start,
                    "memory_after": log_memory_stats("after weight sharing")
                }
            except ImportError as e:
                logging.error(f"Failed to import weight sharing module: {e}")
                logging.warning("Skipping weight sharing")
            except Exception as e:
                logging.error(f"Error during weight sharing: {e}")
                logging.warning("Continuing without weight sharing")

        
        # Step 4: Log initial model statistics
        logging.info("Step 4: Gathering initial model statistics")
        stage_start = time.time()
        initial_zero_params, initial_total_params, initial_sparsity = time_operation(
            "parameter counting", 
            count_zero_params, 
            model
        )
        
        metadata["stages"]["initial_stats"] = {
            "duration_seconds": time.time() - stage_start,
            "zero_params": initial_zero_params,
            "total_params": initial_total_params,
            "sparsity": initial_sparsity
        }
        
        # Step 5: Apply static pruning
        logging.info("Step 5: Applying static pruning")
        stage_start = time.time()
        static_sparsity = sparsity_config["static_pruning"]["sparsity"]
        chunk_size = sparsity_config["static_pruning"].get("chunk_size", 1000000)
        
        logging.info(f"Static pruning with sparsity={static_sparsity}, chunk_size={chunk_size}")
        
        model = time_operation(
            "static pruning", 
            implement_static_pruning, 
            model, 
            sparsity=static_sparsity, 
            chunk_size=chunk_size
        )
        
        # Count parameters after static pruning
        zero_after_static, total_after_static, sparsity_after_static = count_zero_params(model)
        
        metadata["stages"]["static_pruning"] = {
            "duration_seconds": time.time() - stage_start,
            "sparsity_target": static_sparsity,
            "sparsity_achieved": sparsity_after_static,
            "zero_params": zero_after_static,
            "total_params": total_after_static,
            "memory_after": log_memory_stats("after static pruning")
        }
        
        # Step 6: Apply cyclic sparsity
        logging.info("Step 6: Applying cyclic sparsity")
        stage_start = time.time()
        cyclic_config = sparsity_config["cyclic_sparsity"]
        
        logging.info(f"Cyclic sparsity with min={cyclic_config['min_sparsity']}, "
                     f"max={cyclic_config['max_sparsity']}, cycle={cyclic_config['cycle']}")
        
        model = time_operation(
            "cyclic sparsity", 
            apply_cyclic_sparsity,
            model, 
            min_sparsity=cyclic_config["min_sparsity"],
            max_sparsity=cyclic_config["max_sparsity"],
            cycle=cyclic_config["cycle"],
            current_step=1
        )
        
        # Count parameters after cyclic sparsity
        zero_after_cyclic, total_after_cyclic, sparsity_after_cyclic = count_zero_params(model)
        
        metadata["stages"]["cyclic_sparsity"] = {
            "duration_seconds": time.time() - stage_start,
            "min_sparsity": cyclic_config["min_sparsity"],
            "max_sparsity": cyclic_config["max_sparsity"],
            "cycle": cyclic_config["cycle"],
            "sparsity_achieved": sparsity_after_cyclic,
            "zero_params": zero_after_cyclic,
            "total_params": total_after_cyclic,
            "memory_after": log_memory_stats("after cyclic sparsity")
        }
        
        # Step 7: Evaluate if requested
        if args.evaluate:
            logging.info("Step 7: Evaluating model")
            stage_start = time.time()
            
            eval_results = time_operation(
                "model evaluation", 
                evaluate_model_performance, 
                model, 
                tokenizer
            )
            
            # Save evaluation results
            eval_output_path = os.path.join(args.output_dir, "evaluation_results.json")
            with open(eval_output_path, 'w') as f:
                json.dump(eval_results, f, indent=2)
            logging.info(f"Evaluation results saved to {eval_output_path}")
            
            metadata["stages"]["evaluation"] = {
                "duration_seconds": time.time() - stage_start,
                "results_file": eval_output_path,
                "prompts_evaluated": len(eval_results),
                "memory_after": log_memory_stats("after evaluation")
            }
        else:
            logging.info("Skipping evaluation (--evaluate not specified)")
        
        # Step 8: Save model if requested
        if args.save_model:
            logging.info("Step 8: Saving model")
            stage_start = time.time()
            
            save_path = os.path.join(args.output_dir, "final_model")
            time_operation(
                "model saving", 
                save_model, 
                model, 
                save_path, 
                tokenizer
            )
            
            metadata["stages"]["model_saving"] = {
                "duration_seconds": time.time() - stage_start,
                "save_path": save_path,
                "memory_after": log_memory_stats("after model saving")
            }
        else:
            logging.info("Skipping model saving (--save_model not specified)")
        
        # Step 9: Clean up resources
        logging.info("Step 9: Cleaning up resources")
        stage_start = time.time()
        
        time_operation(
            "resource cleanup", 
            safe_cleanup, 
            model, 
            tokenizer
        )
        
        metadata["stages"]["cleanup"] = {
            "duration_seconds": time.time() - stage_start,
            "memory_after": log_memory_stats("after cleanup")
        }
        
        # Log completion
        metadata["end_time"] = datetime.now().isoformat()
        metadata["total_duration_seconds"] = (datetime.now() - datetime.fromisoformat(metadata["start_time"])).total_seconds()
        
        # Save metadata
        metadata_path = os.path.join(args.output_dir, "execution_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logging.info(f"Execution metadata saved to {metadata_path}")
        
        logging.info(f"\nTraining completed successfully in {metadata['total_duration_seconds']:.2f} seconds!")
        logging.info(f"See detailed logs at: {log_file}")
        logging.info("=" * 50)
        
    except Exception as e:
        logging.error(f"Training failed with error: {str(e)}", exc_info=True)
        
        # Save error metadata
        metadata["error"] = str(e)
        metadata["end_time"] = datetime.now().isoformat()
        metadata["total_duration_seconds"] = (datetime.now() - datetime.fromisoformat(metadata["start_time"])).total_seconds()
        metadata_path = os.path.join(args.output_dir, "execution_metadata.json")
        
        try:
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            logging.info(f"Error metadata saved to {metadata_path}")
        except:
            logging.error("Failed to save error metadata")
        
        # Force cleanup on error
        try:
            if 'model' in locals() and 'tokenizer' in locals():
                safe_cleanup(model, tokenizer)
        except:
            logging.error("Failed to clean up resources after error")
        
        logging.error("Training failed. See logs for details.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
