import os
import time
import torch
import numpy as np
import shutil
from typing import Dict, List, Tuple, Optional, Union

# Import required functions from project modules
from src.sparsity.weight_sharing import apply_weight_sharing, enhanced_save_model, enhanced_load_model
from src.sparsity.static_pruning import implement_adaptive_pruning

class ComponentVerification:
    """Framework to verify individual components before combining them"""
    
    def __init__(self, base_model_name="deepseek-ai/deepseek-coder-1b-base", device="auto"):
        self.base_model_name = base_model_name
        self.device = device
        self.results = {}
        self.evaluation_prompts = [
            "def fibonacci(n):",
            "Write a function to sort an array using quicksort.",
            "# Function to calculate the factorial of a number",
            "class BinarySearchTree:",
            "function calculateTax(income) {"
        ]
    
    def load_baseline_model(self):
        """Load the baseline model for comparison"""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        print(f"Loading baseline model: {self.base_model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
            low_cpu_mem_usage=True
        )
        tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        
        return model, tokenizer
    
    def measure_memory_usage(self, model):
        """Measure memory usage of the model"""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            memory_allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
            memory_reserved = torch.cuda.memory_reserved() / (1024 ** 3)    # GB
            return {
                "allocated_gb": memory_allocated,
                "reserved_gb": memory_reserved
            }
        else:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            return {
                "rss_gb": memory_info.rss / (1024 ** 3),
                "vms_gb": memory_info.vms / (1024 ** 3)
            }
    
    def measure_inference_speed(self, model, tokenizer, prompt, num_runs=3):
        """Measure inference speed of the model"""
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # Warm-up run
        with torch.no_grad():
            _ = model.generate(inputs.input_ids, max_length=100)
        
        # Timed runs
        times = []
        for _ in range(num_runs):
            start_time = time.time()
            with torch.no_grad():
                outputs = model.generate(inputs.input_ids, max_length=100)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end_time = time.time()
            times.append(end_time - start_time)
        
        return {
            "avg_time_seconds": np.mean(times),
            "std_dev_seconds": np.std(times),
            "output_tokens": len(outputs[0])
        }
    
    def calculate_perplexity(self, model, tokenizer, text):
        """Calculate perplexity of the model on a given text"""
        encodings = tokenizer(text, return_tensors="pt").to(model.device)
        
        # Get input_ids and create target_ids (shifted by 1)
        input_ids = encodings.input_ids
        target_ids = input_ids.clone()
        
        # Calculate loss with model
        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss
        
        # Calculate perplexity
        perplexity = torch.exp(neg_log_likelihood).item()
        return perplexity
    
    def test_weight_sharing(self, source_layer_idx=11, target_layer_range=(12, 23)):
        """Test weight sharing component in isolation"""
        print("\n=== Testing Weight Sharing Component ===")
        
        # Load baseline model
        model, tokenizer = self.load_baseline_model()
        
        # Measure baseline
        baseline_memory = self.measure_memory_usage(model)
        baseline_speed = self.measure_inference_speed(model, tokenizer, self.evaluation_prompts[0])
        baseline_perplexity = self.calculate_perplexity(model, tokenizer, self.evaluation_prompts[1])
        
        print(f"Baseline Memory: {baseline_memory}")
        print(f"Baseline Speed: {baseline_speed['avg_time_seconds']:.4f}s")
        print(f"Baseline Perplexity: {baseline_perplexity:.4f}")
        
        # Apply weight sharing
        model = apply_weight_sharing(model, source_layer_idx, target_layer_range)
        
        # Measure after weight sharing
        shared_memory = self.measure_memory_usage(model)
        shared_speed = self.measure_inference_speed(model, tokenizer, self.evaluation_prompts[0])
        shared_perplexity = self.calculate_perplexity(model, tokenizer, self.evaluation_prompts[1])
        
        print(f"After Sharing Memory: {shared_memory}")
        print(f"After Sharing Speed: {shared_speed['avg_time_seconds']:.4f}s")
        print(f"After Sharing Perplexity: {shared_perplexity:.4f}")
        
        # Test serialization and loading
        temp_dir = "./temp_model_shared"
        enhanced_save_model(model, temp_dir, tokenizer)
        loaded_model, loaded_tokenizer = enhanced_load_model(temp_dir, device=self.device)
        
        # Measure after loading
        loaded_memory = self.measure_memory_usage(loaded_model)
        loaded_speed = self.measure_inference_speed(loaded_model, loaded_tokenizer, self.evaluation_prompts[0])
        loaded_perplexity = self.calculate_perplexity(loaded_model, loaded_tokenizer, self.evaluation_prompts[1])
        
        print(f"After Loading Memory: {loaded_memory}")
        print(f"After Loading Speed: {loaded_speed['avg_time_seconds']:.4f}s")
        print(f"After Loading Perplexity: {loaded_perplexity:.4f}")
        
        # Collect results
        self.results["weight_sharing"] = {
            "baseline": {
                "memory": baseline_memory,
                "speed": baseline_speed,
                "perplexity": baseline_perplexity
            },
            "after_sharing": {
                "memory": shared_memory,
                "speed": shared_speed,
                "perplexity": shared_perplexity
            },
            "after_loading": {
                "memory": loaded_memory,
                "speed": loaded_speed,
                "perplexity": loaded_perplexity
            }
        }
        
        # Clean up
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        
        del model, tokenizer, loaded_model, loaded_tokenizer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return self.results["weight_sharing"]
    
    def test_adaptive_pruning(self, target_sparsity=0.5):
        """Test adaptive pruning component in isolation"""
        print("\n=== Testing Adaptive Pruning Component ===")
        
        # Load baseline model
        model, tokenizer = self.load_baseline_model()
        
        # Measure baseline
        baseline_memory = self.measure_memory_usage(model)
        baseline_speed = self.measure_inference_speed(model, tokenizer, self.evaluation_prompts[0])
        baseline_perplexity = self.calculate_perplexity(model, tokenizer, self.evaluation_prompts[1])
        
        print(f"Baseline Memory: {baseline_memory}")
        print(f"Baseline Speed: {baseline_speed['avg_time_seconds']:.4f}s")
        print(f"Baseline Perplexity: {baseline_perplexity:.4f}")
        
        # Apply pruning
        model = implement_adaptive_pruning(model, target_sparsity)
        
        # Measure after pruning
        pruned_memory = self.measure_memory_usage(model)
        pruned_speed = self.measure_inference_speed(model, tokenizer, self.evaluation_prompts[0])
        pruned_perplexity = self.calculate_perplexity(model, tokenizer, self.evaluation_prompts[1])
        
        print(f"After Pruning Memory: {pruned_memory}")
        print(f"After Pruning Speed: {pruned_speed['avg_time_seconds']:.4f}s")
        print(f"After Pruning Perplexity: {pruned_perplexity:.4f}")
        
        # Test serialization and loading
        temp_dir = "./temp_model_pruned"
        enhanced_save_model(model, temp_dir, tokenizer)
        loaded_model, loaded_tokenizer = enhanced_load_model(temp_dir, device=self.device)
        
        # Measure after loading
        loaded_memory = self.measure_memory_usage(loaded_model)
        loaded_speed = self.measure_inference_speed(loaded_model, loaded_tokenizer, self.evaluation_prompts[0])
        loaded_perplexity = self.calculate_perplexity(loaded_model, loaded_tokenizer, self.evaluation_prompts[1])
        
        print(f"After Loading Memory: {loaded_memory}")
        print(f"After Loading Speed: {loaded_speed['avg_time_seconds']:.4f}s")
        print(f"After Loading Perplexity: {loaded_perplexity:.4f}")
        
        # Collect results
        self.results["adaptive_pruning"] = {
            "baseline": {
                "memory": baseline_memory,
                "speed": baseline_speed,
                "perplexity": baseline_perplexity
            },
            "after_pruning": {
                "memory": pruned_memory,
                "speed": pruned_speed,
                "perplexity": pruned_perplexity
            },
            "after_loading": {
                "memory": loaded_memory,
                "speed": loaded_speed,
                "perplexity": loaded_perplexity
            }
        }
        
        # Clean up
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        
        del model, tokenizer, loaded_model, loaded_tokenizer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return self.results["adaptive_pruning"]
    
    def test_combined_optimizations(self, source_layer_idx=11, target_layer_range=(12, 23), target_sparsity=0.5):
        """Test combined weight sharing and pruning"""
        print("\n=== Testing Combined Weight Sharing and Pruning ===")
        
        # Load baseline model
        model, tokenizer = self.load_baseline_model()
        
        # Measure baseline
        baseline_memory = self.measure_memory_usage(model)
        baseline_speed = self.measure_inference_speed(model, tokenizer, self.evaluation_prompts[0])
        baseline_perplexities = [self.calculate_perplexity(model, tokenizer, prompt) 
                                for prompt in self.evaluation_prompts]
        
        print(f"Baseline Memory: {baseline_memory}")
        print(f"Baseline Speed: {baseline_speed['avg_time_seconds']:.4f}s")
        print(f"Baseline Avg Perplexity: {np.mean(baseline_perplexities):.4f}")
        
        # Apply weight sharing first
        model = apply_weight_sharing(model, source_layer_idx, target_layer_range)
        
        # Measure after weight sharing
        shared_memory = self.measure_memory_usage(model)
        shared_speed = self.measure_inference_speed(model, tokenizer, self.evaluation_prompts[0])
        shared_perplexities = [self.calculate_perplexity(model, tokenizer, prompt) 
                              for prompt in self.evaluation_prompts]
        
        print(f"After Sharing Memory: {shared_memory}")
        print(f"After Sharing Speed: {shared_speed['avg_time_seconds']:.4f}s")
        print(f"After Sharing Avg Perplexity: {np.mean(shared_perplexities):.4f}")
        
        # Then apply pruning
        model = implement_adaptive_pruning(model, target_sparsity)
        
        # Measure after both optimizations
        combined_memory = self.measure_memory_usage(model)
        combined_speed = self.measure_inference_speed(model, tokenizer, self.evaluation_prompts[0])
        combined_perplexities = [self.calculate_perplexity(model, tokenizer, prompt) 
                                for prompt in self.evaluation_prompts]
        
        print(f"After Combined Memory: {combined_memory}")
        print(f"After Combined Speed: {combined_speed['avg_time_seconds']:.4f}s")
        print(f"After Combined Avg Perplexity: {np.mean(combined_perplexities):.4f}")
        
        # Test serialization and loading
        temp_dir = "./temp_model_combined"
        enhanced_save_model(model, temp_dir, tokenizer)
        loaded_model, loaded_tokenizer = enhanced_load_model(temp_dir, device=self.device)
        
        # Measure after loading
        loaded_memory = self.measure_memory_usage(loaded_model)
        loaded_speed = self.measure_inference_speed(loaded_model, loaded_tokenizer, self.evaluation_prompts[0])
        loaded_perplexities = [self.calculate_perplexity(loaded_model, loaded_tokenizer, prompt) 
                              for prompt in self.evaluation_prompts]
        
        print(f"After Loading Memory: {loaded_memory}")
        print(f"After Loading Speed: {loaded_speed['avg_time_seconds']:.4f}s")
        print(f"After Loading Avg Perplexity: {np.mean(loaded_perplexities):.4f}")
        
        # Collect results
        self.results["combined"] = {
            "baseline": {
                "memory": baseline_memory,
                "speed": baseline_speed,
                "perplexity": np.mean(baseline_perplexities)
            },
            "after_sharing": {
                "memory": shared_memory,
                "speed": shared_speed,
                "perplexity": np.mean(shared_perplexities)
            },
            "after_combined": {
                "memory": combined_memory,
                "speed": combined_speed,
                "perplexity": np.mean(combined_perplexities)
            },
            "after_loading": {
                "memory": loaded_memory,
                "speed": loaded_speed,
                "perplexity": np.mean(loaded_perplexities)
            }
        }
        
        # Clean up
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        
        del model, tokenizer, loaded_model, loaded_tokenizer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return self.results["combined"]
