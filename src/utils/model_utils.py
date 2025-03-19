import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import os
import gc

def load_model_and_tokenizer(model_name="deepseek-ai/deepseek-coder-1b-base", 
                            quantization=True, 
                            dtype=torch.float16):
    """
    Load a model and tokenizer with memory-efficient settings
    
    Args:
        model_name (str): HuggingFace model name
        quantization (bool): Whether to apply 4-bit quantization
        dtype (torch.dtype): Data type for model weights
        
    Returns:
        tuple: (model, tokenizer)
    """
    from .memory_utils import check_system_resources
    
    print(f"Loading model: {model_name}")
    print("Initial resource state:")
    check_system_resources()
    
    try:
        # Set up quantization if enabled
        if quantization:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        else:
            quant_config = None
        
        # Load tokenizer first (minimal memory impact)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load model with memory-optimized settings
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quant_config,
            device_map="auto",
            torch_dtype=dtype,
            low_cpu_mem_usage=True
        )
        
        print("Model loaded successfully!")
        check_system_resources()
        
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

def save_model(model, path, tokenizer=None):
    """Save model with PyTorch's native serialization"""
    os.makedirs(path, exist_ok=True)
    
    try:
        # Use PyTorch's native serialization
        torch.save(model.state_dict(), os.path.join(path, "model.pt"))
        print(f"Model state dictionary saved to {os.path.join(path, 'model.pt')}")
        
        # Save configuration files
        if hasattr(model, 'config'):
            model.config.save_pretrained(path)
            
        # Save tokenizer separately (crucial for DeepSeek)
        if tokenizer is not None:
            tokenizer.save_pretrained(path)
            
        print(f"Model saved to {path} with native PyTorch serialization")
    except Exception as e:
        print(f"Error saving model: {e}")










def safe_cleanup(model=None, tokenizer=None):
    """
    Safely clean up resources
    
    Args:
        model: Optional model to clean up
        tokenizer: Optional tokenizer to clean up
    """
    from .memory_utils import check_system_resources
    
    # Remove model from GPU
    if model is not None:
        try:
            model.cpu()
            del model
        except:
            pass
    
    # Remove tokenizer
    if tokenizer is not None:
        try:
            del tokenizer
        except:
            pass
    
    # Force garbage collection
    gc.collect()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Check final memory state
    print("Final memory state after cleanup:")
    check_system_resources()
