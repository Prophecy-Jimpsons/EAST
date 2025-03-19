import os
import torch
import psutil
import numpy as np
import torch.nn as nn
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

# Memory monitoring function
def check_system_resources():
    ram = psutil.virtual_memory()
    print(f"System RAM: {ram.used/1e9:.2f}GB used / {ram.total/1e9:.2f}GB total")
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            device = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {device.name}")
            print(f"VRAM: {torch.cuda.memory_allocated(i)/1e9:.2f}GB allocated / {device.total_memory/1e9:.2f}GB total")

# Function to rebuild parameter sharing after loading
def rebuild_parameter_sharing(model):
    """
    Rebuild weight sharing between layers to match EAST architecture
    This recreates the sharing where layers 12-23 share weights with layer 11
    """
    print("Rebuilding parameter sharing relationships...")
    shared_params = {}
    total_params = 0
    shared_count = 0
    
    # Only proceed if model has layers attribute
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layers = model.model.layers
        num_layers = len(layers)
        
        # We use layer 11 as the source layer for sharing
        source_layer_idx = 11
        if source_layer_idx < num_layers:
            source_layer = layers[source_layer_idx]
            
            # Count total parameters for statistics
            for name, param in source_layer.named_parameters():
                if param.requires_grad:
                    total_params += param.numel()
            
            # Apply sharing to all subsequent layers (12-23)
            for i in range(source_layer_idx + 1, num_layers):
                print(f"  Layer {i} sharing with layer {source_layer_idx}")
                
                # Get matching parameters
                target_dict = dict(layers[i].named_parameters())
                source_dict = dict(source_layer.named_parameters())
                
                # For each parameter in the target layer
                for name, target_param in layers[i].named_parameters():
                    if name in source_dict:
                        source_param = source_dict[name]
                        
                        # Skip if already shared
                        if id(target_param) in shared_params:
                            continue
                            
                        if target_param.shape == source_param.shape:
                            try:
                                # Create scaling parameter for gradients
                                scaling_factor = nn.Parameter(torch.ones(1, device=target_param.device))
                                
                                # Replace parameter with a reference to source parameter
                                with torch.no_grad():
                                    # Save original parameter for reference
                                    target_param.orig_data = target_param.data.clone()
                                    # Point to the source parameter data
                                    target_param.data = source_param.data
                                
                                # Mark as shared and count
                                shared_params[id(target_param)] = (source_param, scaling_factor)
                                shared_count += target_param.numel()
                                
                                # Only register hook if parameter requires gradients
                                if target_param.requires_grad:
                                    def hook_factory(param, source, scale):
                                        def hook(grad):
                                            return grad * scale
                                        return hook
                                    
                                    target_param.register_hook(hook_factory(target_param, source_param, scaling_factor))
                                    print(f"    Shared parameter {name} with gradient hooks")
                                else:
                                    print(f"    Shared parameter {name} without gradient hooks (frozen parameter)")
                            except Exception as e:
                                print(f"    Failed to share parameter {name}: {str(e)}")
    
    # Print sharing statistics
    print(f"Total parameters: {total_params:,}")
    print(f"Shared parameters: {shared_count:,} ({shared_count/total_params*100:.2f}%)")
    
    return model


# Function to perform inference with the model
def generate_response(model, tokenizer, user_input, max_length=150, temperature=0.7):
    """Generate a response from the model based on user input"""
    try:
        start_time = time.time()
        
        # Prepare input
        inputs = tokenizer(user_input, return_tensors="pt", padding=True).to(model.device)
        
        # Ensure pad_token_id is properly set
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_length=max_length,
                do_sample=True,
                temperature=temperature,
                pad_token_id=pad_token_id
            )
        
        # Decode and return the response
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generation_time = time.time() - start_time
        
        # Return result and generation time
        return decoded, generation_time
    except Exception as e:
        print(f"Generation error: {e}")
        return f"Error generating response: {str(e)}", 0

# Function to load the EAST-processed model
def load_east_model(model_path="outputs/models/east_1b/final_model"):
    print(f"\n=== Loading EAST-processed model from {model_path} ===")
    
    try:
        # Check resources
        check_system_resources()
        
        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Method 1: Load directly from config without quantization
        print("Loading model from config (no quantization)...")
        config = AutoConfig.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_config(config)
        
        # Load state dict with strict=False to handle incompatible parameters
        print("Loading saved weights...")
        state_dict = torch.load(os.path.join(model_path, "model.pt"))
        model.load_state_dict(state_dict, strict=False)
        
        # Move to GPU first (important to do this before rebuilding sharing)
        model = model.to("cuda")
        
        # Rebuild parameter sharing between layers
        model = rebuild_parameter_sharing(model)
        
        # Configure for inference
        model = model.eval()
        
        print("EAST model loaded successfully!")
        check_system_resources()
        
        return model, tokenizer
    except Exception as e:
        print(f"Error in primary loading: {e}")
        
        # Try alternative loading approach
        try:
            print("Attempting alternative loading approach...")
            # For alternative approach, try loading original model and transfer weights
            model = AutoModelForCausalLM.from_pretrained(
                "deepseek-ai/deepseek-coder-1b-base",  
                torch_dtype=torch.float16,  # Use FP16 instead of quantization
                device_map="auto"
            )
            
            # Load and apply saved weights
            state_dict = torch.load(os.path.join(model_path, "model.pt"))
            
            # Filter the state_dict to match the model structure
            filtered_dict = {}
            model_dict = model.state_dict()
            for k, v in state_dict.items():
                if k in model_dict and v.shape == model_dict[k].shape:
                    filtered_dict[k] = v
            
            # Load compatible weights
            model.load_state_dict(filtered_dict, strict=False)
            
            # Apply parameter sharing
            model = rebuild_parameter_sharing(model)
            model = model.eval()
            
            print("Model loaded with alternative approach")
            return model, tokenizer
        except Exception as e2:
            print(f"Alternative loading also failed: {e2}")
        
        return None, None

# Interactive session with the model
def interactive_session(model, tokenizer):
    """Run an interactive session with the model"""
    print("\n=== Starting Interactive Session with EAST-processed DeepSeek Model ===")
    print("Type 'exit', 'quit', or 'q' to end the session.")
    print("Type 'stats' to see memory usage statistics.")
    print("Type 'help' to see these instructions again.")
    
    try:
        while True:
            # Get user input
            user_input = input("\nEnter prompt > ")
            
            # Check for exit commands
            if user_input.lower() in ['exit', 'quit', 'q']:
                print("Exiting interactive session.")
                break
            
            # Check for special commands
            if user_input.lower() == 'stats':
                check_system_resources()
                continue
            
            if user_input.lower() == 'help':
                print("Type 'exit', 'quit', or 'q' to end the session.")
                print("Type 'stats' to see memory usage statistics.")
                print("Type 'help' to see these instructions again.")
                continue
            
            # Skip empty prompts
            if not user_input.strip():
                print("Please enter a valid prompt.")
                continue
            
            # Generate a response
            print("Generating response...")
            response, generation_time = generate_response(model, tokenizer, user_input)
            
            # Display the response and time taken
            print(f"\nGeneration time: {generation_time:.4f} seconds")
            print("\nModel response:")
            print("=" * 40)
            print(response)
            print("=" * 40)
    
    except KeyboardInterrupt:
        print("\nSession interrupted by user. Cleaning up...")
    except Exception as e:
        print(f"\nError in interactive session: {e}")
    
    print("Interactive session ended.")

# Main function
def main():
    print("=== EAST DeepSeek Interactive Session ===")
    
    try:
        # Load the EAST-processed model
        model, tokenizer = load_east_model()
        
        if model is None or tokenizer is None:
            print("Failed to load the EAST model. Exiting.")
            return
        
        # Start interactive session
        interactive_session(model, tokenizer)
        
        # Cleanup
        print("\nCleaning up resources...")
        model.cpu()
        del model
        del tokenizer
        torch.cuda.empty_cache()
        
        print("Resources cleaned up. Goodbye!")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
