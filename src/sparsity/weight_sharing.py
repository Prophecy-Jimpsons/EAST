import os
import json
import torch
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer

def apply_weight_sharing(model, source_layer_idx=11, target_layer_range=(12, 23)):
    """Apply weight sharing between model layers with detailed tracking"""
    print(f"Applying weight sharing from layer {source_layer_idx} to layers {target_layer_range[0]}-{target_layer_range[1]}")
    
    if not hasattr(model, 'model') or not hasattr(model.model, 'layers'):
        print("Model architecture not compatible with weight sharing")
        return model
    
    layers = model.model.layers
    total_shared_params = 0
    sharing_map = {}
    
    # Verify source layer exists
    if source_layer_idx >= len(layers):
        print(f"Source layer {source_layer_idx} does not exist in model with {len(layers)} layers")
        return model
    
    source_layer = layers[source_layer_idx]
    start_idx, end_idx = target_layer_range
    
    # Apply sharing to specified range
    for i in range(start_idx, min(end_idx + 1, len(layers))):
        if i == source_layer_idx:
            continue
            
        target_layer = layers[i]
        layer_shared_params = 0
        
        # Share parameters between layers
        for name, source_param in source_layer.named_parameters():
            if 'weight' in name:  # Only share weights
                target_param_dict = dict(target_layer.named_parameters())
                if name in target_param_dict:
                    target_param = target_param_dict[name]
                    if target_param.shape == source_param.shape:
                        # Point to source parameter data
                        target_param.data = source_param.data
                        layer_shared_params += target_param.numel()
        
        total_shared_params += layer_shared_params
        print(f"  Layer {i} now shares {layer_shared_params:,} parameters with layer {source_layer_idx}")
        sharing_map[str(i)] = source_layer_idx
    
    # Save sharing map to model for serialization
    model.weight_sharing_map = sharing_map
    
    print(f"Total parameters shared: {total_shared_params:,}")
    return model

def enhanced_save_model(model, path, tokenizer=None):
    """Save model with preserved weight sharing relationships"""
    # Create directory structure
    os.makedirs(path, exist_ok=True)
    
    # Create weight sharing metadata
    metadata = {
        "weight_sharing_map": getattr(model, 'weight_sharing_map', {}),
        "model_version": "east-deepseek-1.0",
        "creation_date": datetime.now().isoformat(),
        "sparsity_configuration": getattr(model, 'sparsity_config', {}),
        "training_precision": getattr(model, 'training_precision', "bfloat16")
    }
    
    # Save model weights using PyTorch's native serialization
    torch.save(model.state_dict(), os.path.join(path, "model.pt"))
    
    # Save metadata separately as JSON
    with open(os.path.join(path, "east_metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Save tokenizer
    if tokenizer is not None:
        tokenizer.save_pretrained(path)
    
    print(f"Model successfully saved to {path} with weight sharing metadata")
    return True

def enhanced_load_model(model_path, device="auto", fallback_model="deepseek-ai/deepseek-coder-1b-base"):
    """Load model and properly restore weight sharing"""
    print(f"Loading EAST-processed model from {model_path}")
    
    try:
        # Load metadata first to guide the process
        metadata_path = os.path.join(model_path, "east_metadata.json")
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Determine precision from metadata
        precision = metadata.get("training_precision", "bfloat16")
        dtype = torch.bfloat16 if precision == "bfloat16" else torch.float16
        
        # Load the model using optimized settings
        model = AutoModelForCausalLM.from_pretrained(
            fallback_model,  # Base architecture
            torch_dtype=dtype,
            device_map=device,
            low_cpu_mem_usage=True
        )
        
        # Load state dict with strict=False to handle weight sharing
        state_dict = torch.load(os.path.join(model_path, "model.pt"))
        model.load_state_dict(state_dict, strict=False)
        
        # Rebuild weight sharing based on metadata
        if "weight_sharing_map" in metadata:
            model = rebuild_weight_sharing(model, metadata["weight_sharing_map"])
        
        # Store metadata in model for future use
        model.weight_sharing_map = metadata.get("weight_sharing_map", {})
        model.sparsity_config = metadata.get("sparsity_configuration", {})
        model.training_precision = metadata.get("training_precision", "bfloat16")
        
        model = model.eval()  # Set to evaluation mode
        
        print(f"Successfully loaded model with weight sharing")
        return model, tokenizer
    except Exception as e:
        print(f"Error in model loading: {e}")
        print(f"Attempting fallback loading...")
        
        # Fallback to regular loading
        model = AutoModelForCausalLM.from_pretrained(
            fallback_model,
            torch_dtype=torch.bfloat16,
            device_map=device,
            low_cpu_mem_usage=True
        )
        tokenizer = AutoTokenizer.from_pretrained(fallback_model)
        
        return model, tokenizer

def rebuild_weight_sharing(model, sharing_map):
    """Rebuild weight sharing between layers based on metadata"""
    print("Rebuilding parameter sharing relationships...")
    
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layers = model.model.layers
        shared_count = 0
        
        # Rebuild sharing based on the map
        for target_layer_idx, source_layer_idx in sharing_map.items():
            target_idx = int(target_layer_idx)
            source_idx = int(source_layer_idx)
            
            if target_idx >= len(layers) or source_idx >= len(layers):
                continue
                
            print(f"  Layer {target_idx} sharing with layer {source_idx}")
            
            source_layer = layers[source_idx]
            target_layer = layers[target_idx]
            
            # Share all relevant parameters
            for name, source_param in source_layer.named_parameters():
                if 'weight' in name:  # Only share weights
                    target_param = dict(target_layer.named_parameters()).get(name)
                    if target_param is not None and target_param.shape == source_param.shape:
                        # Point to source parameter data
                        target_param.data = source_param.data
                        shared_count += target_param.numel()
        
        print(f"Rebuilt sharing for {shared_count:,} parameters")
    
    return model
