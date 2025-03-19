import torch
from transformers import AutoTokenizer
from ..utils.memory_utils import check_system_resources

def implement_adaptive_pruning(model, target_sparsity=0.5, 
                              verify_output=True, verification_prompt="def fibonacci(n):"):
    """Implement adaptive pruning with layer-specific sparsity levels"""
    print(f"Applying adaptive pruning with target sparsity {target_sparsity}")
    
    # Store sparsity configuration for serialization
    sparsity_config = {
        'target_sparsity': target_sparsity,
        'layer_specific_sparsity': {
            'embed_tokens': max(0.2, target_sparsity - 0.3),  # Lower sparsity for embeddings
            'lm_head': max(0.2, target_sparsity - 0.3),       # Lower sparsity for output layer
            'q_proj': max(0.3, target_sparsity - 0.2),        # Lower sparsity for attention query
            'k_proj': max(0.3, target_sparsity - 0.2),        # Lower sparsity for attention key
            'v_proj': max(0.3, target_sparsity - 0.2),        # Lower sparsity for attention value
            'o_proj': max(0.3, target_sparsity - 0.2),        # Lower sparsity for attention output
            'default': target_sparsity                         # Default sparsity
        }
    }
    
    # Store config in model for serialization
    model.sparsity_config = sparsity_config
    
    pruned_params = 0
    total_params = 0
    layer_stats = {}
    
    # First pass: measure total parameters for accurate reporting
    for name, param in model.named_parameters():
        if param.requires_grad and 'weight' in name:
            total_params += param.numel()
    
    # Second pass: actual pruning
    for name, param in model.named_parameters():
        if param.requires_grad and 'weight' in name:
            # Determine appropriate sparsity for this layer
            sparsity = sparsity_config['layer_specific_sparsity']['default']
            for layer_name, layer_sparsity_value in sparsity_config['layer_specific_sparsity'].items():
                if layer_name in name:
                    sparsity = layer_sparsity_value
                    break
            
            with torch.no_grad():
                # Compute threshold based on magnitude
                tensor_values = param.abs().float()
                threshold = torch.quantile(tensor_values, sparsity)
                
                # Create binary mask
                mask = (tensor_values > threshold).float()
                
                # Apply mask to weights using multiplication
                param.mul_(mask)
                
                # Track statistics
                pruned_count = (mask == 0).sum().item()
                layer_stats[name] = {
                    'total': param.numel(),
                    'pruned': pruned_count,
                    'sparsity': pruned_count / param.numel()
                }
                pruned_params += pruned_count
    
    print(f"Pruned approximately {pruned_params:,} parameters out of {total_params:,}")
    print(f"Achieved sparsity: {pruned_params / total_params:.2%}")
    
    # Optional verification to ensure model still functions
    if verify_output and hasattr(model, 'generate'):
        try:
            # Get tokenizer from model or load default
            tokenizer = getattr(model, 'tokenizer', None)
            if tokenizer is None:
                if hasattr(model, 'config') and hasattr(model.config, 'name_or_path'):
                    tokenizer = AutoTokenizer.from_pretrained(model.config.name_or_path)
                else:
                    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-1b-base")
            
            # Test generate functionality
            inputs = tokenizer(verification_prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(inputs.input_ids, max_length=50)
            generated_text = tokenizer.decode(outputs[0])
            
            print(f"Verification output after pruning: {generated_text[:100]}...")
            print("Model generation successful after pruning.")
        except Exception as e:
            print(f"Warning: Verification failed with error: {e}")
    
    # Store detailed layer stats in model for further analysis
    model.pruning_stats = layer_stats
    
    return model

def implement_static_pruning(model, sparsity=0.8, chunk_size=1000000):
    """Implement static magnitude-based pruning with chunking for large tensors"""
    pruned_params = 0
    total_params = 0
    
    # Store simple sparsity config for serialization
    model.sparsity_config = {'target_sparsity': sparsity, 'method': 'static'}
    
    for name, param in model.named_parameters():
        if param.requires_grad and 'weight' in name:
            total_params += param.numel()
            
            # Skip extremely small parameters
            if param.numel() < 10:
                continue
                
            with torch.no_grad():
                # For large tensors, estimate threshold using sampling
                if param.numel() > chunk_size:
                    # Flatten the tensor
                    flat_tensor = param.abs().view(-1)
                    # Get tensor length
                    tensor_len = flat_tensor.shape[0]
                    # Calculate number of samples (use 10% of tensor size)
                    num_samples = min(chunk_size, int(tensor_len * 0.1))
                    # Randomly sample indices
                    indices = torch.randint(0, tensor_len, (num_samples,))
                    # Get samples
                    samples = flat_tensor[indices]
                    # Compute threshold from samples
                    threshold = torch.quantile(samples.float(), sparsity)
                else:
                    # For smaller tensors, compute quantile directly
                    threshold = torch.quantile(param.abs().float(), sparsity)
                
                # Create binary mask and apply it
                mask = (param.abs() > threshold).float()
                param.mul_(mask)
                
                # Count zero parameters
                zeros = (param == 0).sum().item()
                pruned_params += zeros
    
    print(f"Pruned approximately {pruned_params:,} parameters out of {total_params:,}")
    return model


def count_zero_params(model):
    """
    Count the number of zero-valued parameters in a model
    
    Args:
        model: The model to analyze
        
    Returns:
        tuple: (zero_params, total_params, sparsity_ratio)
    """
    zero_params = 0
    total_params = 0
    
    for name, param in model.named_parameters():
        if 'weight' in name:
            zero_params += (param == 0).sum().item()
            total_params += param.numel()
    
    sparsity_ratio = zero_params / total_params if total_params > 0 else 0
    
    print(f"Zero parameters: {zero_params:,} out of {total_params:,} ({sparsity_ratio:.4f})")
    return zero_params, total_params, sparsity_ratio


def get_layer_sparsity_distribution(model):
    """
    Analyze sparsity distribution across different layers
    
    Args:
        model: The model to analyze
        
    Returns:
        dict: Dictionary with layer-wise sparsity statistics
    """
    layer_stats = {}
    
    for name, param in model.named_parameters():
        if 'weight' in name:
            zeros = (param == 0).sum().item()
            total = param.numel()
            sparsity = zeros / total if total > 0 else 0
            
            layer_stats[name] = {
                'zeros': zeros,
                'total': total,
                'sparsity': sparsity
            }
    
    return layer_stats
