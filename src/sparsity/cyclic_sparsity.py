import torch
import numpy as np
from ..utils.memory_utils import check_system_resources

def compute_cyclic_sparsity(current_step, min_sparsity=0.8, max_sparsity=0.95, cycle=5):
    """
    Compute cyclic sparsity level based on current step
    
    Args:
        current_step (int): Current training step
        min_sparsity (float): Minimum sparsity level
        max_sparsity (float): Maximum sparsity level
        cycle (int): Number of steps in one cycle
        
    Returns:
        float: Current sparsity level
    """
    return min_sparsity + 0.5 * (max_sparsity - min_sparsity) * (1 + np.sin(np.pi * current_step / cycle))

def apply_cyclic_sparsity(model, min_sparsity=0.8, max_sparsity=0.95, cycle=5, current_step=1):
    """
    Implement cyclic sparsity with controlled memory usage
    
    Args:
        model: The model to apply cyclic sparsity to
        min_sparsity (float): Minimum sparsity level
        max_sparsity (float): Maximum sparsity level
        cycle (int): Number of steps in one cycle
        current_step (int): Current step in the cycle
        
    Returns:
        model: The model with cyclic sparsity applied
    """
    # Calculate current sparsity level based on cycle
    sparsity = compute_cyclic_sparsity(current_step, min_sparsity, max_sparsity, cycle)
    
    print(f"Applying cyclic sparsity with level: {sparsity:.4f} (step {current_step} in cycle {cycle})")
    print("Memory before applying cyclic sparsity:")
    check_system_resources()
    
    # Track which modules have been processed
    processed_modules = set()
    _apply_cyclic_sparsity_recursive(model, sparsity, processed_modules, chunk_size=1000000)
    
    print("Memory after applying cyclic sparsity:")
    check_system_resources()
    
    return model

def _apply_cyclic_sparsity_recursive(module, sparsity, processed_modules, chunk_size=1000000):
    """
    Recursively apply cyclic sparsity to all modules
    
    Args:
        module: The module to apply sparsity to
        sparsity (float): Current sparsity level
        processed_modules: Set of module IDs that have been processed
        chunk_size: Maximum tensor size for direct processing
    """
    # Avoid processing the same module twice
    if id(module) in processed_modules:
        return
    processed_modules.add(id(module))
    
    # Apply sparsity to this module if applicable
    if hasattr(module, 'weight') and hasattr(module, 'weight_orig'):
        with torch.no_grad():
            weight = module.weight
            
            # Handle large tensors with chunking
            if weight.numel() > chunk_size:
                # Flatten the tensor
                flat_tensor = weight.abs().view(-1)
                # Get tensor length
                tensor_len = flat_tensor.shape[0]
                # Calculate number of samples
                num_samples = min(chunk_size, int(tensor_len * 0.1))
                # Randomly sample indices
                indices = torch.randint(0, tensor_len, (num_samples,))
                # Get samples
                samples = flat_tensor[indices]
                # Compute threshold from samples
                threshold = torch.quantile(samples.float(), sparsity)
            else:
                threshold = torch.quantile(weight.abs().float(), sparsity)
                
            mask = (weight.abs() > threshold).float()
            weight.mul_(mask)
    
    # Process children
    for name, child in module.named_children():
        _apply_cyclic_sparsity_recursive(child, sparsity, processed_modules, chunk_size)
