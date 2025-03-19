import torch
import torch.nn as nn
import torch.nn.functional as F

class DyReLU(nn.Module):
    """
    Dynamic ReLU implementation based on the EAST paper
    """
    def __init__(self, channels, reduction=4):
        super(DyReLU, self).__init__()
        self.channels = channels
        self.reduction = reduction
        
        # Context modeling for dynamic activation
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, 2 * channels, bias=True)
        )
        # Initialize final layer with zeros to start with identity mapping
        nn.init.zeros_(self.fc[2].weight)
        nn.init.zeros_(self.fc[2].bias)
        
    def forward(self, x):
        # Calculate dynamic coefficients
        x_shape = x.shape
        if len(x_shape) == 2:
            x_pool = self.avg_pool(x.unsqueeze(2)).squeeze(2)
        else:
            x_pool = self.avg_pool(x.transpose(1, 2)).squeeze(2)
            
        theta = self.fc(x_pool)
        # Split coefficients into two parts
        a1, a2 = theta.chunk(2, dim=1)
        
        # Apply coefficients
        a1 = a1.sigmoid() * 2 + 0.5  # Range: 0.5-2.5
        a2 = a2.sigmoid() * 2        # Range: 0-2
        
        if len(x_shape) == 2:
            a1 = a1.unsqueeze(1)
            a2 = a2.unsqueeze(1)
        else:
            a1 = a1.unsqueeze(1).expand(-1, x_shape[1], -1)
            a2 = a2.unsqueeze(1).expand(-1, x_shape[1], -1)
            
        # Dynamic ReLU function
        return torch.max(x * a1, x * 0 + a2)

def apply_dyrelu_phasing(model, current_epoch, start_epoch=0, end_epoch=75, layer_types=(nn.Linear,)):
    """
    Apply DyReLU phasing to model, gradually transitioning to ReLU
    
    Args:
        model: The model to modify
        current_epoch: Current training epoch
        start_epoch: Epoch to start phasing
        end_epoch: Epoch to complete phasing
        layer_types: Types of layers to apply phasing to
    """
    if current_epoch < start_epoch:
        # Before phasing: full DyReLU
        beta = 1.0
    elif current_epoch > end_epoch:
        # After phasing: full ReLU
        beta = 0.0
    else:
        # During phasing: linear interpolation
        beta = 1.0 - (current_epoch - start_epoch) / (end_epoch - start_epoch)
    
    print(f"DyReLU phasing: beta = {beta:.4f}")
    
    # Only modify if needed
    if beta == 0.0:
        # Replace all DyReLU with ReLU
        _replace_dyrelu_with_relu(model, layer_types)
    elif beta == 1.0:
        # Replace all ReLU with DyReLU
        _replace_relu_with_dyrelu(model, layer_types)
    else:
        # Apply mixed activation
        _apply_mixed_activation(model, beta, layer_types)
    
    return model

def _replace_relu_with_dyrelu(module, layer_types):
    """Replace ReLU activations with DyReLU"""
    for name, child in module.named_children():
        if isinstance(child, nn.ReLU):
            # Get input channels from previous layer if possible
            prev_layer = None
            for n, m in module.named_children():
                if isinstance(m, layer_types) and n != name:
                    prev_layer = m
                    break
            
            channels = 512  # Default fallback
            if prev_layer is not None and hasattr(prev_layer, 'out_features'):
                channels = prev_layer.out_features
            
            # Replace with DyReLU
            setattr(module, name, DyReLU(channels))
        else:
            _replace_relu_with_dyrelu(child, layer_types)

def _replace_dyrelu_with_relu(module, layer_types):
    """Replace DyReLU activations with ReLU"""
    for name, child in module.named_children():
        if isinstance(child, DyReLU):
            setattr(module, name, nn.ReLU(inplace=True))
        else:
            _replace_dyrelu_with_relu(child, layer_types)

def _apply_mixed_activation(module, beta, layer_types):
    """Apply mixed activation based on beta value"""
    for name, child in module.named_children():
        if isinstance(child, nn.ReLU) or isinstance(child, DyReLU):
            # Create a MixedActivation
            if isinstance(child, nn.ReLU):
                channels = 512  # Default fallback
                # Get input channels from previous layer if possible
                prev_layer = None
                for n, m in module.named_children():
                    if isinstance(m, layer_types) and n != name:
                        prev_layer = m
                        break
                
                if prev_layer is not None and hasattr(prev_layer, 'out_features'):
                    channels = prev_layer.out_features
                
                dyrelu = DyReLU(channels)
                relu = child
            else:  # DyReLU
                dyrelu = child
                relu = nn.ReLU(inplace=True)
            
            # Create mixed activation function
            class MixedActivation(nn.Module):
                def __init__(self, dyrelu, relu, beta):
                    super(MixedActivation, self).__init__()
                    self.dyrelu = dyrelu
                    self.relu = relu
                    self.beta = beta
                
                def forward(self, x):
                    if self.beta == 1.0:
                        return self.dyrelu(x)
                    elif self.beta == 0.0:
                        return self.relu(x)
                    else:
                        return self.beta * self.dyrelu(x) + (1 - self.beta) * self.relu(x)
            
            setattr(module, name, MixedActivation(dyrelu, relu, beta))
        else:
            _apply_mixed_activation(child, beta, layer_types)
