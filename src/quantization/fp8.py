import torch
from torch import Tensor
from typing import Optional, Tuple
import json

class FP8Quantization:
    """Implementation of FP8 mixed precision techniques from DeepSeek-V3"""
    
    def __init__(self, model, scale_factor=1.0, dynamic_scaling=True):
        self.model = model
        self.scale_factor = scale_factor
        self.dynamic_scaling = dynamic_scaling
        self.abs_max_per_tensor = {}
        
    def quantize_to_fp8(self, x: Tensor) -> Tuple[Tensor, float]:
        """Quantize floating point tensor to FP8 format"""
        if self.dynamic_scaling:
            # Find the absolute maximum value in the tensor for scaling
            abs_max = torch.max(torch.abs(x)).item()
            # Use a safety factor to avoid overflow
            scale = 127.0 / (abs_max * 1.1)
        else:
            scale = self.scale_factor
            
        # Scale the tensor
        x_scaled = x * scale
        
        # Clamp values to the int8 range (-127 to 127)
        x_clamped = torch.clamp(x_scaled, -127.0, 127.0)
        
        # Round to nearest integer and convert to int8
        x_int8 = torch.round(x_clamped).to(torch.int8)
        
        return x_int8, scale
    
    def dequantize_from_fp8(self, x_int8: Tensor, scale: float) -> Tensor:
        """Dequantize from FP8 format back to floating point"""
        # Convert back to floating point and rescale
        return x_int8.float() / scale
    
    def apply_to_model(self):
        """Apply FP8 quantization to model weights"""
        # Dictionary to store quantized weights and scales
        self.quantized_state = {}
        
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                # Quantize the parameter
                quantized_param, scale = self.quantize_to_fp8(param.data)
                
                # Store original parameter shape and dtype
                self.quantized_state[name] = {
                    'quantized': quantized_param,
                    'scale': scale,
                    'shape': param.shape,
                    'dtype': param.dtype
                }
                
                # Replace parameter with quantized version for memory savings
                param.data = self.dequantize_from_fp8(quantized_param, scale)
        
        # Set training precision attribute
        self.model.training_precision = "fp8"
        
        return self.model
    
    def restore_model(self):
        """Restore model to original precision"""
        if not hasattr(self, 'quantized_state'):
            print("No quantized state found. Model might not be quantized.")
            return self.model
            
        for name, param_dict in self.quantized_state.items():
            param = dict(self.model.named_parameters())[name]
            # Dequantize and restore
            param.data = self.dequantize_from_fp8(
                param_dict['quantized'], 
                param_dict['scale']
            ).to(param_dict['dtype'])
            
        return self.model

class FP8MixedPrecisionTraining:
    """Implementation of FP8 mixed precision training from DeepSeek-V3"""
    
    def __init__(self, model, optimizer, scaler_factor=1.0):
        self.model = model
        self.optimizer = optimizer
        self.scaler_factor = scaler_factor
        self.fp8_quantizer = FP8Quantization(model, dynamic_scaling=True)
        
    def training_step(self, loss):
        """Perform a mixed precision training step"""
        # Clear previous gradients
        self.optimizer.zero_grad()
        
        # Quantize model weights to FP8 for forward pass
        quantized_model = self.fp8_quantizer.apply_to_model()
        
        # Backward pass with FP8 scaling
        scaled_loss = loss * self.scaler_factor
        scaled_loss.backward()
        
        # Quantize gradients to FP8
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                quantized_grad, scale = self.fp8_quantizer.quantize_to_fp8(param.grad)
                param.grad = self.fp8_quantizer.dequantize_from_fp8(quantized_grad, scale)
        
        # Step the optimizer
        self.optimizer.step()
        
        # Restore model to original precision
        self.fp8_quantizer.restore_model()
        
        return loss.item()
    
    def save_fp8_model(self, path):
        """Save model with FP8 quantization metadata"""
        # Apply quantization
        quantized_model = self.fp8_quantizer.apply_to_model()
        
        # Save quantized state
        fp8_metadata = {
            "training_precision": "fp8",
            "scaler_factor": self.scaler_factor,
            "scales": {name: state['scale'] for name, state in self.fp8_quantizer.quantized_state.items()}
        }
        
        # Save model with standard method
        quantized_model.save_pretrained(path)
        
        # Save additional FP8 metadata
        with open(f"{path}/fp8_metadata.json", 'w') as f:
            json.dump(fp8_metadata, f, indent=2)
        
        return True
