# src/models/east.py
import torch
import torch.nn as nn
import math
from .layers import DyReLUB, WeightSharing


class EAST:
    def __init__(self, model, max_sparsity=0.9999, min_sparsity=0.99, 
                 cycle_length=350, num_cycles=3, update_freq=100):
        self.model = model
        self.max_sparsity = max_sparsity
        self.min_sparsity = min_sparsity
        self.cycle_length = cycle_length
        self.num_cycles = num_cycles
        self.update_freq = update_freq
        self.current_step = 0
        self.setup_dyrelu_phasing()
        self.setup_weight_sharing()

    def _phase_out_dyrelu(self, epoch, total_epochs):
        # Calculate phase out timing
        start_phase = 0.3  # Start phasing at 30% of training
        end_phase = 0.75   # Complete phase out at 75% of training
        
        # Calculate current phase
        if start_phase * total_epochs <= epoch <= end_phase * total_epochs:
            # Linear decay from 1 to 0 during the phasing period
            beta = 1.0 - ((epoch - start_phase * total_epochs) / 
                        ((end_phase - start_phase) * total_epochs))
            
            # Phase out DyReLU for each module
            for name, module in self.model.named_modules():
                if isinstance(module, DyReLUB):
                    # Gradually reduce DyReLU contribution
                    module.coefficient_generator.weight.data *= beta
                    module.coefficient_generator.bias.data *= beta


        
    def setup_dyrelu_phasing(self):
        for name, module in self.model.named_modules():
            if isinstance(module, nn.ReLU):
                channels = self._get_channels(module)
                if channels:
                    setattr(self.model, name, DyReLUB(channels))
    
    def setup_weight_sharing(self):
        for name, module in self.model.named_modules():
            if "layer" in name and len(name.split(".")) == 2:
                blocks = list(module.children())
                if len(blocks) > 2:
                    base_block = blocks[1]
                    for i in range(2, len(blocks)):
                        blocks[i] = WeightSharing(base_block)
    
    def _get_channels(self, module):
        for name, param in module.named_parameters():
            if param.dim() == 4:
                return param.size(1)
        return None

    def _get_cyclic_sparsity(self, step):
        cycle = (step % self.cycle_length) / self.cycle_length
        if step < self.num_cycles * self.cycle_length:
            return self.min_sparsity + (self.max_sparsity - self.min_sparsity) * \
                   0.5 * (1 + math.cos(2 * math.pi * cycle))
        return self.max_sparsity

    def _magnitude_prune(self, tensor, sparsity):
        threshold = torch.quantile(torch.abs(tensor), sparsity)
        return torch.where(torch.abs(tensor) > threshold, 1.0, 0.0)

    def _gradient_regrow(self, grad_tensor, current_mask, num_regrow):
        _, indices = torch.topk(torch.abs(grad_tensor) * (1 - current_mask), num_regrow)
        new_mask = current_mask.clone()
        new_mask.view(-1)[indices] = 1.0
        return new_mask

    def step(self, epoch, total_epochs):
        if self.current_step % self.update_freq != 0:
            self.current_step += 1
            return

        target_sparsity = self._get_cyclic_sparsity(self.current_step)
        
        if epoch > total_epochs * 0.3:
            self._phase_out_dyrelu(epoch, total_epochs)
            
        for name, param in self.model.named_parameters():
            if not hasattr(param, 'mask'):
                param.mask = torch.ones_like(param)
                
            current_sparsity = 1.0 - (param.mask.sum() / param.mask.numel())
            
            if target_sparsity > current_sparsity:
                param.mask = self._magnitude_prune(param, target_sparsity)
            else:
                num_regrow = int((target_sparsity - current_sparsity) * param.mask.numel())
                if num_regrow > 0:
                    param.mask = self._gradient_regrow(param.grad, param.mask, num_regrow)
            
            param.data *= param.mask
            
        self.current_step += 1
