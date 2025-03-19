import torch
import psutil

def check_system_resources():
    """
    Monitor and print current system resources (RAM and GPU memory)
    
    Returns:
        dict: Dictionary containing memory usage information
    """
    # System RAM
    ram = psutil.virtual_memory()
    print(f"System RAM: {ram.used/1e9:.2f}GB used / {ram.total/1e9:.2f}GB total")
    
    memory_info = {
        "ram_used_gb": ram.used/1e9,
        "ram_total_gb": ram.total/1e9,
        "ram_percent": ram.percent
    }
    
    # GPU resources
    if torch.cuda.is_available():
        gpu_info = []
        for i in range(torch.cuda.device_count()):
            device = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {device.name}")
            print(f"VRAM: {torch.cuda.memory_allocated(i)/1e9:.2f}GB allocated / {device.total_memory/1e9:.2f}GB total")
            
            gpu_info.append({
                "id": i,
                "name": device.name,
                "vram_allocated_gb": torch.cuda.memory_allocated(i)/1e9,
                "vram_total_gb": device.total_memory/1e9,
                "vram_percent": torch.cuda.memory_allocated(i) / device.total_memory * 100
            })
        
        memory_info["gpu_info"] = gpu_info
    else:
        print("No GPU available")
        memory_info["gpu_info"] = None
        
    return memory_info

def clean_gpu_memory():
    """
    Clean up GPU memory by forcing garbage collection and emptying CUDA cache
    """
    import gc
    
    # Force garbage collection
    gc.collect()
    
    # Empty CUDA cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("GPU memory cache cleared")
