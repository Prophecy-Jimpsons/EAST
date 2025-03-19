import torch
import time

def test_model_inference(model, tokenizer, input_text="def fibonacci(n):"):
    """
    Test model inference with memory monitoring
    
    Args:
        model: The model to test
        tokenizer: The tokenizer to use
        input_text (str): Input text for generation
        
    Returns:
        tuple: (success, generated_text, generation_time)
    """
    from ..utils.memory_utils import check_system_resources
    
    # Record initial memory
    print("Before inference:")
    check_system_resources()
    
    try:
        start_time = time.time()
        
        # Generate with minimal memory settings and proper attention mask
        inputs = tokenizer(input_text, return_tensors="pt", padding=True).to(model.device)
        
        # Ensure pad_token_id is set properly
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_length=100,
                do_sample=True,
                temperature=0.7,
                pad_token_id=pad_token_id
            )
        
        generation_time = time.time() - start_time
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print("\nGenerated Output:")
        print(decoded)
        print(f"Generation time: {generation_time:.4f}s")
        
        # Check memory after inference
        print("\nAfter inference:")
        check_system_resources()
        
        return True, decoded, generation_time
    except Exception as e:
        print(f"Inference error: {e}")
        return False, None, None

def evaluate_model_performance(model, tokenizer, test_prompts=None):
    """
    Simple evaluation to compare performance before/after EAST
    
    Args:
        model: The model to evaluate
        tokenizer: The tokenizer to use
        test_prompts (list): List of prompts to test
        
    Returns:
        list: Evaluation results
    """
    if test_prompts is None:
        test_prompts = [
            "def sort_array(arr):",
            "class BinarySearchTree:",
            "# Function to calculate fibonacci sequence"
        ]
    
    results = []
    
    # Ensure pad_token_id is set properly once
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    
    for i, prompt in enumerate(test_prompts):
        print(f"\nEvaluating prompt {i+1}/{len(test_prompts)}: {prompt}")
        
        # Measure time
        start_time = time.time()
        
        # Process prompt with explicit padding and attention mask
        inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_length=50,
                num_return_sequences=1,
                pad_token_id=pad_token_id
            )
        
        generation_time = time.time() - start_time
        
        # Get output
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Calculate memory usage
        memory_usage = torch.cuda.memory_allocated() / 1e9  # GB
        
        result = {
            "prompt": prompt,
            "generation_time": generation_time,
            "memory_usage": memory_usage,
            "output_length": len(generated_text),
            "generated_text": generated_text
        }
        
        results.append(result)
        
        print(f"  Generation time: {generation_time:.4f}s")
        print(f"  Memory usage: {memory_usage:.4f}GB")
        print(f"  Output length: {len(generated_text)} chars")
    
    return results
