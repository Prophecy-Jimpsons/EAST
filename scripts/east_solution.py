import os
import torch
from src.evaluation.component_verification import ComponentVerification
from src.evaluation.framework import ModelEvaluation
from src.sparsity.weight_sharing import apply_weight_sharing, enhanced_save_model
from src.sparsity.static_pruning import implement_adaptive_pruning


def east_deepseek_solution(model_name="deepseek-ai/deepseek-coder-1b-base", 
                          output_path="./east_deepseek_model",
                          prompt="def fibonacci(n):", 
                          share_ratio=0.5,
                          target_sparsity=0.5,
                          device="auto",
                          verify_components=True):
    """Complete end-to-end solution for EAST-optimized DeepSeek with verification"""
    print(f"\n=== EAST DeepSeek Solution ===")
    print(f"Model: {model_name}")
    print(f"Share Ratio: {share_ratio}")
    print(f"Target Sparsity: {target_sparsity}")
    
    # Step 1: Load model and tokenizer
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print(f"\nLoading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
        low_cpu_mem_usage=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create intermediate save path
    os.makedirs(output_path, exist_ok=True)
    
    # Step 2: Verification framework setup
    if verify_components:
        verifier = ComponentVerification(model_name, device)
    
    # Step 3: Apply weight sharing with verification
    print(f"\nApplying weight sharing...")
    source_layer_idx = 11
    max_layers = len(model.model.layers) if hasattr(model, 'model') and hasattr(model.model, 'layers') else 0
    end_layer_idx = int(max_layers * (1 - share_ratio))
    
    if verify_components:
        print("Running verification on weight sharing component...")
        sharing_results = verifier.test_weight_sharing(
            source_layer_idx=source_layer_idx, 
            target_layer_range=(source_layer_idx + 1, end_layer_idx)
        )
        
        # Check if weight sharing is working correctly
        if sharing_results["after_sharing"]["perplexity"] > sharing_results["baseline"]["perplexity"] * 1.5:
            print("WARNING: Weight sharing caused significant perplexity increase. Proceeding with caution.")
    
    # Apply weight sharing to full model
    model = apply_weight_sharing(
        model, 
        source_layer_idx=source_layer_idx, 
        target_layer_range=(source_layer_idx + 1, end_layer_idx)
    )
    
    # Intermediate save with only weight sharing
    sharing_path = os.path.join(output_path, "sharing_only")
    enhanced_save_model(model, sharing_path, tokenizer)
    
    # Step 4: Apply adaptive pruning with verification
    print(f"\nApplying adaptive pruning...")
    if verify_components:
        print("Running verification on adaptive pruning component...")
        pruning_results = verifier.test_adaptive_pruning(target_sparsity=target_sparsity)
        
        # Check if pruning is working correctly
        if pruning_results["after_pruning"]["perplexity"] > pruning_results["baseline"]["perplexity"] * 2.0:
            print("WARNING: Pruning caused significant perplexity increase. Reducing sparsity level.")
            target_sparsity = max(0.3, target_sparsity - 0.2)
            print(f"Adjusted target sparsity to {target_sparsity}")
    
    # Apply adaptive pruning to full model
    model = implement_adaptive_pruning(model, target_sparsity=target_sparsity)
    
    # Step 5: Save optimized model
    print(f"\nSaving EAST-optimized model to {output_path}...")
    enhanced_save_model(model, output_path, tokenizer)
    
    # Step 6: Evaluate the optimized model
    print(f"\nEvaluating optimized model...")
    evaluation = ModelEvaluation(output_path, device=device)
    base_eval_results = evaluation.full_evaluation(
        perplexity_texts=[
            "def fibonacci(n):",
            "function findSum(a, b) {",
            "Explain how neural networks work in simple terms."
        ]
    )
    
    # Step 7: Generate with robust settings
    print(f"\nGenerating response to prompt: {prompt}")
    generated_text = generate_with_robust_attention(model, tokenizer, prompt)
    
    print(f"\nGenerated text:\n{generated_text}")
    
    # Step 8: Clean up resources
    print(f"\nCleaning up resources...")
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return generated_text, output_path, base_eval_results

def generate_with_robust_attention(model, tokenizer, prompt, max_length=100):
    """Generate text with proper attention mask handling"""
    # Ensure pad token is properly set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Properly encode the prompt with explicit padding
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    
    # Move to model device
    input_ids = inputs.input_ids.to(model.device)
    attention_mask = inputs.attention_mask.to(model.device)
    
    # Generate with robust settings and proper attention mask
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,  # Explicit attention mask
            max_length=max_length,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            top_k=50,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
            num_return_sequences=1
        )
    
    # Decode output properly
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

# Main execution
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="EAST DeepSeek Solution")
    parser.add_argument("--model", type=str, default="deepseek-ai/deepseek-coder-1b-base", 
                       help="Model name or path")
    parser.add_argument("--output", type=str, default="./east_deepseek_model", 
                       help="Output model path")
    parser.add_argument("--prompt", type=str, default="def fibonacci(n):", 
                       help="Prompt for text generation")
    parser.add_argument("--share-ratio", type=float, default=0.5, 
                       help="Ratio of layers to share (0.0-1.0)")
    parser.add_argument("--sparsity", type=float, default=0.5, 
                       help="Target sparsity level (0.0-1.0)")
    parser.add_argument("--device", type=str, default="auto", 
                       help="Device for model computation (auto, cpu, cuda:0, etc.)")
    parser.add_argument("--verify", action="store_true", 
                       help="Enable component verification")
    
    args = parser.parse_args()
    
    generated_text, model_path, eval_results = east_deepseek_solution(
        model_name=args.model,
        output_path=args.output,
        prompt=args.prompt,
        share_ratio=args.share_ratio,
        target_sparsity=args.sparsity,
        device=args.device,
        verify_components=args.verify
    )
    
    print(f"\nEAST DeepSeek Solution completed successfully.")
    print(f"Optimized model saved to: {model_path}")
    print(f"\nSample output for '{args.prompt}':\n{generated_text[:500]}")
