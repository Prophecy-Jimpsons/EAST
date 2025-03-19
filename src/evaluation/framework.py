import torch
import numpy as np
import json
import os
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from tqdm import tqdm
from datetime import datetime
from src.sparsity.weight_sharing import enhanced_load_model

@dataclass
class ModelEvaluation:
    """Comprehensive model evaluation framework"""
    
    model_name: str
    model = None
    tokenizer = None
    device: str = "auto"
    results: Dict = field(default_factory=dict)
    
    def load_model(self, model_path=None):
        """Load model to evaluate"""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        # Use provided path or default to model_name
        path = model_path if model_path else self.model_name
        
        try:
            # Try enhanced loading first
            if os.path.exists(os.path.join(path, "east_metadata.json")):
                print(f"Loading EAST model from {path}")
                self.model, self.tokenizer = enhanced_load_model(path, device=self.device)
            else:
                # Fall back to standard loading
                print(f"Loading standard model from {path}")
                self.model = AutoModelForCausalLM.from_pretrained(
                    path,
                    torch_dtype=torch.bfloat16,
                    device_map=self.device,
                    low_cpu_mem_usage=True
                )
                self.tokenizer = AutoTokenizer.from_pretrained(path)
            
            return self.model, self.tokenizer
        except Exception as e:
            print(f"Error loading model: {e}")
            return None, None
    
    def measure_memory_usage(self):
        """Measure memory usage of the model"""
        if self.model is None:
            self.load_model()
            
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            memory_allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
            memory_reserved = torch.cuda.memory_reserved() / (1024 ** 3)    # GB
            return {
                "allocated_gb": memory_allocated,
                "reserved_gb": memory_reserved
            }
        else:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            return {
                "rss_gb": memory_info.rss / (1024 ** 3),
                "vms_gb": memory_info.vms / (1024 ** 3)
            }
    
    def calculate_perplexity(self, dataset_path=None, texts=None, max_samples=100):
        """Calculate perplexity on a dataset or provided texts"""
        if self.model is None:
            self.load_model()
        
        if dataset_path and os.path.exists(dataset_path):
            # Load dataset from file
            with open(dataset_path, 'r') as f:
                if dataset_path.endswith('.json'):
                    data = json.load(f)
                    if isinstance(data, list):
                        texts = data[:max_samples]
                    elif isinstance(data, dict) and 'text' in data:
                        texts = data['text'][:max_samples]
                else:
                    texts = [line.strip() for line in f.readlines()[:max_samples]]
        
        if not texts:
            print("No texts provided for perplexity calculation")
            return None
            
        # Calculate perplexity for each text
        perplexities = []
        for text in tqdm(texts, desc="Calculating perplexity"):
            try:
                encodings = self.tokenizer(text, return_tensors="pt").to(self.model.device)
                input_ids = encodings.input_ids
                
                # Calculate loss
                with torch.no_grad():
                    outputs = self.model(input_ids, labels=input_ids)
                    loss = outputs.loss
                
                # Calculate perplexity
                perplexity = torch.exp(loss).item()
                perplexities.append(perplexity)
            except Exception as e:
                print(f"Error calculating perplexity for text: {e}")
        
        # Calculate average perplexity
        avg_perplexity = np.mean(perplexities)
        perplexity_std = np.std(perplexities)
        
        self.results["perplexity"] = {
            "average": avg_perplexity,
            "std_dev": perplexity_std,
            "min": np.min(perplexities),
            "max": np.max(perplexities),
            "samples": len(perplexities)
        }
        
        print(f"Average Perplexity: {avg_perplexity:.4f} (± {perplexity_std:.4f})")
        return self.results["perplexity"]
    
    def run_benchmark_suite(self, benchmark_suite=None):
        """Run comprehensive benchmark suite"""
        if self.model is None:
            self.load_model()
            
        # Use default benchmark if none provided
        if benchmark_suite is None:
            benchmark_suite = {
                "code_completion": [
                    "def fibonacci(n):",
                    "class BinarySearchTree:",
                    "function quickSort(arr) {",
                    "def calculate_statistics(data_list):",
                    "async function fetchData() {"
                ],
                "text_generation": [
                    "Explain how neural networks work in simple terms.",
                    "Write a short story about a robot who discovers emotions.",
                    "Summarize the key principles of quantum mechanics.",
                    "Compare and contrast renewable and non-renewable energy sources.",
                    "Write a recipe for chocolate chip cookies."
                ],
                "reasoning": [
                    "If a train travels at 60 mph for 3 hours, how far does it go?",
                    "What is the next number in the sequence: 2, 4, 8, 16, __?",
                    "If all cats have tails, and Fluffy is a cat, what can we conclude?",
                    "A ball costs $1 more than a bat. Together they cost $11. How much does the ball cost?",
                    "What is the capital of France and what is its population?"
                ]
            }
            
        # Initialize results
        self.results["benchmark"] = {}
        
        # Run benchmarks for each category
        for category, prompts in benchmark_suite.items():
            print(f"\nRunning {category} benchmark...")
            
            category_results = {
                "inferences": [],
                "timing": [],
                "token_counts": []
            }
            
            for prompt in tqdm(prompts, desc=category):
                try:
                    # Time the inference
                    input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.model.device)
                    
                    # Warm-up run
                    with torch.no_grad():
                        _ = self.model.generate(input_ids, max_length=100)
                        
                    # Timed run
                    start_time = time.time()
                    with torch.no_grad():
                        output_ids = self.model.generate(
                            input_ids, 
                            max_length=200,
                            do_sample=True,
                            temperature=0.7,
                            top_p=0.95
                        )
                    end_time = time.time()
                    
                    # Decode output
                    output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
                    
                    # Store results
                    inference_time = end_time - start_time
                    token_count = len(output_ids[0])
                    
                    category_results["inferences"].append(output_text)
                    category_results["timing"].append(inference_time)
                    category_results["token_counts"].append(token_count)
                    
                    # Calculate tokens per second
                    tokens_per_second = (token_count - len(input_ids[0])) / inference_time
                    print(f"  - {tokens_per_second:.2f} tokens/sec, {inference_time:.2f}s total")
                    
                except Exception as e:
                    print(f"Error in benchmark for '{prompt[:30]}...': {e}")
                    category_results["inferences"].append(f"ERROR: {str(e)}")
                    category_results["timing"].append(None)
                    category_results["token_counts"].append(None)
            
            # Summarize category results
            valid_timings = [t for t in category_results["timing"] if t is not None]
            valid_token_counts = [c for c in category_results["token_counts"] if c is not None]
            
            if valid_timings:
                category_results["summary"] = {
                    "avg_time": np.mean(valid_timings),
                    "avg_tokens": np.mean(valid_token_counts) if valid_token_counts else None,
                    "avg_tokens_per_second": np.mean([
                        (tc - len(self.tokenizer.encode(p))) / t 
                        for p, tc, t in zip(
                            prompts, 
                            valid_token_counts, 
                            valid_timings
                        )
                    ]) if valid_timings and valid_token_counts else None
                }
            
            self.results["benchmark"][category] = category_results
        
        return self.results["benchmark"]
    
    def full_evaluation(self, benchmark_suite=None, perplexity_texts=None):
        """Run a full model evaluation"""
        print(f"\n=== Starting Full Evaluation of {self.model_name} ===\n")
        
        # Load model if not already loaded
        if self.model is None:
            self.model, self.tokenizer = self.load_model()
            if self.model is None:
                print("Failed to load model. Aborting evaluation.")
                return None
        
        # Measure memory usage
        print("\n--- Measuring Memory Usage ---")
        memory_stats = self.measure_memory_usage()
        self.results["memory"] = memory_stats
        print(f"Memory Usage: {memory_stats}")
        
        # Calculate perplexity
        if perplexity_texts:
            print("\n--- Calculating Perplexity ---")
            self.calculate_perplexity(texts=perplexity_texts)
        
        # Run benchmark suite
        print("\n--- Running Benchmark Suite ---")
        self.run_benchmark_suite(benchmark_suite)
        
        # Summarize results
        print("\n=== Evaluation Summary ===")
        
        print("\nMemory Usage:")
        for key, value in self.results["memory"].items():
            print(f"  {key}: {value:.4f} GB")
        
        if "perplexity" in self.results:
            print("\nPerplexity:")
            print(f"  Average: {self.results['perplexity']['average']:.4f}")
            print(f"  Std Dev: {self.results['perplexity']['std_dev']:.4f}")
        
        if "benchmark" in self.results:
            print("\nBenchmark Performance:")
            for category, results in self.results["benchmark"].items():
                if "summary" in results:
                    summary = results["summary"]
                    print(f"  {category}:")
                    print(f"    Avg Time: {summary['avg_time']:.4f}s")
                    print(f"    Avg Tokens: {summary['avg_tokens']:.1f}")
                    print(f"    Avg Tokens/Sec: {summary['avg_tokens_per_second']:.2f}")
        
        # Save results to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"evaluation_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"\nFull results saved to {results_file}")
        return self.results
