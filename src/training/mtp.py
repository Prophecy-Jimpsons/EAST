from transformers import Trainer, TrainingArguments
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
import torch 
import json 

class DeepSeekMTPTrainer:
    """Implementation of DeepSeek-V3's Multi-Token Prediction technique"""
    
    def __init__(self, model, tokenizer, device="auto"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.mtp_prediction_length = 4  # Number of tokens to predict ahead
    
    def prepare_mtp_dataset(self, texts, max_length=512):
        """Prepare dataset for Multi-Token Prediction training"""
        dataset = []
        
        for text in texts:
            # Tokenize the text
            encodings = self.tokenizer(text, return_tensors="pt", truncation=True, 
                                       max_length=max_length - self.mtp_prediction_length)
            input_ids = encodings.input_ids[0]
            
            # Create input-output pairs for each position
            for idx in range(len(input_ids) - self.mtp_prediction_length):
                # Input: tokens up to position idx
                input_slice = input_ids[:idx+1].clone()
                
                # Target: Next mtp_prediction_length tokens
                target_slice = input_ids[idx+1:idx+1+self.mtp_prediction_length].clone()
                
                if len(target_slice) == self.mtp_prediction_length:
                    dataset.append({
                        "input_ids": input_slice,
                        "target_ids": target_slice
                    })
        
        return dataset
    
    def mtp_loss_function(self, logits, targets):
        """Custom loss function for Multi-Token Prediction"""
        # Reshape logits to match target shape
        batch_size = targets.size(0)
        vocab_size = logits.size(-1)
        
        # Calculate cross-entropy loss for each prediction position
        loss = 0
        for i in range(self.mtp_prediction_length):
            target_i = targets[:, i]
            
            # Use predicted token from previous position as input for next position
            if i == 0:
                # First token prediction uses original logits
                logits_i = logits
            else:
                # For subsequent tokens, use previous prediction
                prev_pred = F.softmax(logits, dim=-1).argmax(dim=-1)
                # Get embeddings for predicted tokens and predict next token
                # This is a simplified version; in practice would need model-specific implementation
                # to feed predictions back through the model
                logits_i = self.model(prev_pred).logits
            
            loss_i = F.cross_entropy(logits_i.view(-1, vocab_size), target_i.view(-1))
            loss += loss_i
        
        return loss / self.mtp_prediction_length
    
    def train_with_mtp(self, train_texts, eval_texts=None, 
                      output_dir="./mtp_model", 
                      num_train_epochs=3,
                      per_device_train_batch_size=8,
                      learning_rate=5e-5):
        """Train model using Multi-Token Prediction objective"""
        
        # Prepare datasets
        train_dataset = self.prepare_mtp_dataset(train_texts)
        eval_dataset = self.prepare_mtp_dataset(eval_texts) if eval_texts else None
        
        # Define training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_train_batch_size,
            logging_dir=f"{output_dir}/logs",
            logging_steps=100,
            save_steps=1000,
            evaluation_strategy="epoch" if eval_dataset else "no",
            learning_rate=learning_rate,
            weight_decay=0.01,
            fp16=True if self.model.dtype == torch.float16 else False,
            bf16=True if self.model.dtype == torch.bfloat16 else False,
        )
        
        # Custom training loop with MTP objective
        class MTPTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False):
                input_ids = inputs["input_ids"].to(model.device)
                target_ids = inputs["target_ids"].to(model.device)
                
                outputs = model(input_ids)
                logits = outputs.logits[:, -1, :]  # Get logits for last position
                
                # Use custom MTP loss
                loss = self.mtp_loss_function(logits, target_ids)
                
                return (loss, outputs) if return_outputs else loss
        
        # Bind the loss function to the instance
        MTPTrainer.mtp_loss_function = self.mtp_loss_function
        
        # Create trainer and train
        trainer = MTPTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )
        
        trainer.train()
        
        # Save the model with MTP capabilities
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        # Create metadata for MTP
        mtp_metadata = {
            "mtp_enabled": True,
            "mtp_prediction_length": self.mtp_prediction_length,
            "training_details": {
                "epochs": num_train_epochs,
                "batch_size": per_device_train_batch_size,
                "learning_rate": learning_rate
            }
        }
        
        with open(f"{output_dir}/mtp_metadata.json", 'w') as f:
            json.dump(mtp_metadata, f, indent=2)
        
        return self.model, self.tokenizer
