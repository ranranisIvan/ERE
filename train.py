#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
import logging
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from typing import Dict, Any
import argparse
from tqdm import tqdm
from ecimp import (
    ecimp_init_tokenizer,
    ecimp_preprocess_data,
    ECIMPCollator,
    ECIMPModel,
    batch_cal_loss_func,
    batch_metrics_func,
    batch_forward_func,
    metrics_cal_func,
    get_optimizer
)
from transformers import get_linear_schedule_with_warmup

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_dataset(file_path: str) -> Dict[str, Any]:
    """Load dataset from JSON file"""
    logger.info(f"Loading dataset from {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def train_epoch(model, train_loader, optimizer, scheduler, device, max_grad_norm, gradient_accumulation_steps):
    """Run one training epoch"""
    model.train()
    total_loss = 0
    metrics = {}
    
    with tqdm(total=len(train_loader), desc="Training") as pbar:
        for batch_idx, batch in enumerate(train_loader):
            # Forward pass
            labels, preds = batch_forward_func(batch, None)
            loss = batch_cal_loss_func(labels, preds, None)
            
            # Backward pass
            loss = loss / gradient_accumulation_steps
            loss.backward()
            total_loss += loss.item()
            
            # Update metrics
            metrics, batch_metrics = batch_metrics_func(labels, preds, metrics, None)
            
            # Gradient accumulation
            if (batch_idx + 1) % gradient_accumulation_steps == 0 or batch_idx == len(train_loader) - 1:
                if max_grad_norm:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            # Update progress bar
            pbar.update(1)
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
            })
    
    return total_loss / len(train_loader), metrics_cal_func(metrics)

def evaluate(model, eval_loader, device):
    """Evaluate model on validation set"""
    model.eval()
    total_loss = 0
    metrics = {}
    
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating"):
            labels, preds = batch_forward_func(batch, None)
            loss = batch_cal_loss_func(labels, preds, None)
            total_loss += loss.item()
            metrics, _ = batch_metrics_func(labels, preds, metrics, None)
    
    return total_loss / len(eval_loader), metrics_cal_func(metrics)

def main():
    parser = argparse.ArgumentParser()
    # Model parameters
    parser.add_argument("--model_name", type=str, default="roberta-base")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    
    # Data parameters
    parser.add_argument("--maven_ere_train", type=str, default="data/processed/maven_ere_train.json")
    parser.add_argument("--maven_ere_dev", type=str, default="data/processed/maven_ere_dev.json")
    parser.add_argument("--eventstoryline", type=str, default="data/processed/eventstoryline.json")
    parser.add_argument("--causaltimebank", type=str, default="data/processed/causaltimebank.json")
    
    # Training parameters
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_workers", type=int, default=4)
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize model and tokenizer
    tokenizer = ecimp_init_tokenizer(args.model_name)
    model = ECIMPModel(
        args.model_name,
        use_event_prompt=True,
        use_signal_prompt=True,
        use_linear=False
    ).to(device)
    
    # Load and preprocess datasets
    train_data = []
    for data_path in [args.maven_ere_train, args.eventstoryline, args.causaltimebank]:
        if os.path.exists(data_path):
            train_data.extend(ecimp_preprocess_data(load_dataset(data_path), tokenizer))
    
    dev_data = []
    if os.path.exists(args.maven_ere_dev):
        dev_data.extend(ecimp_preprocess_data(load_dataset(args.maven_ere_dev), tokenizer))
    
    logger.info(f"Loaded {len(train_data)} training examples and {len(dev_data)} validation examples")
    
    # Create dataloaders
    collator = ECIMPCollator(tokenizer, use_event_prompt=True, use_signal_prompt=True)
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        sampler=RandomSampler(train_data),
        num_workers=args.num_workers,
        collate_fn=collator
    )
    
    dev_loader = DataLoader(
        dev_data,
        batch_size=args.batch_size,
        sampler=SequentialSampler(dev_data),
        num_workers=args.num_workers,
        collate_fn=collator
    ) if dev_data else None
    
    # Initialize optimizer and scheduler
    optimizer = get_optimizer(model, args.learning_rate)
    total_steps = len(train_loader) * args.epochs // args.gradient_accumulation_steps
    warmup_steps = min(args.warmup_steps, total_steps // 10)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Training loop
    best_metric = 0
    for epoch in range(args.epochs):
        logger.info(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        # Train
        train_loss, train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler,
            device, args.max_grad_norm, args.gradient_accumulation_steps
        )
        logger.info(f"Training loss: {train_loss:.4f}")
        for k, v in train_metrics.items():
            logger.info(f"Training {k}: {v}")
        
        # Evaluate
        if dev_loader:
            val_loss, val_metrics = evaluate(model, dev_loader, device)
            logger.info(f"Validation loss: {val_loss:.4f}")
            for k, v in val_metrics.items():
                logger.info(f"Validation {k}: {v}")
            
            # Save best model
            if val_metrics.get('f1', 0) > best_metric:
                best_metric = val_metrics['f1']
                torch.save(
                    model.state_dict(),
                    os.path.join(args.output_dir, "best_model.pt")
                )
        
        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }, os.path.join(args.output_dir, f"checkpoint_epoch_{epoch+1}.pt"))

if __name__ == "__main__":
    main() 