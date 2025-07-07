import torch
import os
import time
import numpy as np
import pickle
from torch.utils.data import DataLoader
from tqdm import tqdm

class Trainer:
    def __init__(self, model, optimizer, output_dir, training_dataset, valid_dataset, test_dataset, metrics_key,
                 epochs, batch_size, num_workers, batch_forward_func, batch_cal_loss_func, batch_metrics_func,
                 metrics_cal_func, collate_fn, device, train_dataset_sampler, valid_dataset_sampler, valid_step=1,
                 start_epoch=0, gradient_accumulate=1, lr_scheduler=None, max_grad_norm=None):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.output_dir = output_dir
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.batch_forward_func = batch_forward_func
        self.batch_cal_loss_func = batch_cal_loss_func
        self.batch_metrics_func = batch_metrics_func
        self.metrics_cal_func = metrics_cal_func
        self.collate_fn = collate_fn
        self.device = device
        self.valid_step = valid_step
        self.start_epoch = start_epoch
        self.gradient_accumulate = gradient_accumulate
        self.metrics_key = metrics_key
        self.best_metric = -float('inf')
        self.best_epoch = 0
        self.epoch_metrics = {}
        self.lr_scheduler = lr_scheduler
        self.max_grad_norm = max_grad_norm

        # Create output directory if not exists
        os.makedirs(self.output_dir, exist_ok=True)

        # Data loaders
        self.train_loader = DataLoader(
            training_dataset,
            batch_size=batch_size,
            sampler=train_dataset_sampler,
            collate_fn=collate_fn,
            num_workers=num_workers
        )

        self.valid_loader = DataLoader(
            valid_dataset,
            batch_size=batch_size,
            sampler=valid_dataset_sampler,
            collate_fn=collate_fn,
            num_workers=num_workers
        )

        self.test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            sampler=valid_dataset_sampler,
            collate_fn=collate_fn,
            num_workers=num_workers
        ) if test_dataset is not None else None

    def train(self):
        for epoch in range(self.start_epoch, self.epochs):
            start_time = time.time()
            print(f"\nEpoch {epoch + 1}/{self.epochs}")
            print("-'*50")

            # Training phase
            self.model.train()
            train_loss = 0.0
            train_preds = []
            train_labels = []

            train_pbar = tqdm(self.train_loader, desc=f"Training Epoch {epoch + 1}")
            for step, batch in enumerate(train_pbar):
                batch = {k: v.to(self.device) for k, v in batch.items()}

                outputs = self.batch_forward_func(self.model, batch)
                loss = self.batch_cal_loss_func(outputs, batch)
                loss = loss / self.gradient_accumulate

                loss.backward()
                train_loss += loss.item() * self.gradient_accumulate

                # Gradient accumulation
                if (step + 1) % self.gradient_accumulate == 0:
                    # Apply gradient clipping if max_grad_norm is set
                    if self.max_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    
                    # Step the learning rate scheduler if it exists
                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step()

                # Get predictions and labels for metrics
                preds, labels = self.batch_metrics_func(outputs, batch)
                train_preds.extend(preds)
                train_labels.extend(labels)

                train_pbar.set_postfix({'batch_loss': loss.item() * self.gradient_accumulate})

            # Ensure any remaining gradients are applied
            if (step + 1) % self.gradient_accumulate != 0:
                # Apply gradient clipping if max_grad_norm is set
                if self.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                # Step the learning rate scheduler if it exists and we haven't stepped in the last batch
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()

            # Calculate training metrics
            train_avg_loss = train_loss / len(self.train_loader)
            train_metrics = self.metrics_cal_func(train_preds, train_labels)
            train_metrics['loss'] = train_avg_loss

            # Log training results
            print(f"Training Results - Loss: {train_avg_loss:.4f}")
            for key, value in train_metrics.items():
                if key != 'loss':
                    print(f"{key}: {value:.4f}")

            # Validation phase
            if (epoch + 1) % self.valid_step == 0:
                valid_metrics = self.evaluate()
                self.epoch_metrics[epoch] = {'train': train_metrics, 'valid': valid_metrics}

                # Save best model
                if valid_metrics[self.metrics_key] > self.best_metric:
                    self.best_metric = valid_metrics[self.metrics_key]
                    self.best_epoch = epoch
                    self.save_model(epoch, is_best=True)
                else:
                    self.save_model(epoch, is_best=False)

            else:
                self.epoch_metrics[epoch] = {'train': train_metrics}

            # Print epoch time
            epoch_time = time.time() - start_time
            print(f"Epoch {epoch + 1} completed in {epoch_time:.2f} seconds")

        # Save metrics
        with open(os.path.join(self.output_dir, 'epoch_metrics.pkl'), 'wb') as f:
            pickle.dump(self.epoch_metrics, f)

        print(f"\nTraining completed! Best {self.metrics_key}: {self.best_metric:.4f} at epoch {self.best_epoch + 1}")

    def evaluate(self, test=False):
        self.model.eval()
        eval_loss = 0.0
        eval_preds = []
        eval_labels = []
        loader = self.test_loader if test else self.valid_loader
        desc = 'Testing' if test else 'Validation'

        with torch.no_grad():
            eval_pbar = tqdm(loader, desc=desc)
            for batch in eval_pbar:
                batch = {k: v.to(self.device) for k, v in batch.items()}

                outputs = self.batch_forward_func(self.model, batch)
                loss = self.batch_cal_loss_func(outputs, batch)
                eval_loss += loss.item()

                preds, labels = self.batch_metrics_func(outputs, batch)
                eval_preds.extend(preds)
                eval_labels.extend(labels)

                eval_pbar.set_postfix({'batch_loss': loss.item()})

        # Calculate evaluation metrics
        eval_avg_loss = eval_loss / len(loader)
        eval_metrics = self.metrics_cal_func(eval_preds, eval_labels)
        eval_metrics['loss'] = eval_avg_loss

        # Log evaluation results
        print(f"{desc} Results - Loss: {eval_avg_loss:.4f}")
        for key, value in eval_metrics.items():
            if key != 'loss':
                print(f"{key}: {value:.4f}")

        return eval_metrics

    def test(self):
        if self.test_loader is None:
            raise ValueError("No test dataset provided during initialization")
        return self.evaluate(test=True)

    def save_model(self, epoch, is_best=False):
        # Save model checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_metric': self.best_metric if is_best else None,
        }
        
        # Save lr_scheduler state if it exists
        if self.lr_scheduler is not None:
            checkpoint['lr_scheduler_state_dict'] = self.lr_scheduler.state_dict()

        # Create checkpoint path
        checkpoint_path = os.path.join(self.output_dir, f'checkpoint_epoch_{epoch + 1}.pth')
        torch.save(checkpoint, checkpoint_path)
        print(f"Model checkpoint saved to {checkpoint_path}")

        # Save best model separately
        if is_best:
            best_model_path = os.path.join(self.output_dir, 'best_model.pth')
            torch.save(checkpoint, best_model_path)
            print(f"Best model saved to {best_model_path}")

    def load_model(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load lr_scheduler state if it exists in the checkpoint and we have a scheduler
        if 'lr_scheduler_state_dict' in checkpoint and self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
            
        if 'epoch' in checkpoint:
            self.start_epoch = checkpoint['epoch'] + 1
        if 'best_metric' in checkpoint and checkpoint['best_metric'] is not None:
            self.best_metric = checkpoint['best_metric']
        print(f"Model loaded from {checkpoint_path}")

    def get_best_epoch(self):
        return self.best_epoch