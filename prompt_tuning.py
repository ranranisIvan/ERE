import torch
import torch.nn as nn
from torch.utils.data import Sampler
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
import random
import numpy as np
import os

class PromptTuningModel(nn.Module):
    def __init__(self, model_type='bert-base-uncased', num_labels=2, prompt_length=10):
        super(PromptTuningModel, self).__init__()
        self.model_type = model_type
        self.prompt_length = prompt_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_type)
        self.base_model = AutoModelForSequenceClassification.from_pretrained(model_type, num_labels=num_labels)
        self.config = self.base_model.config

        # Initialize prompt embeddings
        self.prompt_embeddings = nn.Embedding(prompt_length, self.config.hidden_size)
        # Freeze base model parameters
        for param in self.base_model.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        # Get base model embeddings
        batch_size = input_ids.size(0)
        inputs_embeds = self.base_model.get_input_embeddings()(input_ids)

        # Insert prompt embeddings at the beginning
        prompt_embeds = self.prompt_embeddings.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        inputs_embeds = torch.cat([prompt_embeds, inputs_embeds], dim=1)

        # Adjust attention mask and token type ids
        if attention_mask is not None:
            prompt_attention_mask = torch.ones(batch_size, self.prompt_length, device=attention_mask.device)
            attention_mask = torch.cat([prompt_attention_mask, attention_mask], dim=1)

        if token_type_ids is not None:
            prompt_token_type_ids = torch.zeros(batch_size, self.prompt_length, device=token_type_ids.device)
            token_type_ids = torch.cat([prompt_token_type_ids, token_type_ids], dim=1)

        # Forward pass through base model
        outputs = self.base_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels
        )
        return outputs

class PromptTuningCollator:
    def __init__(self, tokenizer, max_seq_length=128):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, examples):
        batch = {
            'input_ids': [],
            'attention_mask': [],
            'token_type_ids': [],
            'labels': []
        }

        for example in examples:
            input_ids = example['input_ids']
            attention_mask = example['attention_mask']
            token_type_ids = example.get('token_type_ids', [0]*len(input_ids))
            labels = example.get('label', 0)

            # Truncate or pad sequences
            if len(input_ids) > self.max_seq_length:
                input_ids = input_ids[:self.max_seq_length]
                attention_mask = attention_mask[:self.max_seq_length]
                token_type_ids = token_type_ids[:self.max_seq_length]
            else:
                padding_length = self.max_seq_length - len(input_ids)
                input_ids += [self.pad_token_id] * padding_length
                attention_mask += [0] * padding_length
                token_type_ids += [0] * padding_length

            batch['input_ids'].append(input_ids)
            batch['attention_mask'].append(attention_mask)
            batch['token_type_ids'].append(token_type_ids)
            batch['labels'].append(labels)

        # Convert to tensors
        for key in batch:
            batch[key] = torch.tensor(batch[key])

        return batch

class PromptTuningSampler(Sampler):
    def __init__(self, data_source, shuffle=True):
        self.data_source = data_source
        self.shuffle = shuffle
        self.epoch = 0

    def __iter__(self):
        indices = list(range(len(self.data_source)))
        if self.shuffle:
            # Shuffle with a seed based on epoch
            random.seed(self.epoch)
            random.shuffle(indices)
        return iter(indices)

    def __len__(self):
        return len(self.data_source)

    def set_epoch(self, epoch):
        self.epoch = epoch

# Initialization functions

def prompt_tuning_init_tokenizer(model_type, save_path=None):
    tokenizer = AutoTokenizer.from_pretrained(model_type)
    if save_path and not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
        tokenizer.save_pretrained(save_path)
    return tokenizer

# Preprocessing functions

def prompt_tuning_preprocess_data(data, tokenizer, max_seq_length=128):
    """Preprocess data for Prompt-Tuning task"""
    examples = []
    text = data.get('text', '')
    label = data.get('label', 0)

    if not text:
        return examples

    # Tokenize text
    encoding = tokenizer(
        text,
        truncation=True,
        max_length=max_seq_length,
        padding=False,
        return_attention_mask=True,
        return_token_type_ids=True
    )

    examples.append({
        'input_ids': encoding['input_ids'],
        'attention_mask': encoding['attention_mask'],
        'token_type_ids': encoding['token_type_ids'],
        'label': label
    })

    return examples

# Utility functions

def prompt_tuning_get_optimizer(model, learning_rate=2e-5):
    """Get optimizer for Prompt-Tuning model"""
    # Only optimize prompt embeddings
    return AdamW(model.prompt_embeddings.parameters(), lr=learning_rate)

def prompt_tuning_batch_forward_func(model, batch):
    """Forward pass function for batch processing"""
    return model(
        input_ids=batch['input_ids'],
        attention_mask=batch['attention_mask'],
        token_type_ids=batch['token_type_ids'],
        labels=batch['labels']
    )

def prompt_tuning_batch_cal_loss_func(outputs, batch):
    """Calculate loss for batch"""
    return outputs.loss

def prompt_tuning_batch_metrics_func(outputs, batch):
    """Calculate metrics for batch"""
    logits = outputs.logits
    preds = torch.argmax(logits, dim=-1)
    labels = batch['labels']
    return preds.cpu().numpy(), labels.cpu().numpy()

def prompt_tuning_metrics_cal_func(preds, labels):
    """Calculate overall metrics"""
    from sklearn.metrics import accuracy_score, f1_score

    flat_preds = []
    flat_labels = []

    for p, l in zip(preds, labels):
        flat_preds.extend(p.tolist() if isinstance(p, np.ndarray) else [p])
        flat_labels.extend(l.tolist() if isinstance(l, np.ndarray) else [l])

    # Calculate accuracy and F1 score
    accuracy = accuracy_score(flat_labels, flat_preds)
    f1 = f1_score(flat_labels, flat_preds, average='weighted')

    return {'accuracy': accuracy, 'f1': f1}