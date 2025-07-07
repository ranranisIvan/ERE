import torch
import torch.nn as nn
from torch.utils.data import Sampler
from transformers import AutoTokenizer, AutoModelForMaskedLM, AdamW
import random
import numpy as np

class MlmBaseModel(nn.Module):
    def __init__(self, mlm_type='bert-base-uncased'):
        super(MlmBaseModel, self).__init__()
        self.mlm_type = mlm_type
        self.tokenizer = AutoTokenizer.from_pretrained(mlm_type)
        self.model = AutoModelForMaskedLM.from_pretrained(mlm_type)
        self.config = self.model.config

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels
        )
        return outputs

class MlmBaseCollator:
    def __init__(self, tokenizer, max_seq_length=128, mlm_probability=0.15):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.mlm_probability = mlm_probability
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

            # Create masked labels
            labels = input_ids.copy()
            probability_matrix = torch.full(labels.shape, self.mlm_probability)
            special_tokens_mask = [1 if token in self.tokenizer.all_special_ids else 0 for token in labels]
            probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
            masked_indices = torch.bernoulli(probability_matrix).bool()
            labels[~masked_indices] = -100  # We only compute loss on masked tokens

            # Replace masked input tokens with mask token
            input_ids = np.array(input_ids)
            input_ids[masked_indices] = self.tokenizer.mask_token_id

            batch['input_ids'].append(input_ids.tolist())
            batch['attention_mask'].append(attention_mask)
            batch['token_type_ids'].append(token_type_ids)
            batch['labels'].append(labels)

        # Convert to tensors
        for key in batch:
            batch[key] = torch.tensor(batch[key])

        return batch

class MlmBaseSampler(Sampler):
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

def mlm_base_init_tokenizer(mlm_type, save_path=None):
    tokenizer = AutoTokenizer.from_pretrained(mlm_type)
    if save_path and not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
        tokenizer.save_pretrained(save_path)
    return tokenizer

# Preprocessing functions

def mlm_base_preprocess_data(data, tokenizer, max_seq_length=128):