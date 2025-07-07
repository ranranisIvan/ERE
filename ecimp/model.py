import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel, RobertaTokenizer
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

class PolaLinearAttention(nn.Module):
    """
    Polarity-aware Linear Attention mechanism for feature fusion
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Projections for Q, K, V
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        
        # Output projection
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # Scaling factor for dot products
        self.scale = hidden_size ** -0.5

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: Tensor of shape [batch_size, num_tasks, hidden_size]
                     Contains features from different tasks (ECI, CWD, CED)
        Returns:
            fused_features: Tensor of shape [batch_size, num_tasks, hidden_size]
        """
        # Apply layer norm first
        features = self.layer_norm(features)
        
        # Project to Q, K, V
        Q = self.q_proj(features)  # [batch_size, num_tasks, hidden_size]
        K = self.k_proj(features)  # [batch_size, num_tasks, hidden_size]
        V = self.v_proj(features)  # [batch_size, num_tasks, hidden_size]
        
        # Split into positive and negative channels
        Q_pos = F.relu(Q)
        Q_neg = F.relu(-Q)
        K_pos = F.relu(K)
        K_neg = F.relu(-K)
        
        # Compute attention scores for both channels
        # Positive channel
        attn_pos = torch.bmm(Q_pos, K_pos.transpose(-2, -1)) * self.scale  # [batch_size, num_tasks, num_tasks]
        attn_pos = F.softmax(attn_pos, dim=-1)
        
        # Negative channel
        attn_neg = torch.bmm(Q_neg, K_neg.transpose(-2, -1)) * self.scale  # [batch_size, num_tasks, num_tasks]
        attn_neg = F.softmax(attn_neg, dim=-1)
        
        # Apply attention to values
        out_pos = torch.bmm(attn_pos, V)  # [batch_size, num_tasks, hidden_size]
        out_neg = torch.bmm(attn_neg, V)  # [batch_size, num_tasks, hidden_size]
        
        # Combine channels
        out = out_pos - out_neg
        
        # Final projection
        out = self.out_proj(out)
        
        # Residual connection
        return features + out

class ECIMPModel(nn.Module):
    def __init__(
        self,
        pretrained_model_name: str,
        use_event_prompt: bool = True,
        use_signal_prompt: bool = True,
        use_linear: bool = False,
        use_sep_gate: bool = False,
        use_mask1_gate: bool = False
    ):
        super().__init__()
        
        # Load pretrained RoBERTa
        self.roberta = RobertaModel.from_pretrained(pretrained_model_name)
        self.hidden_size = self.roberta.config.hidden_size
        
        # Feature fusion using PolaLinearAttention
        self.feature_fusion = PolaLinearAttention(self.hidden_size)
        
        # Task-specific heads
        self.eci_head = nn.Linear(self.hidden_size, 3)  # 3 classes: Cause, CausedBy, NA
        self.cwd_head = nn.Linear(self.hidden_size, 2)  # Binary: Has explicit signal or not
        self.ced_head = nn.Linear(self.hidden_size, 2)  # Binary: Has implicit signal or not
        
        # Configuration flags
        self.use_event_prompt = use_event_prompt
        self.use_signal_prompt = use_signal_prompt
        self.use_linear = use_linear
        self.use_sep_gate = use_sep_gate
        self.use_mask1_gate = use_mask1_gate

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        mask_pos: Optional[torch.Tensor] = None,
        sep_pos: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            token_type_ids: Token type IDs (不用于RoBERTa，可以为None)
            mask_pos: Position of the [MASK] token
            sep_pos: Position of the [SEP] token
            
        Returns:
            Tuple containing logits for ECI, CWD, and CED tasks
        """
        # Get RoBERTa outputs - RoBERTa不使用token_type_ids
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            # token_type_ids参数在RoBERTa中不使用，忽略它
            return_dict=True
        )
        
        sequence_output = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        
        # Extract features for each task
        batch_size = sequence_output.size(0)
        
        # Get [MASK] token features for ECI task
        mask_pos = mask_pos.unsqueeze(1).unsqueeze(2).expand(-1, -1, self.hidden_size)
        h_eci = torch.gather(sequence_output, 1, mask_pos).squeeze(1)  # [batch_size, hidden_size]
        
        # Apply mask gating if enabled
        if self.use_mask1_gate:
            # Create a simple gating mechanism using sigmoid activation
            mask_gate = torch.sigmoid(self.eci_head(h_eci)).unsqueeze(1)  # [batch_size, 1, 3]
            # Apply the gate - scale features based on predicted importance
            h_eci = h_eci * mask_gate[:, 0, 0].unsqueeze(1)  # Use the first dimension of the gate
        
        # Get [SEP] token features for CWD and CED tasks
        sep_pos = sep_pos.unsqueeze(1).unsqueeze(2).expand(-1, -1, self.hidden_size)
        h_sep = torch.gather(sequence_output, 1, sep_pos).squeeze(1)  # [batch_size, hidden_size]
        
        # Apply sep gating if enabled
        if self.use_sep_gate:
            # Create a simple gating mechanism using sigmoid activation
            sep_gate = torch.sigmoid(self.cwd_head(h_sep)).unsqueeze(1)  # [batch_size, 1, 2]
            # Apply the gate - scale features based on predicted importance
            h_sep = h_sep * sep_gate[:, 0, 0].unsqueeze(1)  # Use the first dimension of the gate
        
        # Stack features for fusion
        task_features = torch.stack([h_eci, h_sep, h_sep], dim=1)  # [batch_size, 3, hidden_size]
        
        # Apply feature fusion
        fused_features = self.feature_fusion(task_features)  # [batch_size, 3, hidden_size]
        
        # Extract task-specific features after fusion
        h_eci_fused = fused_features[:, 0]  # [batch_size, hidden_size]
        h_cwd_fused = fused_features[:, 1]  # [batch_size, hidden_size]
        h_ced_fused = fused_features[:, 2]  # [batch_size, hidden_size]
        
        # Get predictions for each task
        eci_logits = self.eci_head(h_eci_fused)  # [batch_size, 3]
        cwd_logits = self.cwd_head(h_cwd_fused)  # [batch_size, 2]
        ced_logits = self.ced_head(h_ced_fused)  # [batch_size, 2]
        
        return eci_logits, cwd_logits, ced_logits

# Helper functions
def ecimp_init_tokenizer(mlm_type: str, cache_dir: str) -> RobertaTokenizer:
    print(f"初始化tokenizer: {mlm_type}, 缓存目录: {cache_dir}")
    try:
        # 设置Hugging Face的参数，不检查在线更新，增加超时时间
        from transformers.utils import hub
        hub.REQUESTS_KWARGS = {'timeout': 300}
        
        # 强制使用离线模式
        import os
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        
        # 尝试从预训练模型初始化tokenizer
        try:
            tokenizer = RobertaTokenizer.from_pretrained(
                mlm_type, 
                cache_dir=cache_dir, 
                local_files_only=True,  # 只使用本地文件
                use_fast=True
            )
            print("成功从本地加载tokenizer")
        except Exception as e:
            print(f"从本地加载失败: {e}")
            # 尝试不使用本地限制加载
            tokenizer = RobertaTokenizer.from_pretrained(
                mlm_type, 
                cache_dir=cache_dir
            )
    except Exception as e:
        print(f"无法加载tokenizer: {e}")
        # 如果无法加载，尝试使用基本配置创建
        print("尝试使用基本配置创建tokenizer...")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            "roberta-base", 
            use_fast=False
        )
        
    # 添加特殊标记
    special_tokens = {'additional_special_tokens': ['<c>', '</c>']}
    tokenizer.add_special_tokens(special_tokens)
    return tokenizer

def get_optimizer(model: nn.Module, learning_rate: float) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=learning_rate)

def batch_forward_func(model, batch):
    # Extract inputs from batch dictionary
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']
    token_type_ids = batch['token_type_ids']
    mask_pos = batch['mask_pos']
    sep_pos = batch['sep_pos']
    
    # RoBERTa doesn't use token_type_ids, so we can pass None if it exists but is all zeros
    if token_type_ids is not None and (token_type_ids == 0).all():
        token_type_ids = None
    
    eci_logits, cwd_logits, ced_logits = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        mask_pos=mask_pos,
        sep_pos=sep_pos
    )
    
    return (eci_logits, cwd_logits, ced_logits)

def batch_cal_loss_func(outputs, batch):
    # Extract labels from batch
    labels = batch['labels']
    eci_labels, cwd_labels, ced_labels = labels[:, 0], labels[:, 1], labels[:, 2]
    
    # Extract logits from outputs
    eci_logits, cwd_logits, ced_logits = outputs
    
    # Calculate losses for each task
    eci_loss = F.cross_entropy(eci_logits, eci_labels)
    cwd_loss = F.cross_entropy(cwd_logits, cwd_labels)
    ced_loss = F.cross_entropy(ced_logits, ced_labels)
    
    # Combine losses
    total_loss = eci_loss + cwd_loss + ced_loss
    return total_loss

def batch_metrics_func(outputs, batch):
    # Extract labels from batch
    labels = batch['labels']
    eci_labels = labels[:, 0]
    
    # Extract logits from outputs
    eci_logits, _, _ = outputs
    eci_preds = torch.argmax(eci_logits, dim=1)
    
    # Return predictions and labels for metrics calculation
    return eci_preds.detach().cpu().numpy().tolist(), eci_labels.detach().cpu().numpy().tolist()

def metrics_cal_func(preds, labels):
    # Calculate accuracy
    accuracy = accuracy_score(labels, preds)
    
    # Calculate precision, recall, and F1 score
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, 
        preds, 
        average='macro',  # Use macro averaging for multi-class
        zero_division=0  # Avoid division by zero
    )
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

# Data processing functions
def ecimp_preprocess_data(data: Dict, tokenizer: RobertaTokenizer) -> List[Dict]:
    """预处理数据，将原始数据转换为模型输入格式"""
    results = []
    tokens = data["tokens"]
    relations = data["relations"]
    
    # 打印tokenizer信息，便于调试
    print(f"Tokenizer mask token: {tokenizer.mask_token}, id: {tokenizer.mask_token_id}")
    print(f"Tokenizer sep token: {tokenizer.sep_token}, id: {tokenizer.sep_token_id}")
    
    for rel in relations:
        # 提取事件1和事件2的文本
        event1_text = " ".join(tokens[rel["event1_start_index"]:rel["event1_end_index"]+1])
        event2_text = " ".join(tokens[rel["event2_start_index"]:rel["event2_end_index"]+1])
        
        # 构建输入文本 - 使用tokenizer.mask_token而不是硬编码的[MASK]
        input_text = f"In this sentence, '{event1_text}' <c> {tokenizer.mask_token} </c> '{event2_text}'."
        
        # Tokenize
        encoded = tokenizer(
            input_text,
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors=None
        )
        
        # 尝试找到mask和sep的位置
        try:
            # 找到 mask 和 sep 的位置
            mask_pos = encoded.input_ids.index(tokenizer.mask_token_id)
            sep_pos = encoded.input_ids.index(tokenizer.sep_token_id)
        except ValueError as e:
            print(f"找不到mask或sep token，跳过此样本: {e}")
            print(f"输入文本: {input_text}")
            print(f"Token IDs: {encoded.input_ids}")
            continue
        
        # 构建标签
        # cause: 1 表示 event1 导致 event2, -1 表示 event2 导致 event1, 0 表示无因果关系
        if rel["cause"] == 1:
            label = 0  # Cause
        elif rel["cause"] == -1:
            label = 1  # CausedBy
        else:
            label = 2  # NA
            
        # 构建辅助任务标签
        has_signal = rel["signal_start_index"] >= 0  # 是否有显式信号词
        cwd_label = 1 if has_signal else 0
        ced_label = 1 if not has_signal and rel["cause"] != 0 else 0  # 无信号词但有因果关系
        
        # 对于RoBERTa，不使用token_type_ids
        # 如果token_type_ids不存在，创建一个全零的替代
        token_type_ids = encoded.get('token_type_ids', [0] * len(encoded['input_ids']))
        
        # 组装结果
        result = {
            "input_ids": encoded.input_ids,
            "attention_mask": encoded.attention_mask,
            "token_type_ids": token_type_ids,
            "mask_pos": mask_pos,
            "sep_pos": sep_pos,
            "labels": [label, cwd_label, ced_label]  # [ECI, CWD, CED]
        }
        results.append(result)
    
    return results

def valid_data_preprocess(data: Dict) -> Dict:
    """验证集数据预处理，保持数据结构不变"""
    return data.copy()

class ECIMPCollator:
    def __init__(self, tokenizer: RobertaTokenizer, use_event_prompt: bool = True, use_signal_prompt: bool = True):
        self.tokenizer = tokenizer
        self.use_event_prompt = use_event_prompt
        self.use_signal_prompt = use_signal_prompt
    
    def __call__(self, batch: List[Dict]) -> Tuple[torch.Tensor, ...]:
        """将批次数据整理成张量格式"""
        input_ids = torch.tensor([x["input_ids"] for x in batch])
        attention_mask = torch.tensor([x["attention_mask"] for x in batch])
        token_type_ids = torch.tensor([x["token_type_ids"] for x in batch])
        mask_pos = torch.tensor([x["mask_pos"] for x in batch])
        sep_pos = torch.tensor([x["sep_pos"] for x in batch])
        labels = torch.tensor([x["labels"] for x in batch])
        
        return input_ids, attention_mask, token_type_ids, mask_pos, sep_pos, labels

class ECIMPSampler(torch.utils.data.Sampler):
    def __init__(self, dataset: List[Dict], shuffle: bool):
        self.dataset = dataset
        self.shuffle = shuffle
        self.indices = list(range(len(dataset)))
    
    def __iter__(self):
        if self.shuffle:
            import random
            random.shuffle(self.indices)
        return iter(self.indices)
    
    def __len__(self):
        return len(self.dataset)

class ECIMPLrScheduler:
    """
    已在 run_ecimp.py 中使用 transformers.get_linear_schedule_with_warmup 替代
    此类可以删除
    """
    pass 