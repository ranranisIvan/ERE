# ECIMP
`Event Causality Identification via Derivative Prompt Joint Learning`

论文代码
## Requirements
- beautifulsoup4==4.10.0
- numpy==1.18.5
- scikit_learn==1.0.2
- torch==1.10.1+cu113
- tqdm==4.47.0
- transformers==4.14.1
- typing_extensions==4.1.1

## 运行代码
使用了Timebank和Eventstory两种数据集，且Timebank十折交叉验证  Eventstory五折交叉验证

### 改进说明
本代码在原始论文基础上进行了以下改进：
1. 添加了第四个模板作为负样本，增强了模型对负例的识别能力
2. 使用PolaLinearAttention替代原先的门控单元，提升了特征融合效果

### Roberta+linear baseline
当使用Timebank数据集时:
```shell
nohup python -u run_mlm_base.py --main_device=0 --device_ids=0 --data_path=data/timebank.json --mlm_name_or_path=../roberta-base --split_n=10 --gradient_accumulate=1 > log/mlm_base_timebank.log &
```
使用Eventstory时:
```shell
nohup python -u run_mlm_base.py --main_device=0 --device_ids=0,1 --data_path=data/eventstory.json --mlm_name_or_path=../roberta-base --split_n=5 --gradient_accumulate=1 > log/mlm_base_eventstory.log &
```
以此类推,具体参数配置见run_*.py文件

### single prompt 
Timebank
```shell 
nohup python -u run_prompt_tuning.py --main_device=0 --device_ids=0,1 --mlm_name_or_path=../roberta-base --data_path=data/timebank.json --split_n=10 --gradient_accumulate=1 > log/prompt_tuning_timebank.log &
```
### Event Causality Identification via Derivative Prompt Joint Learning
Timebank
```shell
nohup python -u run_ecimp.py --data_path data/timebank.json --split_n=10 --use_event_prompt=1 --use_signal_prompt=1 --use_sep_gate=1 --use_mask1_gate=1 --reuse=1 --batch_size=6 --use_linear=1 --device_ids=1 --main_device=1 --gradient_accumulate=3 --learning_rate=0.0001 --mlm_name_or_path=../roberta-base > log/ecimp_timebank_full.log &
```
### 军事数据集
```shell
nohup python -u run_ecimp.py --data_path data/data_trans3.json --split_n=5 --use_event_prompt=1 --use_signal_prompt=1 --use_sep_gate=1 --use_mask1_gate=1 --reuse=1 --batch_size=6 --use_linear=1 --device_ids=0 --main_device=0 --gradient_accumulate=3 --learning_rate=0.0002 --mlm_name_or_path=hfl/chinese-roberta-wwm-ext > log/ecimp_trans3_full_linear.log &
```