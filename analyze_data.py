#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
from typing import Dict, List, Any, Optional
from collections import Counter
import logging
from pathlib import Path
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def validate_document(doc: Dict[str, Any]) -> bool:
    """验证文档格式是否正确"""
    required_fields = ["tokens", "sentences", "relations", "token_index2sentence_index"]
    return all(field in doc for field in required_fields)

def analyze_cross_sentence_relations(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """分析跨句子的因果关系"""
    if not validate_document(data):
        logger.warning("Invalid document format, skipping...")
        return []
        
    tokens = data["tokens"]
    token_index2sentence_index = data["token_index2sentence_index"]
    relations = data["relations"]
    sentences = data["sentences"]
    
    cross_sent_relations = []
    for rel in relations:
        try:
            # 标记事件位置
            marked_tokens = tokens.copy()
            event1_start = rel["event1_start_index"]
            event1_end = rel["event1_end_index"]
            event2_start = rel["event2_start_index"]
            event2_end = rel["event2_end_index"]
            
            # 使用标准XML格式标记
            marked_tokens[event1_start] = "<t1>" + marked_tokens[event1_start]
            marked_tokens[event1_end] = marked_tokens[event1_end] + "</t1>"
            marked_tokens[event2_start] = "<t2>" + marked_tokens[event2_start]
            marked_tokens[event2_end] = marked_tokens[event2_end] + "</t2>"
            
            # 检查是否跨句
            sent_idx1 = token_index2sentence_index[min(event1_start, event2_start)]
            sent_idx2 = token_index2sentence_index[max(event1_end, event2_end)]
            
            if sent_idx2 - sent_idx1 >= 2 and rel.get("cause", 0) != 0:
                cross_sent_relations.append({
                    "cause": rel["cause"],
                    "sentence": " ".join(marked_tokens[sentences[sent_idx1]["start"]:sentences[sent_idx2]["end"]+1]),
                    "distance": sent_idx2 - sent_idx1  # 添加句子距离信息
                })
        except KeyError as e:
            logger.warning(f"Missing field in relation: {e}")
            continue
        except Exception as e:
            logger.error(f"Error processing relation: {e}")
            continue
    
    return cross_sent_relations

def analyze_causal_signals(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """分析因果信号词及其分布"""
    no_signal_examples = []
    with_signal_examples = []
    signal_counter = Counter()
    
    for article in data:
        if not validate_document(article):
            continue
            
        tokens = article["tokens"]
        sentences = article["sentences"]
        token_index2sentence_index = article["token_index2sentence_index"]
        
        for rel in article["relations"]:
            try:
                # 检查必要字段
                if "cause" not in rel or rel.get("cause", 0) == 0:
                    continue
                
                # 确定源事件和目标事件
                if rel["cause"] > 0:
                    source = " ".join(tokens[rel["event2_start_index"]:rel["event2_end_index"]+1])
                    target = " ".join(tokens[rel["event1_start_index"]:rel["event1_end_index"]+1])
                else:
                    source = " ".join(tokens[rel["event1_start_index"]:rel["event1_end_index"]+1])
                    target = " ".join(tokens[rel["event2_start_index"]:rel["event2_end_index"]+1])
                
                # 构建示例
                example = {
                    "source": source,
                    "target": target,
                    "cause_direction": "forward" if rel["cause"] > 0 else "backward"
                }
                
                # 处理有无信号词的情况
                if rel.get("signal_start_index", -1) < 0:
                    # 无信号词
                    sent_start = sentences[token_index2sentence_index[min(rel["event1_start_index"], rel["event2_start_index"])]]["start"]
                    sent_end = sentences[token_index2sentence_index[max(rel["event1_end_index"], rel["event2_end_index"])]]["end"]
                    example["sentence"] = " ".join(tokens[sent_start:sent_end+1])
                    example["signal"] = ""
                    no_signal_examples.append(example)
                else:
                    # 有信号词
                    sent_start = sentences[token_index2sentence_index[min(rel["event1_start_index"], rel["event2_start_index"], rel["signal_start_index"])]]["start"]
                    sent_end = sentences[token_index2sentence_index[max(rel["event1_end_index"], rel["event2_end_index"], rel["signal_end_index"])]]["end"]
                    signal = " ".join(tokens[rel["signal_start_index"]:rel["signal_end_index"]+1])
                    example["sentence"] = " ".join(tokens[sent_start:sent_end+1])
                    example["signal"] = signal
                    with_signal_examples.append(example)
                    signal_counter[signal] += 1
            except KeyError as e:
                logger.warning(f"Missing field in relation: {e}")
                continue
            except Exception as e:
                logger.error(f"Error processing relation: {e}")
                continue
    
    # 按信号词频率排序
    signal_stats = sorted(signal_counter.items(), key=lambda x: x[1], reverse=True)
    with_signal_examples.sort(key=lambda x: signal_counter[x["signal"]], reverse=True)
    
    return {
        "no_signal_examples": no_signal_examples,
        "with_signal_examples": with_signal_examples,
        "signal_stats": signal_stats
    }

def save_results(results: Dict[str, Any], output_dir: Path) -> None:
    """保存分析结果到文件"""
    try:
        output_dir.mkdir(exist_ok=True)
        
        for filename, data in results.items():
            output_path = output_dir / filename
            logger.info(f"Saving results to {output_path}")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Error saving results: {e}")
        raise

def main():
    output_dir = Path("analysis_results")
    
    # 加载数据
    datasets = []
    found_data = False
    for dataset_path in ["data/eventstory.json", "data/timebank.json"]:
        if Path(dataset_path).exists():
            logger.info(f"Loading dataset from {dataset_path}")
            try:
                with open(dataset_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        datasets.extend(data)
                        found_data = True
                    else:
                        logger.error(f"Invalid data format in {dataset_path}, expected a list")
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON format in {dataset_path}")
            except Exception as e:
                logger.error(f"Error loading {dataset_path}: {e}")
        else:
            logger.warning(f"Dataset not found: {dataset_path}")
    
    if not found_data:
        logger.error("No valid datasets found!")
        sys.exit(1)
    
    logger.info(f"Loaded {len(datasets)} documents in total")
    
    try:
        # 分析跨句关系
        cross_sent_results = []
        for doc in datasets:
            cross_sent_results.extend(analyze_cross_sentence_relations(doc))
        
        # 分析因果信号
        causal_results = analyze_causal_signals(datasets)
        
        # 准备输出结果
        output_files = {
            "cross_sentence_relations.json": cross_sent_results,
            "no_signal_examples.json": causal_results["no_signal_examples"],
            "with_signal_examples.json": causal_results["with_signal_examples"],
            "signal_statistics.json": causal_results["signal_stats"]
        }
        
        # 保存结果
        save_results(output_files, output_dir)
        
        # 打印统计信息
        logger.info("\nAnalysis Summary:")
        logger.info(f"- Total documents analyzed: {len(datasets)}")
        logger.info(f"- Cross-sentence relations: {len(cross_sent_results)}")
        logger.info(f"- Examples without signals: {len(causal_results['no_signal_examples'])}")
        logger.info(f"- Examples with signals: {len(causal_results['with_signal_examples'])}")
        logger.info(f"- Unique signal types: {len(causal_results['signal_stats'])}")
        
        # 打印最常见的信号词
        logger.info("\nTop 10 most common signals:")
        for signal, count in causal_results["signal_stats"][:10]:
            logger.info(f"  {signal}: {count} occurrences")
            
        # 打印跨句距离统计
        distances = [rel["distance"] for rel in cross_sent_results]
        if distances:
            avg_distance = sum(distances) / len(distances)
            max_distance = max(distances)
            logger.info(f"\nCross-sentence Statistics:")
            logger.info(f"- Average sentence distance: {avg_distance:.2f}")
            logger.info(f"- Maximum sentence distance: {max_distance}")
            
    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 