#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
from typing import Dict, List, Any
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def process_maven_ere(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Process MAVEN-ERE dataset into our unified format
    
    Args:
        data: List of documents from MAVEN-ERE dataset
        
    Returns:
        List of processed documents
    """
    logger.info(f"Processing MAVEN-ERE data with {len(data)} documents")
    
    processed_docs = []
    relation_count = 0
    
    for doc_index, doc in enumerate(data):
        try:
            # 检查文档格式
            if not isinstance(doc, dict) or "tokens" not in doc:
                logger.warning(f"Document {doc_index} has invalid format, skipping")
                continue
                
            processed_doc = {
                "tokens": doc["tokens"],
                "token_index2sentence_index": [],  # Will be filled
                "sentences": [],  # Will be filled
                "relations": []  # Will be filled
            }
            
            # 构建句子索引映射
            sentences = doc.get("sentence_ends", [])
            current_sent_idx = 0
            sent_start = 0
            
            # 如果没有sentence_ends，尝试创建一个简单的句子
            if not sentences:
                logger.warning(f"Document {doc_index} has no sentence info, treating as single sentence")
                processed_doc["sentences"] = [{"start": 0, "end": len(doc["tokens"])-1}]
                processed_doc["token_index2sentence_index"] = [0] * len(doc["tokens"])
            else:
                for i, token in enumerate(doc["tokens"]):
                    if i in sentences:
                        processed_doc["sentences"].append({
                            "start": sent_start,
                            "end": i
                        })
                        sent_start = i + 1
                        current_sent_idx += 1
                    processed_doc["token_index2sentence_index"].append(current_sent_idx)
            
            # 处理因果关系
            doc_has_causal = False
            for relation in doc.get("relations", []):
                try:
                    # 检查关系是否是因果关系
                    if relation.get("type") == "CAUSE":
                        doc_has_causal = True
                        
                        # 获取事件参数
                        arg1 = relation.get("arg1", {})
                        arg2 = relation.get("arg2", {})
                        
                        if not arg1 or not arg2:
                            continue
                        
                        # 创建因果关系
                        rel_dict = {
                            "event1_start_index": arg1.get("start", 0),
                            "event1_end_index": arg1.get("end", 0) - 1,
                            "event2_start_index": arg2.get("start", 0),
                            "event2_end_index": arg2.get("end", 0) - 1,
                            "signal_start_index": -1,  # MAVEN-ERE doesn't mark signal words
                            "signal_end_index": -1,
                            "cause": 1  # Direction: 1 means arg1->arg2
                        }
                        processed_doc["relations"].append(rel_dict)
                        relation_count += 1
                except Exception as e:
                    logger.warning(f"Error processing relation in doc {doc_index}: {e}")
            
            if processed_doc["relations"]:  # Only add documents with causal relations
                processed_docs.append(processed_doc)
                if doc_has_causal:
                    logger.info(f"Found document {doc_index} with {len(processed_doc['relations'])} causal relations")
                
        except Exception as e:
            logger.error(f"Error processing document {doc_index}: {e}")
    
    logger.info(f"Processed {len(processed_docs)} documents with {relation_count} causal relations")
    return processed_docs

def main():
    """处理MAVEN-ERE数据集"""
    # Create processed directory if it doesn't exist
    os.makedirs("data/processed", exist_ok=True)
    
    # Process train, dev, test sets
    for split, filename in [("train", "train.jsonl"), ("dev", "valid.jsonl"), ("test", "test.jsonl")]:
        input_path = f"data/MAVEN_ERE/{filename}"
        if os.path.exists(input_path):
            logger.info(f"Processing {input_path}")
            
            # 读取JSONL文件
            documents = []
            try:
                with open(input_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            doc = json.loads(line.strip())
                            documents.append(doc)
                        except json.JSONDecodeError as e:
                            logger.warning(f"Error parsing JSON line: {e}")
                            continue
                
                logger.info(f"Loaded {len(documents)} documents from {input_path}")
                
                # 处理数据
                processed_data = process_maven_ere(documents)
                
                # 保存处理后的数据
                output_path = f"data/processed/maven_ere_{split}.json"
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(processed_data, f, ensure_ascii=False, indent=2)
                logger.info(f"Saved processed data to {output_path}")
                
            except Exception as e:
                logger.error(f"Error processing {input_path}: {e}")
        else:
            logger.warning(f"Input file not found: {input_path}")

if __name__ == "__main__":
    main() 