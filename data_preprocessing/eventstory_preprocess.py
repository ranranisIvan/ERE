#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EventStory Dataset Preprocessor

This script processes the EventStory dataset from XML format to JSON format.
It handles event extraction, causal relations, and coreference chains.

The output JSON format contains:
- tokens: List of tokens in the document
- sentences: List of sentence boundaries
- relations: List of causal relations between events
"""

import json
import os
from typing import Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass
from pathlib import Path
import logging
from bs4 import BeautifulSoup
import bs4
from bs4.element import Tag

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class Event:
    """Represents an event mention in text"""
    token_indexs: List[int]
    token_tids: List[int]
    group_id: int = -1
    mid: int = -1
    
    def __init__(self, token_str: str) -> None:
        self.token_indexs = []
        self.token_tids = []
        
        # Parse token IDs
        if "_" in token_str:
            self.token_tids = [int(num) for num in token_str.split("_")]
        else:
            self.token_tids = [int(token_str)]
            
        # Convert to 0-based indices
        self.token_indexs = [tid - 1 for tid in self.token_tids]
    
    def key(self) -> str:
        """Returns a unique key for the event"""
        return "_".join(map(str, self.token_tids))

@dataclass
class Relation:
    """Represents a causal relation between events"""
    source_group_id: int = -1
    target_group_id: int = -1
    signal_token_start_index: int = -1
    signal_token_end_index: int = -1

@dataclass
class EventGroup:
    """Represents a group of coreferent events"""
    events: List[Event]
    relations: List[Relation]
    id: int = -1

def read_coreference_chains(chain_file: Path) -> Tuple[List[EventGroup], List[Relation], Dict[str, Event]]:
    """
    Reads event coreference chains from a tab-separated file.
    
    Args:
        chain_file: Path to the coreference chain file
        
    Returns:
        Tuple containing:
        - List of event groups
        - List of relations between groups
        - Dictionary mapping event keys to Event objects
    """
    if not chain_file.exists():
        return [], [], {}
        
    try:
        event_groups: List[Set[str]] = []
        group_relations: List[Tuple[int, int]] = []
        added_groups: Dict[str, int] = {}
        
        with chain_file.open('r') as f:
            for line in f:
                line = line.strip()
                if len(line) <= 2:
                    continue
                    
                source, target, rel_type = line.split('\t')
                
                # Process source tokens
                source_tokens = sorted(source.split())
                source_key = " ".join(source_tokens)
                if source_key not in added_groups:
                    event_groups.append(set(source_tokens))
                    added_groups[source_key] = len(event_groups) - 1
                source_idx = added_groups[source_key]
                
                if source == target:
                    continue
                    
                # Process target tokens
                target_tokens = sorted(target.split())
                target_key = " ".join(target_tokens)
                if target_key not in added_groups:
                    event_groups.append(set(target_tokens))
                    added_groups[target_key] = len(event_groups) - 1
                target_idx = added_groups[target_key]
                
                # Add relation based on type
                if rel_type == "PRECONDITION":
                    group_relations.append((source_idx, target_idx))
                elif rel_type == "FALLING_ACTION":
                    group_relations.append((target_idx, source_idx))
                
                if source_idx == target_idx:
                    continue
        
        # Validate no overlap between groups
        for i in range(len(event_groups)-1):
            for j in range(i+1, len(event_groups)):
                assert len(event_groups[i] & event_groups[j]) == 0, \
                    f"Event appears in multiple groups: {event_groups[i] & event_groups[j]}"
        
        # Convert to final format
        groups = []
        relations = []
        key_to_event = {}
        
        # Create groups
        for i, token_group in enumerate(event_groups):
            group = EventGroup([], [], i)
            for token in token_group:
                event = Event(token)
                event.group_id = i
                group.events.append(event)
                key_to_event[event.key()] = event
            groups.append(group)
        
        # Create relations
        for source_idx, target_idx in group_relations:
            relations.append(Relation(source_idx, target_idx))
            
        return groups, relations, key_to_event
        
    except Exception as e:
        logger.error(f"Error reading coreference chain file {chain_file}: {e}")
        return [], [], {}

def process_xml_file(xml_path: Path, chain_path: Optional[Path] = None) -> Optional[Dict]:
    """
    Processes a single XML file from the EventStory dataset.
    
    Args:
        xml_path: Path to the XML file
        chain_path: Optional path to the coreference chain file
        
    Returns:
        Dictionary containing processed document data or None if processing fails
    """
    logger.info(f"Processing {xml_path}")
    
    try:
        # Read and parse XML
        with xml_path.open('r') as f:
            soup = BeautifulSoup(f.read(), 'lxml')
        
        # Process coreference chains if available
        event_groups, group_relations, key_to_events = [], [], {}
        if chain_path and chain_path.exists():
            event_groups, group_relations, key_to_events = read_coreference_chains(chain_path)
        
        # Extract tokens and sentences
        tokens = []
        token_to_sentence = []
        sentences = []
        id_to_token = {}
        
        for token in soup.find_all("token"):
            t_id = token["t_id"]
            sent_idx = int(token["sentence"])
            
            id_to_token[t_id] = token
            tokens.append(token.text)
            token_to_sentence.append(sent_idx)
            
            # Update sentence boundaries
            while len(sentences) <= sent_idx:
                sentences.append({"start": len(tokens)-1, "end": -1})
            sentences[sent_idx-1]["end"] = len(tokens)-1
        
        # Process events
        mid_to_event = {}
        # 处理所有类型的ACTION标签
        action_tags = ["ACTION_OCCURRENCE", "ACTION_ASPECTUAL", "ACTION_REPORTING", 
                       "ACTION_STATE", "NEG_ACTION_STATE", "NEG_ACTION_OCCURRENCE"]
                       
        for event_elem in soup.find_all(action_tags):
            try:
                m_id = int(event_elem["m_id"])
                
                # 获取所有token_anchor子元素
                token_anchors = event_elem.find_all("token_anchor")
                if not token_anchors:
                    continue
                    
                # 获取事件的token IDs
                event_tokens = []
                for token in token_anchors:
                    if "t_id" in token.attrs:
                        event_tokens.append(token["t_id"])
                
                if not event_tokens:
                    continue
                    
                key = "_".join(event_tokens)
                
                # 创建事件对象
                event = Event(key)
                event.mid = m_id
                mid_to_event[m_id] = event
            except Exception as e:
                logger.warning(f"Error processing event {event_elem['m_id'] if 'm_id' in event_elem.attrs else 'unknown'}: {e}")
                continue
        
        # Process relations
        relations = {}
        
        # Add potential relations for events in same sentence
        mids = list(mid_to_event.keys())
        for i in range(len(mids)-1):
            for j in range(i+1, len(mids)):
                mid1, mid2 = mids[i], mids[j]
                event1, event2 = mid_to_event[mid1], mid_to_event[mid2]
                
                # Check if events are in same sentence
                if token_to_sentence[event1.token_indexs[0]] == token_to_sentence[event2.token_indexs[-1]]:
                    relations[f"{mid1}_{mid2}"] = {
                        "event1_start_index": event1.token_indexs[0],
                        "event1_end_index": event1.token_indexs[-1],
                        "event2_start_index": event2.token_indexs[0],
                        "event2_end_index": event2.token_indexs[-1],
                        "signal_start_index": -1,
                        "signal_end_index": -1,
                        "cause": 0
                    }
        
        # Process plot links
        for link in soup.find_all("plot_link"):
            try:
                # 访问子元素中的m_id属性
                source_id = int(link.find("source")["m_id"])
                target_id = int(link.find("target")["m_id"])
                
                # 检查relType属性（大写）而不是reltype（小写）
                rel_type = link.get("relType", "")
                
                # 为每一个PLOT_LINK创建或更新关系
                key = f"{source_id}_{target_id}"
                if key not in relations:
                    # 如果不存在，创建新关系
                    if source_id in mid_to_event and target_id in mid_to_event:
                        event1 = mid_to_event[source_id]
                        event2 = mid_to_event[target_id]
                        relations[key] = {
                            "event1_start_index": event1.token_indexs[0],
                            "event1_end_index": event1.token_indexs[-1],
                            "event2_start_index": event2.token_indexs[0],
                            "event2_end_index": event2.token_indexs[-1],
                            "signal_start_index": -1,
                            "signal_end_index": -1,
                            "cause": 0
                        }
                
                # 根据relType设置cause值
                if key in relations:
                    if rel_type == "PRECONDITION":
                        relations[key]["cause"] = 1
                    elif rel_type == "FALLING_ACTION":
                        relations[key]["cause"] = -1
                
                # 处理SIGNAL属性（如果存在）
                signal_str = link.get("SIGNAL", "")
                if signal_str and signal_str.isdigit():
                    signal_id = int(signal_str)
                    if signal_id in mid_to_event:
                        signal_event = mid_to_event[signal_id]
                        if key in relations:
                            relations[key]["signal_start_index"] = signal_event.token_indexs[0]
                            relations[key]["signal_end_index"] = signal_event.token_indexs[-1]
            
            except Exception as e:
                logger.warning(f"Error processing plot link: {e}")
                continue
        
        # Update relations from coreference chains
        for group_rel in group_relations:
            group1 = event_groups[group_rel.source_group_id]
            group2 = event_groups[group_rel.target_group_id]
            
            for event1 in group1.events:
                for event2 in group2.events:
                    if event1.mid == event2.mid:
                        continue
                        
                    key = f"{event1.mid}_{event2.mid}"
                    if key in relations:
                        relations[key].update({
                            "cause": 1,
                            "signal_start_index": group_rel.signal_token_start_index,
                            "signal_end_index": group_rel.signal_token_end_index
                        })
                    else:
                        key = f"{event2.mid}_{event1.mid}"
                        if key in relations:
                            relations[key].update({
                                "cause": -1,
                                "signal_start_index": group_rel.signal_token_start_index,
                                "signal_end_index": group_rel.signal_token_end_index
                            })
        
        return {
            "tokens": tokens,
            "token_index2sentence_index": token_to_sentence,
            "sentences": sentences,
            "relations": list(relations.values())
        }
        
    except Exception as e:
        logger.error(f"Error processing file {xml_path}: {e}")
        return None

def main():
    """
    Main function to process EventStory dataset
    """
    # 创建输出目录
    os.makedirs("data", exist_ok=True)
    
    # Configuration
    input_root = "data/eventstory/data"  # 修改为正确的数据路径
    output_path = "data/eventstory.json"
    result = []
    count = 0
    
    # Check if input directory exists
    if not os.path.exists(input_root):
        logger.error(f"Raw data directory not found: {input_root}")
        return
    
    # Process each subdirectory (story)
    for story_id in os.listdir(input_root):
        story_dir = Path(input_root) / story_id
        if not story_dir.is_dir():
            continue
            
        logger.info(f"Processing story {story_id}")
        
        # Process each XML file in the story directory
        for xml_file in story_dir.glob("*.xml"):
            chain_file = None  # EventStory doesn't have separate chain files
            
            # Process the XML file
            doc = process_xml_file(xml_file, chain_file)
            if doc and doc.get("relations"):
                result.append(doc)
                count += len(doc["relations"])
    
    # Save processed data
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
        
    logger.info(f"Processed {len(result)} documents with {count} relations, saved to {output_path}")

if __name__ == "__main__":
    main() 