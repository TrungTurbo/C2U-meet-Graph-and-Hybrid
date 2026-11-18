#!/usr/bin/env python3
"""
split_units.py
Phân chia dữ liệu TOFU, Amazon, MNLI, ArXiv thành các Unit đánh giá chuẩn theo C2U:
- target_profile.jsonl: 20 tài liệu đại diện cho Target Concept
- calibration_control.jsonl: Tập đối chứng (Control) để ước lượng D và hiệu chỉnh ngưỡng gate
- test_target.jsonl: 100 cặp (query, document) thuộc Target Concept
- test_control.jsonl: 300 cặp (query, document) thuộc Control Attribute
- corpus.jsonl: Kho chỉ mục dùng chung gồm 100 target docs + 300 control docs
- train_triplets.jsonl: Tập triplet (query, pos_doc, neg_doc) để huấn luyện ACR

LƯU Ý: Đã chính thức loại bỏ hoàn toàn IMDb khỏi dự án (theo Mục B.3 và Bảng 9 bài báo gốc).
"""

import os
import json
import random
import argparse
from collections import defaultdict
from typing import Dict, List, Any

# Từ khóa xác định chủ đề nhạy cảm (Family, Location, Birth, Personal Info) cho TOFU
TOFU_SENSITIVE_KEYWORDS = [
    "parent", "father", "mother", "family", "wife", "husband", "son", "daughter",
    "child", "children", "born", "birth", "reside", "live", "city", "country",
    "hometown", "location", "grew up", "where does", "who is this", "relatives",
    "heritage", "marry", "married", "death", "die", "age", "old is"
]


def save_jsonl(data: List[Dict[str, Any]], filepath: str):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  -> Đã lưu {len(data)} mẫu vào {filepath}")


def process_tofu(
    input_file: str,
    output_dir: str,
    seed: int = 42
):
    print("\n" + "=" * 60)
    print("[Split] Đang xử lý TOFU (Lọc chủ đề nhạy cảm Family/Location)...")
    print("=" * 60)
    random.seed(seed)
    
    sensitive_items = []
    career_items = []
    
    with open(input_file, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            item = json.loads(line.strip())
            item["author_id"] = idx // 20
            q_lower = item["query"].lower()
            
            if any(kw in q_lower for kw in TOFU_SENSITIVE_KEYWORDS):
                item["concept_type"] = "sensitive_target"
                sensitive_items.append(item)
            else:
                item["concept_type"] = "career_control"
                career_items.append(item)
                
    random.shuffle(sensitive_items)
    random.shuffle(career_items)
    
    print(f"  Tổng số câu hỏi nhạy cảm (Target Concept): {len(sensitive_items)}")
    print(f"  Tổng số câu hỏi sự nghiệp (Control Concept): {len(career_items)}")
    
    target_profile = [{"document": x["document"], "author_id": x["author_id"], "is_target": 1} for x in sensitive_items[:20]]
    test_target = sensitive_items[20:120]
    for x in test_target:
        x["is_target"] = 1
        
    rem_sensitive = sensitive_items[120:]
    
    calibration_control = career_items[:200]
    for x in calibration_control:
        x["is_target"] = 0
        
    test_control = career_items[200:500]
    for x in test_control:
        x["is_target"] = 0
        
    rem_career = career_items[500:]
    
    # Kho chỉ mục corpus dùng chung gồm 100 target docs + 300 control docs (theo Mục 4.B)
    corpus = []
    for x in test_target:
        corpus.append({"document": x["document"], "is_target": 1, "author_id": x["author_id"]})
    for x in test_control:
        corpus.append({"document": x["document"], "is_target": 0, "author_id": x["author_id"]})
        
    author_to_career = defaultdict(list)
    for x in rem_career:
        author_to_career[x["author_id"]].append(x["document"])
    all_authors = list(author_to_career.keys())
    
    train_triplets = []
    for x in rem_career[:2000]:
        q = x["query"]
        pos_doc = x["document"]
        a_id = x["author_id"]
        other_authors = [other for other in all_authors if other != a_id]
        if other_authors:
            neg_a = random.choice(other_authors)
            if author_to_career[neg_a]:
                neg_doc = random.choice(author_to_career[neg_a])
                train_triplets.append({
                    "query": q,
                    "pos_doc": pos_doc,
                    "neg_doc": neg_doc
                })
                
    ds_out = os.path.join(output_dir, "tofu")
    save_jsonl(target_profile, os.path.join(ds_out, "target_profile.jsonl"))
    save_jsonl(calibration_control, os.path.join(ds_out, "calibration_control.jsonl"))
    save_jsonl(test_target, os.path.join(ds_out, "test_target.jsonl"))
    save_jsonl(test_control, os.path.join(ds_out, "test_control.jsonl"))
    save_jsonl(corpus, os.path.join(ds_out, "corpus.jsonl"))
    save_jsonl(train_triplets, os.path.join(ds_out, "train_triplets.jsonl"))
    print(f"[Split] TOFU hoàn tất. Corpus size: {len(corpus)}")


def process_mnli(
    input_file: str,
    output_dir: str,
    target_genre: str = "travel",
    seed: int = 42
):
    print("\n" + "=" * 60)
    print(f"[Split] Đang xử lý MNLI (Target Concept: genre='{target_genre}')...")
    print("=" * 60)
    random.seed(seed)
    
    target_items = []
    control_items = []
    
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line.strip())
            genre = item.get("target_genre", "")
            if genre == target_genre:
                target_items.append(item)
            else:
                control_items.append(item)
                
    random.shuffle(target_items)
    random.shuffle(control_items)
    
    target_profile = [{"document": x["document"], "target_genre": x.get("target_genre", ""), "is_target": 1} for x in target_items[:20]]
    test_target = target_items[20:120]
    for x in test_target:
        x["is_target"] = 1
        
    calibration_control = control_items[:200]
    for x in calibration_control:
        x["is_target"] = 0
        
    test_control = control_items[200:500]
    for x in test_control:
        x["is_target"] = 0
        
    rem_control = control_items[500:]
    
    corpus = []
    for x in test_target:
        corpus.append({"document": x["document"], "is_target": 1, "target_genre": x.get("target_genre", "")})
    for x in test_control:
        corpus.append({"document": x["document"], "is_target": 0, "target_genre": x.get("target_genre", "")})
        
    genre_to_docs = defaultdict(list)
    for x in rem_control:
        genre_to_docs[x.get("target_genre", "")].append(x["document"])
    all_genres = list(genre_to_docs.keys())
    
    train_triplets = []
    for x in rem_control[:3000]:
        g = x.get("target_genre", "")
        q = x["query"]
        pos_doc = x["document"]
        diff_genres = [other_g for other_g in all_genres if other_g != g]
        if diff_genres:
            neg_genre = random.choice(diff_genres)
            if genre_to_docs[neg_genre]:
                neg_doc = random.choice(genre_to_docs[neg_genre])
                train_triplets.append({
                    "query": q,
                    "pos_doc": pos_doc,
                    "neg_doc": neg_doc
                })
                
    ds_out = os.path.join(output_dir, "mnli")
    save_jsonl(target_profile, os.path.join(ds_out, "target_profile.jsonl"))
    save_jsonl(calibration_control, os.path.join(ds_out, "calibration_control.jsonl"))
    save_jsonl(test_target, os.path.join(ds_out, "test_target.jsonl"))
    save_jsonl(test_control, os.path.join(ds_out, "test_control.jsonl"))
    save_jsonl(corpus, os.path.join(ds_out, "corpus.jsonl"))
    save_jsonl(train_triplets, os.path.join(ds_out, "train_triplets.jsonl"))
    print(f"[Split] MNLI hoàn tất. Corpus size: {len(corpus)}")


def process_amazon(
    input_file: str,
    output_dir: str,
    seed: int = 42
):
    print("\n" + "=" * 60)
    print("[Split] Đang xử lý Amazon Polarity...")
    print("=" * 60)
    random.seed(seed)
    
    pos_items = []
    neg_items = []
    
    with open(input_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 40000:
                break
            if not line.strip():
                continue
            item = json.loads(line.strip())
            if item.get("sentiment") == 0:
                neg_items.append(item)
            else:
                pos_items.append(item)
                
    random.shuffle(neg_items)
    random.shuffle(pos_items)
    
    target_profile = [{"document": x["document"], "sentiment": 0, "is_target": 1} for x in neg_items[:20]]
    test_target = neg_items[20:120]
    for x in test_target:
        x["is_target"] = 1
        
    calibration_control = pos_items[:200]
    for x in calibration_control:
        x["is_target"] = 0
        
    test_control = pos_items[200:500]
    for x in test_control:
        x["is_target"] = 0
        
    rem_pos = pos_items[500:3000]
    rem_neg = neg_items[120:2500]
    
    corpus = []
    for x in test_target:
        corpus.append({"document": x["document"], "is_target": 1, "sentiment": 0})
    for x in test_control:
        corpus.append({"document": x["document"], "is_target": 0, "sentiment": 1})
        
    train_triplets = []
    for x in rem_pos[:2000]:
        q = x["query"]
        pos_doc = x["document"]
        neg_doc = random.choice(rem_neg)["document"]
        train_triplets.append({
            "query": q,
            "pos_doc": pos_doc,
            "neg_doc": neg_doc
        })
        
    ds_out = os.path.join(output_dir, "amazon")
    save_jsonl(target_profile, os.path.join(ds_out, "target_profile.jsonl"))
    save_jsonl(calibration_control, os.path.join(ds_out, "calibration_control.jsonl"))
    save_jsonl(test_target, os.path.join(ds_out, "test_target.jsonl"))
    save_jsonl(test_control, os.path.join(ds_out, "test_control.jsonl"))
    save_jsonl(corpus, os.path.join(ds_out, "corpus.jsonl"))
    save_jsonl(train_triplets, os.path.join(ds_out, "train_triplets.jsonl"))
    print(f"[Split] Amazon hoàn tất. Corpus size: {len(corpus)}")


def process_arxiv(
    input_file: str,
    output_dir: str,
    target_category: str = "quant-ph",
    max_lines: int = 15000,
    seed: int = 42
):
    print("\n" + "=" * 60)
    print(f"[Split] Đang xử lý ArXiv Metadata (Target Category: '{target_category}')...")
    print("=" * 60)
    random.seed(seed)
    
    target_items = []
    control_items = []
    
    with open(input_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_lines:
                break
            if not line.strip():
                continue
            item = json.loads(line.strip())
            title = item.get("title", "").replace("\n", " ").strip()
            abstract = item.get("abstract", "").replace("\n", " ").strip()
            cats = item.get("categories", "").split()
            
            if not title or not abstract:
                continue
                
            entry = {
                "query": title,
                "document": abstract,
                "categories": item.get("categories", "")
            }
            if target_category in cats:
                target_items.append(entry)
            else:
                control_items.append(entry)
                
    random.shuffle(target_items)
    random.shuffle(control_items)
    
    print(f"  Target items ('{target_category}'): {len(target_items)}")
    print(f"  Control items: {len(control_items)}")
    
    if len(target_items) < 120 or len(control_items) < 500:
        print("[Warning] Số lượng mẫu ArXiv chưa đủ 120 target hoặc 500 control. Đang điều chỉnh tỷ lệ...")
        target_profile = [{"document": x["document"], "is_target": 1} for x in target_items[:min(20, len(target_items))]]
        test_target = target_items[20:min(120, len(target_items))]
    else:
        target_profile = [{"document": x["document"], "is_target": 1} for x in target_items[:20]]
        test_target = target_items[20:120]
        
    for x in test_target:
        x["is_target"] = 1
        
    calibration_control = control_items[:200]
    for x in calibration_control:
        x["is_target"] = 0
        
    test_control = control_items[200:500]
    for x in test_control:
        x["is_target"] = 0
        
    rem_control = control_items[500:]
    
    corpus = []
    for x in test_target:
        corpus.append({"document": x["document"], "is_target": 1})
    for x in test_control:
        corpus.append({"document": x["document"], "is_target": 0})
        
    train_triplets = []
    for x in rem_control[:2000]:
        q = x["query"]
        pos_doc = x["document"]
        if target_items:
            neg_doc = random.choice(target_items)["document"]
            train_triplets.append({
                "query": q,
                "pos_doc": pos_doc,
                "neg_doc": neg_doc
            })
            
    ds_out = os.path.join(output_dir, "arxiv")
    save_jsonl(target_profile, os.path.join(ds_out, "target_profile.jsonl"))
    save_jsonl(calibration_control, os.path.join(ds_out, "calibration_control.jsonl"))
    save_jsonl(test_target, os.path.join(ds_out, "test_target.jsonl"))
    save_jsonl(test_control, os.path.join(ds_out, "test_control.jsonl"))
    save_jsonl(corpus, os.path.join(ds_out, "corpus.jsonl"))
    save_jsonl(train_triplets, os.path.join(ds_out, "train_triplets.jsonl"))
    print(f"[Split] ArXiv hoàn tất. Corpus size: {len(corpus)}")


def main():
    parser = argparse.ArgumentParser(description="Chia các bộ dữ liệu thành đơn vị chuẩn C2U (Disjoint Splits).")
    parser.add_argument("--data_dir", type=str, default="data", help="Đường dẫn thư mục chứa dữ liệu thô")
    parser.add_argument("--output_dir", type=str, default="processed_splits", help="Đường dẫn thư mục lưu splits")
    parser.add_argument("--dataset", type=str, default="all", choices=["tofu", "mnli", "amazon", "arxiv", "all"], help="Tập dữ liệu cần chia")
    parser.add_argument("--seed", type=int, default=42, help="Seed ngẫu nhiên")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    tofu_path = os.path.join(args.data_dir, "TOFU", "tofu_retrieval.jsonl")
    mnli_path = os.path.join(args.data_dir, "MNLI", "mnli_retrieval.jsonl")
    amazon_path = os.path.join(args.data_dir, "amazon", "amazon_test_retrieval.jsonl")
    arxiv_path = os.path.join(args.data_dir, "arxiv", "arxiv-metadata-oai-snapshot.json")
    
    if args.dataset in ["tofu", "all"] and os.path.exists(tofu_path):
        process_tofu(tofu_path, args.output_dir, seed=args.seed)
    if args.dataset in ["mnli", "all"] and os.path.exists(mnli_path):
        process_mnli(mnli_path, args.output_dir, seed=args.seed)
    if args.dataset in ["amazon", "all"] and os.path.exists(amazon_path):
        process_amazon(amazon_path, args.output_dir, seed=args.seed)
    if args.dataset in ["arxiv", "all"] and os.path.exists(arxiv_path):
        process_arxiv(arxiv_path, args.output_dir, seed=args.seed)
        
    print("\n[Done] Hoàn tất chia toàn bộ dữ liệu thành các đơn vị chuẩn C2U!")


if __name__ == "__main__":
    main()
