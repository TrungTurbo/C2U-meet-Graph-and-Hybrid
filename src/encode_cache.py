#!/usr/bin/env python3
"""
encode_cache.py
Trích xuất text embeddings bằng encoder đóng băng (Frozen Pretrained Encoders):
- sentence-transformers/all-MiniLM-L6-v2 (d=384)
- BAAI/bge-base-en-v1.5 (d=768)
- intfloat/e5-base-v2 (d=768)
Chuẩn hóa L2 và lưu cache ra file .pt trên GPU (hoặc CPU).
"""

import os
import json
import argparse
from typing import List, Dict, Any
import torch

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


def load_jsonl(filepath: str) -> List[Dict[str, Any]]:
    items = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line.strip()))
    return items


def encode_texts(
    model: Any,
    texts: List[str],
    batch_size: int = 128,
    device: str = "cuda"
) -> torch.Tensor:
    """Encode danh sách văn bản và chuẩn hóa L2 tensor."""
    if not texts:
        return torch.empty(0)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_tensor=True,
        device=device
    )
    return embeddings.cpu()


def process_dataset_cache(
    split_dir: str,
    output_dir: str,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 128,
    device: str = "cuda"
):
    if SentenceTransformer is None:
        raise ImportError("Vui lòng cài đặt sentence-transformers: pip install sentence-transformers")

    print(f"\n[Encoder] Loading model '{model_name}' on device='{device}'...")
    encoder = SentenceTransformer(model_name, device=device)
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Corpus
    corpus_file = os.path.join(split_dir, "corpus.jsonl")
    if os.path.exists(corpus_file):
        print(f"[Encoder] Encoding Corpus from {corpus_file}...")
        corpus_data = load_jsonl(corpus_file)
        corpus_texts = [x["document"] for x in corpus_data]
        corpus_emb = encode_texts(encoder, corpus_texts, batch_size=batch_size, device=device)
        torch.save(corpus_emb, os.path.join(output_dir, "corpus_emb.pt"))
        
        # Save metadata
        corpus_meta = [{
            "id": i,
            "is_target": x.get("is_target", 0),
            "target_genre": x.get("target_genre", "")
        } for i, x in enumerate(corpus_data)]
        with open(os.path.join(output_dir, "corpus_meta.json"), "w", encoding="utf-8") as f:
            json.dump(corpus_meta, f, ensure_ascii=False)
        print(f"  -> Corpus embeddings saved: {corpus_emb.shape}")
        
    # 2. Target Profile (20 documents)
    target_prof_file = os.path.join(split_dir, "target_profile.jsonl")
    if os.path.exists(target_prof_file):
        print(f"[Encoder] Encoding Target Profile...")
        prof_data = load_jsonl(target_prof_file)
        prof_texts = [x["document"] for x in prof_data]
        prof_emb = encode_texts(encoder, prof_texts, batch_size=batch_size, device=device)
        torch.save(prof_emb, os.path.join(output_dir, "target_profile_emb.pt"))
        print(f"  -> Target Profile embeddings saved: {prof_emb.shape}")
        
    # 3. Calibration Control
    cal_file = os.path.join(split_dir, "calibration_control.jsonl")
    if os.path.exists(cal_file):
        print(f"[Encoder] Encoding Calibration Control...")
        cal_data = load_jsonl(cal_file)
        cal_texts = [x.get("query", x.get("document", "")) for x in cal_data]
        cal_emb = encode_texts(encoder, cal_texts, batch_size=batch_size, device=device)
        torch.save(cal_emb, os.path.join(output_dir, "calibration_control_emb.pt"))
        print(f"  -> Calibration Control embeddings saved: {cal_emb.shape}")
        
    # 4. Test Target (100 pairs)
    test_target_file = os.path.join(split_dir, "test_target.jsonl")
    if os.path.exists(test_target_file):
        print(f"[Encoder] Encoding Test Target queries & docs...")
        tt_data = load_jsonl(test_target_file)
        tt_queries = [x["query"] for x in tt_data]
        tt_docs = [x["document"] for x in tt_data]
        tt_q_emb = encode_texts(encoder, tt_queries, batch_size=batch_size, device=device)
        tt_doc_emb = encode_texts(encoder, tt_docs, batch_size=batch_size, device=device)
        torch.save(tt_q_emb, os.path.join(output_dir, "test_target_q_emb.pt"))
        torch.save(tt_doc_emb, os.path.join(output_dir, "test_target_doc_emb.pt"))
        print(f"  -> Test Target Q: {tt_q_emb.shape}, Doc: {tt_doc_emb.shape}")
        
    # 5. Test Control (300 pairs)
    test_control_file = os.path.join(split_dir, "test_control.jsonl")
    if os.path.exists(test_control_file):
        print(f"[Encoder] Encoding Test Control queries & docs...")
        tc_data = load_jsonl(test_control_file)
        tc_queries = [x["query"] for x in tc_data]
        tc_docs = [x["document"] for x in tc_data]
        tc_q_emb = encode_texts(encoder, tc_queries, batch_size=batch_size, device=device)
        tc_doc_emb = encode_texts(encoder, tc_docs, batch_size=batch_size, device=device)
        torch.save(tc_q_emb, os.path.join(output_dir, "test_control_q_emb.pt"))
        torch.save(tc_doc_emb, os.path.join(output_dir, "test_control_doc_emb.pt"))
        print(f"  -> Test Control Q: {tc_q_emb.shape}, Doc: {tc_doc_emb.shape}")
        
    # 6. Train Triplets (query, pos_doc, neg_doc)
    triplets_file = os.path.join(split_dir, "train_triplets.jsonl")
    if os.path.exists(triplets_file):
        print(f"[Encoder] Encoding Train Triplets for ACR...")
        triplets_data = load_jsonl(triplets_file)
        q_texts = [x["query"] for x in triplets_data]
        pos_texts = [x["pos_doc"] for x in triplets_data]
        neg_texts = [x["neg_doc"] for x in triplets_data]
        
        q_emb = encode_texts(encoder, q_texts, batch_size=batch_size, device=device)
        pos_emb = encode_texts(encoder, pos_texts, batch_size=batch_size, device=device)
        neg_emb = encode_texts(encoder, neg_texts, batch_size=batch_size, device=device)
        
        torch.save({
            "q_emb": q_emb,
            "pos_emb": pos_emb,
            "neg_emb": neg_emb
        }, os.path.join(output_dir, "train_triplets_emb.pt"))
        print(f"  -> Triplets saved: {len(q_texts)} triplets")
        
    print(f"\n[Encoder] All embeddings successfully cached to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Encode and cache text splits to .pt embeddings.")
    parser.add_argument("--data_dir", type=str, default="processed_splits/imdb", help="Directory containing dataset splits")
    parser.add_argument("--output_dir", type=str, default="cache_emb/imdb", help="Output directory for .pt caches")
    parser.add_argument("--model_name", type=str, default="sentence-transformers/all-MiniLM-L6-v2", help="Pretrained encoder model")
    parser.add_argument("--batch_size", type=int, default=128, help="Encoding batch size")
    parser.add_argument("--device", type=str, default=None, help="Device: cuda or cpu")
    args = parser.parse_args()
    
    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    process_dataset_cache(
        split_dir=args.data_dir,
        output_dir=args.output_dir,
        model_name=args.model_name,
        batch_size=args.batch_size,
        device=device
    )


if __name__ == "__main__":
    main()
