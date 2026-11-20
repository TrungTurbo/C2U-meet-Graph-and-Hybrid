#!/usr/bin/env python3
"""
src/split_units.py
Phân loại & Chia tách đơn vị đánh giá chuẩn C2U Giai đoạn 1 (v2):
- Không rò rỉ dữ liệu (Disjoint Splits tuyệt đối theo row_id và text hash).
- Cấu hình YAML danh sách từ khóa được khóa trước khi split.
- Phân chia: 4 dataset (TOFU, Amazon, MNLI, ArXiv) x 3 seed = 12 thư mục split (+ arxiv_multi_seed<k>).
- Mỗi unit gồm:
    * train_acr.jsonl: 1.500 - 2.000 cặp (query, doc) có nhãn.
    * target_profile.jsonl: 20 tài liệu target (chỉ doc).
    * calibration_control.jsonl: 300 dòng control (có cả doc_text và query_text).
    * calibration_labeled.jsonl: 400 dòng = 300 control + 100 target mới.
    * index_corpus.jsonl: 400 tài liệu = 100 Target + 300 Control (chỉ doc).
    * val_queries.jsonl: 100 Target + 300 Control (selection, chỉ query).
    * confirm_queries.jsonl: 100 Target + 300 Control (confirmation, chỉ query).
    * test_queries.jsonl: 100 Target + 300 Control (final test, chỉ query).
- Dòng nguồn của truy vấn val/confirm/test KHÔNG nằm trong index_corpus (query_source_in_index: false).
- Sinh manifest SHA256 từng file cho từng split.
"""

import os
import json
import random
import hashlib
import re
from typing import List, Dict, Any, Set, Tuple
import yaml


def normalize_text(text: str) -> str:
    """Chuẩn hóa văn bản: lowercase và bỏ khoảng trắng thừa."""
    return " ".join(text.lower().split())


def compute_sha256(filepath: str) -> str:
    """Tính SHA256 của một file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def compute_text_hash(text: str) -> str:
    """Tính hash của văn bản đã chuẩn hóa để kiểm tra rời rạc."""
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def save_jsonl(data: List[Dict[str, Any]], filepath: str):
    """Lưu danh sách dict thành file JSON Lines."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def verify_split_disjointness(split_dir: str, is_multi: bool = False):
    """
    Kiểm tra nghiệm thu bắt buộc (assert rời rạc tuyệt đối):
    1. Rời rạc theo row_id và text hash giữa các rổ.
    2. Đúng số lượng từng rổ.
    """
    files = [
        "train_acr.jsonl",
        "target_profile.jsonl",
        "calibration_control.jsonl",
        "calibration_labeled.jsonl",
        "index_corpus.jsonl",
        "val_queries.jsonl",
        "confirm_queries.jsonl",
        "test_queries.jsonl",
    ]
    for fn in files:
        fp = os.path.join(split_dir, fn)
        assert os.path.exists(fp), f"Thiếu file {fp}"

    def read_jsonl(fn):
        with open(os.path.join(split_dir, fn), "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    train_acr = read_jsonl("train_acr.jsonl")
    target_profile = read_jsonl("target_profile.jsonl")
    calib_control = read_jsonl("calibration_control.jsonl")
    calib_labeled = read_jsonl("calibration_labeled.jsonl")
    index_corpus = read_jsonl("index_corpus.jsonl")
    val_q = read_jsonl("val_queries.jsonl")
    confirm_q = read_jsonl("confirm_queries.jsonl")
    test_q = read_jsonl("test_queries.jsonl")

    # 1. Kiểm tra số lượng
    if not is_multi:
        assert 1500 <= len(train_acr) <= 2000, f"train_acr count error: {len(train_acr)}"
        assert len(target_profile) == 20, f"target_profile count error: {len(target_profile)}"
        assert len(calib_control) == 300, f"calib_control count error: {len(calib_control)}"
        assert len(calib_labeled) == 400, f"calib_labeled count error: {len(calib_labeled)}"
        assert len(index_corpus) == 400, f"index_corpus count error: {len(index_corpus)}"
        assert len(val_q) == 400, f"val_queries count error: {len(val_q)}"
        assert len(confirm_q) == 400, f"confirm_queries count error: {len(confirm_q)}"
        assert len(test_q) == 400, f"test_queries count error: {len(test_q)}"
        
        # Kiểm tra tỷ lệ target / control trong index và queries
        assert sum(1 for x in index_corpus if x["is_target"] == 1) == 100
        assert sum(1 for x in index_corpus if x["is_target"] == 0) == 300
        for q_set in [val_q, confirm_q, test_q]:
            assert sum(1 for x in q_set if x["is_target"] == 1) == 100
            assert sum(1 for x in q_set if x["is_target"] == 0) == 300
    else:
        assert 1500 <= len(train_acr) <= 2000, f"Multi train_acr count error: {len(train_acr)}"
        assert len(target_profile) == 80, f"Multi target_profile count error: {len(target_profile)}"
        assert len(calib_control) == 300, f"Multi calib_control count error: {len(calib_control)}"
        assert len(calib_labeled) == 400, f"Multi calib_labeled count error: {len(calib_labeled)}"
        assert len(index_corpus) == 400, f"Multi index_corpus count error: {len(index_corpus)}"
        assert len(val_q) == 1100, f"Multi val_queries count error: {len(val_q)}"
        assert len(confirm_q) == 1100, f"Multi confirm_queries count error: {len(confirm_q)}"
        assert len(test_q) == 2200, f"Multi test_queries count error: {len(test_q)}"

    # 2. calib_control phải là tập con chính xác của calib_labeled
    calib_ctrl_ids = {x["row_id"] for x in calib_control}
    calib_lab_ids = {x["row_id"] for x in calib_labeled}
    assert calib_ctrl_ids.issubset(calib_lab_ids), "calib_control must be a subset of calib_labeled"

    # 3. Rời rạc row_id giữa các rổ độc lập
    pools = {
        "train_acr": {x["row_id"] for x in train_acr},
        "target_profile": {x["row_id"] for x in target_profile},
        "calibration_labeled": calib_lab_ids,
        "index_corpus": {x["row_id"] for x in index_corpus},
        "val_queries": {x["row_id"] for x in val_q},
        "confirm_queries": {x["row_id"] for x in confirm_q},
        "test_queries": {x["row_id"] for x in test_q},
    }
    pool_names = list(pools.keys())
    for i in range(len(pool_names)):
        for j in range(i + 1, len(pool_names)):
            n1, n2 = pool_names[i], pool_names[j]
            inter = pools[n1].intersection(pools[n2])
            assert len(inter) == 0, f"Row ID leak between {n1} and {n2}: {len(inter)} shared IDs"

    # 4. Rời rạc tuyệt đối theo text hash đã chuẩn hóa
    text_pools = {
        "train_acr": {compute_text_hash(x["query"]) for x in train_acr}.union(
            {compute_text_hash(x["document"]) for x in train_acr}
        ),
        "target_profile": {compute_text_hash(x["document"]) for x in target_profile},
        "calibration_labeled": {compute_text_hash(x["query"]) for x in calib_labeled}.union(
            {compute_text_hash(x["document"]) for x in calib_labeled}
        ),
        "index_corpus": {compute_text_hash(x["document"]) for x in index_corpus},
        "val_queries": {compute_text_hash(x["query"]) for x in val_q},
        "confirm_queries": {compute_text_hash(x["query"]) for x in confirm_q},
        "test_queries": {compute_text_hash(x["query"]) for x in test_q},
    }
    for i in range(len(pool_names)):
        for j in range(i + 1, len(pool_names)):
            n1, n2 = pool_names[i], pool_names[j]
            inter = text_pools[n1].intersection(text_pools[n2])
            assert len(inter) == 0, f"Text hash leak between {n1} and {n2}: {len(inter)} shared hashes"

    print(f"  [PASS Assert] {os.path.basename(split_dir)}: Đã kiểm tra tính rời rạc 100% không rò rỉ!")


def write_manifest(split_dir: str, dataset_name: str, seed: int, config_path: str, is_multi: bool = False):
    """Ghi manifest thông tin metadata, phân phối nhãn và SHA256 từng file."""
    manifest_path = os.path.join("manifests", f"{os.path.basename(split_dir)}.json")
    os.makedirs("manifests", exist_ok=True)
    
    file_hashes = {}
    for fn in sorted(os.listdir(split_dir)):
        if fn.endswith(".jsonl"):
            fp = os.path.join(split_dir, fn)
            file_hashes[fn] = compute_sha256(fp)
            
    config_hash = compute_sha256(config_path) if os.path.exists(config_path) else None
    
    # Đọc số lượng và phân phối nhãn
    label_distribution = {}
    for fn in ["train_acr.jsonl", "target_profile.jsonl", "calibration_control.jsonl",
                 "calibration_labeled.jsonl", "index_corpus.jsonl", "val_queries.jsonl",
                 "confirm_queries.jsonl", "test_queries.jsonl"]:
        fp = os.path.join(split_dir, fn)
        with open(fp, "r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f if line.strip()]
        target_cnt = sum(1 for x in lines if x.get("is_target", 0) == 1)
        control_cnt = sum(1 for x in lines if x.get("is_target", 0) == 0)
        label_distribution[fn] = {
            "total": len(lines),
            "target": target_cnt,
            "control": control_cnt
        }

    manifest = {
        "dataset": dataset_name,
        "seed": seed,
        "is_multi_concept": is_multi,
        "query_source_in_index": False,
        "config_path": config_path,
        "config_sha256": config_hash,
        "file_sha256": file_hashes,
        "distribution": label_distribution
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"  -> Đã ghi manifest vào {manifest_path}")


def process_tofu(
    input_file: str,
    output_base: str,
    seeds: List[int] = [42, 43, 44],
    config_path: str = "configs/tofu_keywords.yaml"
):
    """Xử lý dataset TOFU theo quy định v2."""
    print("\n" + "=" * 60)
    print(f"[TOFU] Đang xử lý TOFU từ {input_file}...")
    print("=" * 60)
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    keywords = [k.lower() for k in cfg.get("target_keywords", [])]

    # Đọc dữ liệu thô
    raw_target = []
    raw_control = []
    seen_texts = set()

    with open(input_file, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            item = json.loads(line.strip())
            author_id = idx // 20
            q = item["query"].strip()
            ans = item["document"].strip()
            doc = f"Author {author_id}: {ans}"
            
            # Deduplicate text
            qh = compute_text_hash(q)
            dh = compute_text_hash(doc)
            if qh in seen_texts or dh in seen_texts:
                continue
            seen_texts.add(qh)
            seen_texts.add(dh)

            q_lower = q.lower()
            is_target = 1 if any(kw in q_lower for kw in keywords) else 0
            label = "Family" if is_target == 1 else f"author_{author_id}"
            
            row = {
                "row_id": f"tofu_{idx}",
                "query": q,
                "document": doc,
                "is_target": is_target,
                "label": label,
                "author_id": author_id
            }
            if is_target == 1:
                raw_target.append(row)
            else:
                raw_control.append(row)

    print(f"  Tổng Target duy nhất: {len(raw_target)}, Control duy nhất: {len(raw_control)}")
    assert len(raw_target) >= 520, f"Không đủ Target cho TOFU: {len(raw_target)}"
    assert len(raw_control) >= 1500, f"Không đủ Control cho TOFU: {len(raw_control)}"

    for seed_idx, seed_val in enumerate(seeds, start=1):
        split_dir = os.path.join(output_base, f"tofu_seed{seed_idx}")
        rng = random.Random(seed_val)
        targets = raw_target.copy()
        controls = raw_control.copy()
        rng.shuffle(targets)
        rng.shuffle(controls)

        # 1. target_profile: 20 target docs (chỉ doc)
        target_profile = [
            {"row_id": x["row_id"], "document": x["document"], "is_target": 1, "label": x["label"]}
            for x in targets[:20]
        ]
        
        # 2. calibration_control: 300 control (có query và doc)
        calib_ctrl_raw = controls[:300]
        calibration_control = [
            {"row_id": x["row_id"], "query": x["query"], "document": x["document"], "is_target": 0, "label": x["label"]}
            for x in calib_ctrl_raw
        ]
        
        # 3. calibration_labeled: 300 control ở trên + 100 target mới (rời rạc)
        calib_target_raw = targets[20:120]
        calibration_labeled = list(calibration_control) + [
            {"row_id": x["row_id"], "query": x["query"], "document": x["document"], "is_target": 1, "label": x["label"]}
            for x in calib_target_raw
        ]
        
        # 4. index_corpus: 100 target docs + 300 control docs (chỉ doc)
        idx_target = targets[120:220]
        idx_control = controls[300:600]
        index_corpus = [
            {"row_id": x["row_id"], "document": x["document"], "is_target": 1, "label": x["label"]}
            for x in idx_target
        ] + [
            {"row_id": x["row_id"], "document": x["document"], "is_target": 0, "label": x["label"]}
            for x in idx_control
        ]

        # 5. val_queries: 100 target queries + 300 control queries (chỉ query)
        val_target = targets[220:320]
        val_control = controls[600:900]
        val_queries = [
            {"row_id": x["row_id"], "query": x["query"], "is_target": 1, "label": x["label"]}
            for x in val_target
        ] + [
            {"row_id": x["row_id"], "query": x["query"], "is_target": 0, "label": x["label"]}
            for x in val_control
        ]

        # 6. confirm_queries: 100 target queries + 300 control queries (chỉ query)
        conf_target = targets[320:420]
        conf_control = controls[900:1200]
        confirm_queries = [
            {"row_id": x["row_id"], "query": x["query"], "is_target": 1, "label": x["label"]}
            for x in conf_target
        ] + [
            {"row_id": x["row_id"], "query": x["query"], "is_target": 0, "label": x["label"]}
            for x in conf_control
        ]

        # 7. test_queries: 100 target queries + 300 control queries (chỉ query)
        test_target = targets[420:520]
        test_control = controls[1200:1500]
        test_queries = [
            {"row_id": x["row_id"], "query": x["query"], "is_target": 1, "label": x["label"]}
            for x in test_target
        ] + [
            {"row_id": x["row_id"], "query": x["query"], "is_target": 0, "label": x["label"]}
            for x in test_control
        ]

        # 8. train_acr: còn lại khoảng 450 target + 1.350 control = 1.800 cặp
        rem_target = targets[520:]
        rem_control = controls[1500:]
        num_target_train = min(len(rem_target), 450)
        num_ctrl_train = min(len(rem_control), 1350)
        train_acr = [
            {"row_id": x["row_id"], "query": x["query"], "document": x["document"], "is_target": 1, "label": x["label"]}
            for x in rem_target[:num_target_train]
        ] + [
            {"row_id": x["row_id"], "query": x["query"], "document": x["document"], "is_target": 0, "label": x["label"]}
            for x in rem_control[:num_ctrl_train]
        ]
        rng.shuffle(train_acr)

        save_jsonl(train_acr, os.path.join(split_dir, "train_acr.jsonl"))
        save_jsonl(target_profile, os.path.join(split_dir, "target_profile.jsonl"))
        save_jsonl(calibration_control, os.path.join(split_dir, "calibration_control.jsonl"))
        save_jsonl(calibration_labeled, os.path.join(split_dir, "calibration_labeled.jsonl"))
        save_jsonl(index_corpus, os.path.join(split_dir, "index_corpus.jsonl"))
        save_jsonl(val_queries, os.path.join(split_dir, "val_queries.jsonl"))
        save_jsonl(confirm_queries, os.path.join(split_dir, "confirm_queries.jsonl"))
        save_jsonl(test_queries, os.path.join(split_dir, "test_queries.jsonl"))

        verify_split_disjointness(split_dir)
        write_manifest(split_dir, "tofu", seed_val, config_path)


def process_amazon(
    input_file: str,
    output_base: str,
    seeds: List[int] = [42, 43, 44],
    config_path: str = "configs/amazon_keywords.yaml"
):
    """Xử lý dataset Amazon Polarity theo quy định v2."""
    print("\n" + "=" * 60)
    print(f"[Amazon] Đang xử lý Amazon từ {input_file}...")
    print("=" * 60)
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    categories = cfg.get("categories", {})

    raw_target = []
    raw_control = []
    seen_texts = set()

    with open(input_file, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            item = json.loads(line.strip())
            q = item.get("query", "").strip()
            doc = item.get("document", "").strip()
            sentiment = item.get("sentiment", -1)
            
            if not q or not doc:
                continue

            # Kiểm tra phân loại danh mục sản phẩm trên document
            doc_lower = doc.lower()
            matched_cats = [c for c, kws in categories.items() if any(kw in doc_lower for kw in kws)]
            # Quy định: loại bỏ dòng khớp 0 hoặc từ 2 danh mục trở lên
            if len(matched_cats) != 1:
                continue
            cat_name = matched_cats[0]

            qh = compute_text_hash(q)
            dh = compute_text_hash(doc)
            if qh in seen_texts or dh in seen_texts:
                continue
            seen_texts.add(qh)
            seen_texts.add(dh)

            is_target = 1 if sentiment == 1 else 0
            row = {
                "row_id": f"amazon_{idx}",
                "query": q,
                "document": doc,
                "is_target": is_target,
                "label": cat_name,
                "sentiment": sentiment
            }
            if is_target == 1:
                raw_target.append(row)
            else:
                raw_control.append(row)

            # Dừng sớm nếu đã thu thập thừa thãi cho tất cả seed
            if len(raw_target) >= 6000 and len(raw_control) >= 6000:
                break

    print(f"  Amazon Target duy nhất: {len(raw_target)}, Control duy nhất: {len(raw_control)}")
    assert len(raw_target) >= 1600, f"Không đủ Target cho Amazon: {len(raw_target)}"
    assert len(raw_control) >= 2500, f"Không đủ Control cho Amazon: {len(raw_control)}"

    for seed_idx, seed_val in enumerate(seeds, start=1):
        split_dir = os.path.join(output_base, f"amazon_seed{seed_idx}")
        rng = random.Random(seed_val)
        targets = raw_target.copy()
        controls = raw_control.copy()
        rng.shuffle(targets)
        rng.shuffle(controls)

        # target_profile: 20 target docs
        target_profile = [
            {"row_id": x["row_id"], "document": x["document"], "is_target": 1, "label": x["label"]}
            for x in targets[:20]
        ]
        # calibration_control: 300 control
        calib_ctrl_raw = controls[:300]
        calibration_control = [
            {"row_id": x["row_id"], "query": x["query"], "document": x["document"], "is_target": 0, "label": x["label"]}
            for x in calib_ctrl_raw
        ]
        # calibration_labeled: 300 control + 100 target mới
        calibration_labeled = list(calibration_control) + [
            {"row_id": x["row_id"], "query": x["query"], "document": x["document"], "is_target": 1, "label": x["label"]}
            for x in targets[20:120]
        ]
        # index_corpus: 100 target docs + 300 control docs
        index_corpus = [
            {"row_id": x["row_id"], "document": x["document"], "is_target": 1, "label": x["label"]}
            for x in targets[120:220]
        ] + [
            {"row_id": x["row_id"], "document": x["document"], "is_target": 0, "label": x["label"]}
            for x in controls[300:600]
        ]
        # val_queries: 100 target queries + 300 control queries
        val_queries = [
            {"row_id": x["row_id"], "query": x["query"], "is_target": 1, "label": x["label"]}
            for x in targets[220:320]
        ] + [
            {"row_id": x["row_id"], "query": x["query"], "is_target": 0, "label": x["label"]}
            for x in controls[600:900]
        ]
        # confirm_queries: 100 target queries + 300 control queries
        confirm_queries = [
            {"row_id": x["row_id"], "query": x["query"], "is_target": 1, "label": x["label"]}
            for x in targets[320:420]
        ] + [
            {"row_id": x["row_id"], "query": x["query"], "is_target": 0, "label": x["label"]}
            for x in controls[900:1200]
        ]
        # test_queries: 100 target queries + 300 control queries
        test_queries = [
            {"row_id": x["row_id"], "query": x["query"], "is_target": 1, "label": x["label"]}
            for x in targets[420:520]
        ] + [
            {"row_id": x["row_id"], "query": x["query"], "is_target": 0, "label": x["label"]}
            for x in controls[1200:1500]
        ]
        # train_acr: 1.000 target + 1.000 control = 2.000 cặp
        train_acr = [
            {"row_id": x["row_id"], "query": x["query"], "document": x["document"], "is_target": 1, "label": x["label"]}
            for x in targets[520:1520]
        ] + [
            {"row_id": x["row_id"], "query": x["query"], "document": x["document"], "is_target": 0, "label": x["label"]}
            for x in controls[1500:2500]
        ]
        rng.shuffle(train_acr)

        save_jsonl(train_acr, os.path.join(split_dir, "train_acr.jsonl"))
        save_jsonl(target_profile, os.path.join(split_dir, "target_profile.jsonl"))
        save_jsonl(calibration_control, os.path.join(split_dir, "calibration_control.jsonl"))
        save_jsonl(calibration_labeled, os.path.join(split_dir, "calibration_labeled.jsonl"))
        save_jsonl(index_corpus, os.path.join(split_dir, "index_corpus.jsonl"))
        save_jsonl(val_queries, os.path.join(split_dir, "val_queries.jsonl"))
        save_jsonl(confirm_queries, os.path.join(split_dir, "confirm_queries.jsonl"))
        save_jsonl(test_queries, os.path.join(split_dir, "test_queries.jsonl"))

        verify_split_disjointness(split_dir)
        write_manifest(split_dir, "amazon", seed_val, config_path)


def process_mnli(
    input_file: str,
    output_base: str,
    seeds: List[int] = [42, 43, 44],
    config_path: str = "configs/mnli_keywords.yaml"
):
    """Xử lý dataset MNLI theo quy định v2 (nhóm theo premise để chống rò rỉ tuyệt đối)."""
    print("\n" + "=" * 60)
    print(f"[MNLI] Đang xử lý MNLI từ {input_file}...")
    print("=" * 60)
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    target_genre = cfg.get("target_genre", "fiction")

    from collections import defaultdict
    target_groups = defaultdict(list)
    control_groups = defaultdict(list)

    with open(input_file, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            item = json.loads(line.strip())
            q = item.get("query", "").strip()
            doc = item.get("document", "").strip()
            genre = item.get("target_genre", "").strip()
            control_lbl = item.get("control_label", 0)

            if not q or not doc:
                continue

            is_target = 1 if genre == target_genre else 0
            label = "fiction" if is_target == 1 else f"label_{control_lbl}"
            row = {
                "row_id": f"mnli_{idx}",
                "query": q,
                "document": doc,
                "is_target": is_target,
                "label": label,
                "control_label": control_lbl
            }
            if is_target == 1:
                target_groups[doc].append(row)
            else:
                control_groups[doc].append(row)

    target_docs = list(target_groups.keys())
    control_docs = list(control_groups.keys())
    print(f"  MNLI Nhóm Document Target duy nhất: {len(target_docs)}, Control duy nhất: {len(control_docs)}")
    assert len(target_docs) >= 520, f"Không đủ nhóm doc Target cho MNLI: {len(target_docs)}"
    assert len(control_docs) >= 1500, f"Không đủ nhóm doc Control cho MNLI: {len(control_docs)}"

    for seed_idx, seed_val in enumerate(seeds, start=1):
        split_dir = os.path.join(output_base, f"mnli_seed{seed_idx}")
        rng = random.Random(seed_val)
        shuffled_t_docs = target_docs.copy()
        shuffled_c_docs = control_docs.copy()
        rng.shuffle(shuffled_t_docs)
        rng.shuffle(shuffled_c_docs)

        # 1. target_profile: 20 target docs
        target_profile = [
            {"row_id": target_groups[d][0]["row_id"], "document": d, "is_target": 1, "label": "fiction"}
            for d in shuffled_t_docs[:20]
        ]
        
        # 2. calibration_control: 300 control (query + doc)
        calib_ctrl_raw = [control_groups[d][0] for d in shuffled_c_docs[:300]]
        calibration_control = [
            {"row_id": x["row_id"], "query": x["query"], "document": x["document"], "is_target": 0, "label": x["label"]}
            for x in calib_ctrl_raw
        ]
        
        # 3. calibration_labeled: 300 control + 100 target mới
        calib_t_raw = [target_groups[d][0] for d in shuffled_t_docs[20:120]]
        calibration_labeled = list(calibration_control) + [
            {"row_id": x["row_id"], "query": x["query"], "document": x["document"], "is_target": 1, "label": x["label"]}
            for x in calib_t_raw
        ]
        
        # 4. index_corpus: 100 target docs + 300 control docs
        index_corpus = [
            {"row_id": target_groups[d][0]["row_id"], "document": d, "is_target": 1, "label": "fiction"}
            for d in shuffled_t_docs[120:220]
        ] + [
            {"row_id": control_groups[d][0]["row_id"], "document": d, "is_target": 0, "label": control_groups[d][0]["label"]}
            for d in shuffled_c_docs[300:600]
        ]
        
        # 5. val_queries: 100 target queries + 300 control queries
        val_queries = [
            {"row_id": target_groups[d][0]["row_id"], "query": target_groups[d][0]["query"], "is_target": 1, "label": "fiction"}
            for d in shuffled_t_docs[220:320]
        ] + [
            {"row_id": control_groups[d][0]["row_id"], "query": control_groups[d][0]["query"], "is_target": 0, "label": control_groups[d][0]["label"]}
            for d in shuffled_c_docs[600:900]
        ]
        
        # 6. confirm_queries: 100 target queries + 300 control queries
        confirm_queries = [
            {"row_id": target_groups[d][0]["row_id"], "query": target_groups[d][0]["query"], "is_target": 1, "label": "fiction"}
            for d in shuffled_t_docs[320:420]
        ] + [
            {"row_id": control_groups[d][0]["row_id"], "query": control_groups[d][0]["query"], "is_target": 0, "label": control_groups[d][0]["label"]}
            for d in shuffled_c_docs[900:1200]
        ]
        
        # 7. test_queries: 100 target queries + 300 control queries
        test_queries = [
            {"row_id": target_groups[d][0]["row_id"], "query": target_groups[d][0]["query"], "is_target": 1, "label": "fiction"}
            for d in shuffled_t_docs[420:520]
        ] + [
            {"row_id": control_groups[d][0]["row_id"], "query": control_groups[d][0]["query"], "is_target": 0, "label": control_groups[d][0]["label"]}
            for d in shuffled_c_docs[1200:1500]
        ]
        
        # 8. train_acr: từ các nhóm doc còn lại (1.800 cặp)
        rem_t_docs = shuffled_t_docs[520:]
        rem_c_docs = shuffled_c_docs[1500:]
        train_acr = []
        for d in rem_t_docs:
            for item in target_groups[d]:
                train_acr.append({
                    "row_id": item["row_id"],
                    "query": item["query"],
                    "document": item["document"],
                    "is_target": 1,
                    "label": item["label"]
                })
        for d in rem_c_docs:
            for item in control_groups[d]:
                train_acr.append({
                    "row_id": item["row_id"],
                    "query": item["query"],
                    "document": item["document"],
                    "is_target": 0,
                    "label": item["label"]
                })
                if len(train_acr) >= 1800:
                    break
            if len(train_acr) >= 1800:
                break
        rng.shuffle(train_acr)

        save_jsonl(train_acr, os.path.join(split_dir, "train_acr.jsonl"))
        save_jsonl(target_profile, os.path.join(split_dir, "target_profile.jsonl"))
        save_jsonl(calibration_control, os.path.join(split_dir, "calibration_control.jsonl"))
        save_jsonl(calibration_labeled, os.path.join(split_dir, "calibration_labeled.jsonl"))
        save_jsonl(index_corpus, os.path.join(split_dir, "index_corpus.jsonl"))
        save_jsonl(val_queries, os.path.join(split_dir, "val_queries.jsonl"))
        save_jsonl(confirm_queries, os.path.join(split_dir, "confirm_queries.jsonl"))
        save_jsonl(test_queries, os.path.join(split_dir, "test_queries.jsonl"))

        verify_split_disjointness(split_dir)
        write_manifest(split_dir, "mnli", seed_val, config_path)


def extract_first_sentence(text: str) -> str:
    """Trích xuất câu đầu tiên của văn bản."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    if sentences and len(sentences[0]) >= 10:
        return sentences[0].strip()
    return text.strip()


def process_arxiv(
    input_file: str,
    output_base: str,
    seeds: List[int] = [42, 43, 44],
    config_path: str = "configs/arxiv_keywords.yaml"
):
    """
    Xử lý ArXiv đơn khái niệm (cs.CR System Attacks) và đa khái niệm (K=4)
    bằng Streaming và Reservoir Sampling có seed trên toàn bộ kho lưu trữ.
    """
    print("\n" + "=" * 60)
    print(f"[ArXiv] Đang streaming xử lý ArXiv từ {input_file}...")
    print("=" * 60)
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    target_kws = [k.lower() for k in cfg.get("target_keywords", [])]
    target_cat = cfg.get("target_category", "cs.CR")
    multi_cfg = cfg.get("multi_concept", {}).get("concepts", [])
    multi_cats = {c["category"]: c["name"] for c in multi_cfg}

    # Kích thước reservoir cần thiết:
    # Single target: 20 + 100 + 100 + 100 + 100 + 100 + 1000 = 1520 -> lấy 3000
    # Single control: 300 + 300 + 300 + 300 + 300 + 1000 = 2500 -> lấy 6000
    # Multi per concept: 20 + 200 + 200 + 400 + 25 = 845 -> lấy 1500
    # Multi control: 1200 + 300 + 2000 = 3500 -> lấy 6000
    res_single_target = []
    res_single_control = []
    res_multi_concepts = {cat: [] for cat in multi_cats}
    res_multi_control = []

    k_single_target = 3000
    k_single_control = 6000
    k_multi_concept = 1500
    k_multi_control = 6000

    rng_sample = random.Random(999)
    count_single_t = 0
    count_single_c = 0
    counts_multi = {cat: 0 for cat in multi_cats}
    count_multi_c = 0

    seen_single_hashes = set()
    seen_multi_hashes = set()

    with open(input_file, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            item = json.loads(line.strip())
            title = item.get("title", "").replace("\n", " ").strip()
            abstract = item.get("abstract", "").replace("\n", " ").strip()
            cats = item.get("categories", "").split()
            paper_id = item.get("id", str(idx))

            if not title or not abstract or not cats:
                continue

            primary_cat = cats[0]
            doc = f"{title}. {abstract}"
            abs_first = extract_first_sentence(abstract)
            # Chọn query: 50% là title, 50% là câu đầu của abstract
            q = title if rng_sample.random() < 0.5 else abs_first
            
            qh = compute_text_hash(q)
            dh = compute_text_hash(doc)
            abs_lower = abstract.lower()

            # 1. Single-concept check: cs.CR + keywords
            is_cr = any(c.startswith(target_cat) for c in cats)
            has_kw = any(kw in abs_lower for kw in target_kws)

            if is_cr and has_kw:
                if qh not in seen_single_hashes and dh not in seen_single_hashes:
                    count_single_t += 1
                    row = {
                        "row_id": f"arxiv_{paper_id}",
                        "query": q,
                        "document": doc,
                        "is_target": 1,
                        "label": "System Attacks",
                        "primary_category": primary_cat
                    }
                    if len(res_single_target) < k_single_target:
                        res_single_target.append(row)
                    else:
                        j = rng_sample.randint(0, count_single_t - 1)
                        if j < k_single_target:
                            res_single_target[j] = row
                    seen_single_hashes.add(qh)
                    seen_single_hashes.add(dh)
            elif not is_cr:
                if qh not in seen_single_hashes and dh not in seen_single_hashes:
                    count_single_c += 1
                    row = {
                        "row_id": f"arxiv_{paper_id}",
                        "query": q,
                        "document": doc,
                        "is_target": 0,
                        "label": primary_cat,
                        "primary_category": primary_cat
                    }
                    if len(res_single_control) < k_single_control:
                        res_single_control.append(row)
                    else:
                        j = rng_sample.randint(0, count_single_c - 1)
                        if j < k_single_control:
                            res_single_control[j] = row
                    seen_single_hashes.add(qh)
                    seen_single_hashes.add(dh)

            # 2. Multi-concept check: cs.SE, cs.AI, q-bio.NC, math.NT
            matched_multi = None
            for m_cat in multi_cats:
                if any(c.startswith(m_cat) for c in cats):
                    matched_multi = m_cat
                    break

            if matched_multi:
                if qh not in seen_multi_hashes and dh not in seen_multi_hashes:
                    counts_multi[matched_multi] += 1
                    row = {
                        "row_id": f"arxiv_{paper_id}",
                        "query": q,
                        "document": doc,
                        "is_target": 1,
                        "label": multi_cats[matched_multi],
                        "concept_category": matched_multi,
                        "primary_category": primary_cat
                    }
                    pool = res_multi_concepts[matched_multi]
                    if len(pool) < k_multi_concept:
                        pool.append(row)
                    else:
                        j = rng_sample.randint(0, counts_multi[matched_multi] - 1)
                        if j < k_multi_concept:
                            pool[j] = row
                    seen_multi_hashes.add(qh)
                    seen_multi_hashes.add(dh)
            else:
                if qh not in seen_multi_hashes and dh not in seen_multi_hashes:
                    count_multi_c += 1
                    row = {
                        "row_id": f"arxiv_{paper_id}",
                        "query": q,
                        "document": doc,
                        "is_target": 0,
                        "label": primary_cat,
                        "primary_category": primary_cat
                    }
                    if len(res_multi_control) < k_multi_control:
                        res_multi_control.append(row)
                    else:
                        j = rng_sample.randint(0, count_multi_c - 1)
                        if j < k_multi_control:
                            res_multi_control[j] = row
                    seen_multi_hashes.add(qh)
                    seen_multi_hashes.add(dh)

    print(f"  ArXiv Single Target: {len(res_single_target)}, Single Control: {len(res_single_control)}")
    print(f"  ArXiv Multi Concepts: { {k: len(v) for k, v in res_multi_concepts.items()} }, Multi Control: {len(res_multi_control)}")

    # Xử lý 3 seed cho ArXiv đơn khái niệm
    for seed_idx, seed_val in enumerate(seeds, start=1):
        split_dir = os.path.join(output_base, f"arxiv_seed{seed_idx}")
        rng = random.Random(seed_val)
        targets = res_single_target.copy()
        controls = res_single_control.copy()
        rng.shuffle(targets)
        rng.shuffle(controls)

        target_profile = [
            {"row_id": x["row_id"], "document": x["document"], "is_target": 1, "label": x["label"]}
            for x in targets[:20]
        ]
        calib_ctrl_raw = controls[:300]
        calibration_control = [
            {"row_id": x["row_id"], "query": x["query"], "document": x["document"], "is_target": 0, "label": x["label"]}
            for x in calib_ctrl_raw
        ]
        calibration_labeled = list(calibration_control) + [
            {"row_id": x["row_id"], "query": x["query"], "document": x["document"], "is_target": 1, "label": x["label"]}
            for x in targets[20:120]
        ]
        index_corpus = [
            {"row_id": x["row_id"], "document": x["document"], "is_target": 1, "label": x["label"]}
            for x in targets[120:220]
        ] + [
            {"row_id": x["row_id"], "document": x["document"], "is_target": 0, "label": x["label"]}
            for x in controls[300:600]
        ]
        val_queries = [
            {"row_id": x["row_id"], "query": x["query"], "is_target": 1, "label": x["label"]}
            for x in targets[220:320]
        ] + [
            {"row_id": x["row_id"], "query": x["query"], "is_target": 0, "label": x["label"]}
            for x in controls[600:900]
        ]
        confirm_queries = [
            {"row_id": x["row_id"], "query": x["query"], "is_target": 1, "label": x["label"]}
            for x in targets[320:420]
        ] + [
            {"row_id": x["row_id"], "query": x["query"], "is_target": 0, "label": x["label"]}
            for x in controls[900:1200]
        ]
        test_queries = [
            {"row_id": x["row_id"], "query": x["query"], "is_target": 1, "label": x["label"]}
            for x in targets[420:520]
        ] + [
            {"row_id": x["row_id"], "query": x["query"], "is_target": 0, "label": x["label"]}
            for x in controls[1200:1500]
        ]
        train_acr = [
            {"row_id": x["row_id"], "query": x["query"], "document": x["document"], "is_target": 1, "label": x["label"]}
            for x in targets[520:1520]
        ] + [
            {"row_id": x["row_id"], "query": x["query"], "document": x["document"], "is_target": 0, "label": x["label"]}
            for x in controls[1500:2500]
        ]
        rng.shuffle(train_acr)

        save_jsonl(train_acr, os.path.join(split_dir, "train_acr.jsonl"))
        save_jsonl(target_profile, os.path.join(split_dir, "target_profile.jsonl"))
        save_jsonl(calibration_control, os.path.join(split_dir, "calibration_control.jsonl"))
        save_jsonl(calibration_labeled, os.path.join(split_dir, "calibration_labeled.jsonl"))
        save_jsonl(index_corpus, os.path.join(split_dir, "index_corpus.jsonl"))
        save_jsonl(val_queries, os.path.join(split_dir, "val_queries.jsonl"))
        save_jsonl(confirm_queries, os.path.join(split_dir, "confirm_queries.jsonl"))
        save_jsonl(test_queries, os.path.join(split_dir, "test_queries.jsonl"))

        verify_split_disjointness(split_dir)
        write_manifest(split_dir, "arxiv", seed_val, config_path)

    # Xử lý 3 seed cho ArXiv đa khái niệm K=4
    for seed_idx, seed_val in enumerate(seeds, start=1):
        multi_dir = os.path.join(output_base, f"arxiv_multi_seed{seed_idx}")
        rng = random.Random(seed_val)
        
        # Kiểm tra ngân sách từng concept (cần >= 845)
        for cat, items in res_multi_concepts.items():
            if len(items) < 845:
                raise RuntimeError(f"Capacity-deficient: concept {cat} has only {len(items)} < 845 rows!")

        shuffled_concepts = {}
        for cat, items in res_multi_concepts.items():
            pool = items.copy()
            rng.shuffle(pool)
            shuffled_concepts[cat] = pool

        shuffled_ctrl = res_multi_control.copy()
        rng.shuffle(shuffled_ctrl)

        # 1. target_profile: 20 tài liệu target mỗi khái niệm = 80 docs
        target_profile = []
        for cat in multi_cats:
            for x in shuffled_concepts[cat][:20]:
                target_profile.append({"row_id": x["row_id"], "document": x["document"], "is_target": 1, "label": x["label"]})

        # 2. calibration_control: 300 control
        calib_ctrl_raw = shuffled_ctrl[:300]
        calibration_control = [
            {"row_id": x["row_id"], "query": x["query"], "document": x["document"], "is_target": 0, "label": x["label"]}
            for x in calib_ctrl_raw
        ]

        # 3. calibration_labeled: 300 control + 25 target mỗi khái niệm = 400 dòng
        calib_labeled_targets = []
        for cat in multi_cats:
            for x in shuffled_concepts[cat][20:45]:
                calib_labeled_targets.append({"row_id": x["row_id"], "query": x["query"], "document": x["document"], "is_target": 1, "label": x["label"]})
        calibration_labeled = list(calibration_control) + calib_labeled_targets

        # 4. index_corpus: 25 target docs mỗi khái niệm (100 target) + 300 control docs = 400 docs
        index_target = []
        for cat in multi_cats:
            for x in shuffled_concepts[cat][45:70]:
                index_target.append({"row_id": x["row_id"], "document": x["document"], "is_target": 1, "label": x["label"]})
        index_control = [
            {"row_id": x["row_id"], "document": x["document"], "is_target": 0, "label": x["label"]}
            for x in shuffled_ctrl[300:600]
        ]
        index_corpus = index_target + index_control

        # 5. val_queries (Selection): 200 target mỗi khái niệm (800) + 300 control = 1.100 queries
        val_target = []
        for cat in multi_cats:
            for x in shuffled_concepts[cat][70:270]:
                val_target.append({"row_id": x["row_id"], "query": x["query"], "is_target": 1, "label": x["label"]})
        val_control = [
            {"row_id": x["row_id"], "query": x["query"], "is_target": 0, "label": x["label"]}
            for x in shuffled_ctrl[600:900]
        ]
        val_queries = val_target + val_control

        # 6. confirm_queries: 200 target mỗi khái niệm (800) + 300 control = 1.100 queries
        conf_target = []
        for cat in multi_cats:
            for x in shuffled_concepts[cat][270:470]:
                conf_target.append({"row_id": x["row_id"], "query": x["query"], "is_target": 1, "label": x["label"]})
        conf_control = [
            {"row_id": x["row_id"], "query": x["query"], "is_target": 0, "label": x["label"]}
            for x in shuffled_ctrl[900:1200]
        ]
        confirm_queries = conf_target + conf_control

        # 7. test_queries: 400 target mỗi khái niệm (1.600) + 600 control = 2.200 queries
        test_target = []
        for cat in multi_cats:
            for x in shuffled_concepts[cat][470:870]:
                test_target.append({"row_id": x["row_id"], "query": x["query"], "is_target": 1, "label": x["label"]})
        test_control = [
            {"row_id": x["row_id"], "query": x["query"], "is_target": 0, "label": x["label"]}
            for x in shuffled_ctrl[1200:1800]
        ]
        test_queries = test_target + test_control

        # 8. train_acr: 2.000 cặp huấn luyện
        train_acr = [
            {"row_id": x["row_id"], "query": x["query"], "document": x["document"], "is_target": 0, "label": x["label"]}
            for x in shuffled_ctrl[1800:3800]
        ]
        rng.shuffle(train_acr)

        save_jsonl(train_acr, os.path.join(multi_dir, "train_acr.jsonl"))
        save_jsonl(target_profile, os.path.join(multi_dir, "target_profile.jsonl"))
        save_jsonl(calibration_control, os.path.join(multi_dir, "calibration_control.jsonl"))
        save_jsonl(calibration_labeled, os.path.join(multi_dir, "calibration_labeled.jsonl"))
        save_jsonl(index_corpus, os.path.join(multi_dir, "index_corpus.jsonl"))
        save_jsonl(val_queries, os.path.join(multi_dir, "val_queries.jsonl"))
        save_jsonl(confirm_queries, os.path.join(multi_dir, "confirm_queries.jsonl"))
        save_jsonl(test_queries, os.path.join(multi_dir, "test_queries.jsonl"))

        verify_split_disjointness(multi_dir, is_multi=True)
        write_manifest(multi_dir, "arxiv_multi", seed_val, config_path, is_multi=True)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Chia các đơn vị đánh giá chuẩn C2U Giai đoạn 1 (v2)")
    parser.add_argument("--data_dir", type=str, default="data", help="Thư mục chứa dữ liệu thô")
    parser.add_argument("--output_dir", type=str, default="processed_splits", help="Thư mục lưu các split")
    parser.add_argument("--dataset", type=str, default="all", choices=["tofu", "amazon", "mnli", "arxiv", "all"])
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    seeds = [42, 43, 44]

    tofu_path = os.path.join(args.data_dir, "TOFU", "tofu_retrieval.jsonl")
    amazon_path = os.path.join(args.data_dir, "amazon", "amazon_test_retrieval.jsonl")
    mnli_path = os.path.join(args.data_dir, "MNLI", "mnli_retrieval.jsonl")
    arxiv_path = os.path.join(args.data_dir, "arxiv", "arxiv-metadata-oai-snapshot.json")

    if args.dataset in ["tofu", "all"] and os.path.exists(tofu_path):
        process_tofu(tofu_path, args.output_dir, seeds=seeds)
    if args.dataset in ["amazon", "all"] and os.path.exists(amazon_path):
        process_amazon(amazon_path, args.output_dir, seeds=seeds)
    if args.dataset in ["mnli", "all"] and os.path.exists(mnli_path):
        process_mnli(mnli_path, args.output_dir, seeds=seeds)
    if args.dataset in ["arxiv", "all"] and os.path.exists(arxiv_path):
        process_arxiv(arxiv_path, args.output_dir, seeds=seeds)

    print("\n[Done] Hoàn tất toàn bộ chia tách dữ liệu và sinh manifest!")


if __name__ == "__main__":
    main()
