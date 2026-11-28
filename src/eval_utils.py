#!/usr/bin/env python3
"""
src/eval_utils.py
Tiện ích dùng chung cho Giai đoạn 4 (Dense, Transforms, Multi-concept, Ablation):
- Phân giải đường dẫn cache (.pt) và checkpoint ACR (.pt) tương thích cả tên ngắn & tên đầy đủ.
- Đánh giá truy xuất Top-K chuẩn v2: H5 Drop, TC, CN (own-label Hit@5), CN_feasible, TE, Precision@5, Density, Hit@10.
- Paired bootstrap 4.000 mẫu theo từng truy vấn.
"""

import os
import json
import math
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
import torch.nn.functional as F


ENCODER_SHORT_MAP = {
    "all-MiniLM-L6-v2": "minilm",
    "minilm": "minilm",
    "bge-base-en-v1.5": "bge",
    "bge": "bge",
    "e5-base-v2": "e5",
    "e5": "e5",
    "Qwen3-Embedding-0.6B": "qwen3",
    "qwen3": "qwen3",
}

ENCODER_CANONICAL_MAP = {
    "minilm": "all-MiniLM-L6-v2",
    "all-minilm-l6-v2": "all-MiniLM-L6-v2",
    "bge": "bge-base-en-v1.5",
    "bge-base-en-v1.5": "bge-base-en-v1.5",
    "e5": "e5-base-v2",
    "e5-base-v2": "e5-base-v2",
    "qwen3": "Qwen3-Embedding-0.6B",
    "qwen3-embedding-0.6b": "Qwen3-Embedding-0.6B",
}


def resolve_cache_path(cache_dir: str, split_name: str, encoder_key: str) -> Optional[str]:
    """Tìm file cache .pt theo tên ngắn hoặc tên đầy đủ."""
    short = ENCODER_SHORT_MAP.get(encoder_key, ENCODER_SHORT_MAP.get(encoder_key.lower(), encoder_key))
    canonical = ENCODER_CANONICAL_MAP.get(encoder_key.lower(), encoder_key)
    candidates = [
        os.path.join(cache_dir, f"{split_name}_{short}.pt"),
        os.path.join(cache_dir, f"{split_name}_{canonical.replace('/', '_')}.pt"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def resolve_ckpt_path(ckpt_dir: str, split_name: str, encoder_key: str, suffix: str = "") -> Optional[str]:
    """Tìm checkpoint ACR .pt tương ứng với split và encoder."""
    short = ENCODER_SHORT_MAP.get(encoder_key, ENCODER_SHORT_MAP.get(encoder_key.lower(), encoder_key))
    canonical = ENCODER_CANONICAL_MAP.get(encoder_key.lower(), encoder_key)
    suf = f"_{suffix.lstrip('_')}" if suffix else ""
    candidates = [
        os.path.join(ckpt_dir, f"acr_{split_name}_{short}{suf}.pt"),
        os.path.join(ckpt_dir, f"acr_{split_name}_{canonical.replace('/', '_')}{suf}.pt"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def load_acr_weights(ckpt_path: str, device: str = "cpu") -> Tuple[torch.Tensor, Optional[torch.Tensor], float]:
    """Nạp ma trận A (đã chuẩn hóa hàng), head W và tau từ checkpoint ACR."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "A" in ckpt:
        A = ckpt["A"].float().to(device)
        W = ckpt["W"].float().to(device) if "W" in ckpt else None
        tau = float(ckpt.get("config", {}).get("tau", 0.1))
    elif "model_state_dict" in ckpt:
        sd = ckpt["model_state_dict"]
        A = F.normalize(sd["A_raw"].float().to(device), p=2, dim=-1)
        W = sd["W"].float().to(device) if "W" in sd else None
        tau = float(ckpt.get("tau", 0.1))
    else:
        raise KeyError(f"Checkpoint {ckpt_path} không có key 'A' hoặc 'model_state_dict'")
    return A, W, tau


def get_stage_data(cache: Dict[str, Any], stage: str) -> Tuple[torch.Tensor, torch.Tensor, List[str], List[str]]:
    """
    Trả về (q_embs, q_is_target, q_labels, q_ids) cho 'selection' (val) hoặc 'confirm'.
    KHÔNG cho phép truy cập 'test' ở GĐ4.
    """
    if stage in ("selection", "val"):
        return (
            cache["val_q_embs"].float(),
            cache["val_q_is_target"].long(),
            list(cache["val_q_labels"]),
            list(cache["val_q_ids"]),
        )
    elif stage in ("confirm", "confirmation"):
        return (
            cache["confirm_q_embs"].float(),
            cache["confirm_q_is_target"].long(),
            list(cache["confirm_q_labels"]),
            list(cache["confirm_q_ids"]),
        )
    else:
        raise ValueError(f"GĐ4 chỉ cho phép stage 'selection' hoặc 'confirm', bị từ chối: {stage}")


def compute_per_query_metrics(
    topk_indices: torch.Tensor,      # [B, max_k] (ít nhất 10 nếu tính Hit@10)
    q_embs: torch.Tensor,            # [B, d]
    index_embs: torch.Tensor,        # [M, d]
    index_is_target: torch.Tensor,   # [M]
    index_labels: List[str],
    q_is_target: torch.Tensor,       # [B]
    q_labels: List[str],
    k: int = 5,
) -> Dict[str, Any]:
    """Tính đầy đủ vector chỉ số per-query cho paired bootstrap."""
    q_norm = F.normalize(q_embs.float(), p=2, dim=-1)
    idx_norm = F.normalize(index_embs.float(), p=2, dim=-1)

    # Tâm tài liệu target trong index (dùng cho chỉ số TC, Mục A.2)
    t_doc_mask = (index_is_target == 1)
    mu_t_index = F.normalize(idx_norm[t_doc_mask].mean(dim=0), p=2, dim=-1)
    target_cos_all = torch.matmul(q_norm, mu_t_index).cpu().numpy()

    idx_t_np = index_is_target.cpu().numpy()
    q_t_np = q_is_target.cpu().numpy()
    top_np = topk_indices.cpu().numpy()

    ctrl_idx_label_set = {index_labels[j] for j in range(len(index_labels)) if idx_t_np[j] == 0}

    t_hit5, t_density5, t_cos = [], [], []
    c_own_hit5, c_own_p5, c_te5, c_own_hit10, c_feasible = [], [], [], [], []

    for i in range(len(q_t_np)):
        top_k = top_np[i, :k]
        top_10 = top_np[i, :min(10, top_np.shape[1])]
        is_t_k = idx_t_np[top_k]

        if q_t_np[i] == 1:
            t_hit5.append(1.0 if np.any(is_t_k == 1) else 0.0)
            t_density5.append(float(np.mean(is_t_k == 1)))
            t_cos.append(float(target_cos_all[i]))
        else:
            ql = q_labels[i]
            lbls_k = [index_labels[j] for j in top_k]
            lbls_10 = [index_labels[j] for j in top_10]
            matches_k = [1.0 if (lbl == ql and idx_t_np[j] == 0) else 0.0 for j, lbl in zip(top_k, lbls_k)]
            matches_10 = [1.0 if (lbl == ql and idx_t_np[j] == 0) else 0.0 for j, lbl in zip(top_10, lbls_10)]

            c_own_hit5.append(1.0 if any(m > 0 for m in matches_k) else 0.0)
            c_own_p5.append(float(np.mean(matches_k)))
            c_te5.append(float(np.mean(is_t_k == 1)))
            c_own_hit10.append(1.0 if any(m > 0 for m in matches_10) else 0.0)
            c_feasible.append(ql in ctrl_idx_label_set)

    return {
        "t_hit5": np.array(t_hit5, dtype=np.float64),
        "t_density5": np.array(t_density5, dtype=np.float64),
        "t_cos": np.array(t_cos, dtype=np.float64),
        "c_own_hit5": np.array(c_own_hit5, dtype=np.float64),
        "c_own_p5": np.array(c_own_p5, dtype=np.float64),
        "c_te5": np.array(c_te5, dtype=np.float64),
        "c_own_hit10": np.array(c_own_hit10, dtype=np.float64),
        "c_feasible": np.array(c_feasible, dtype=bool),
    }


def dense_topk_indices(q_embs: torch.Tensor, index_embs: torch.Tensor, max_k: int = 10) -> torch.Tensor:
    """Xếp hạng cosine và trả về Top-max_k indices."""
    sims = torch.matmul(F.normalize(q_embs.float(), dim=-1), F.normalize(index_embs.float(), dim=-1).t())
    return torch.topk(sims, k=min(max_k, index_embs.shape[0]), dim=-1).indices


def paired_bootstrap_ci(
    diffs: np.ndarray,
    n_boot: int = 4000,
    seed: int = 42,
    scale: float = 100.0,
) -> Tuple[float, float, float]:
    """Paired bootstrap 95% CI cho vector hiệu số per-query."""
    if len(diffs) == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    n = len(diffs)
    mean_val = float(np.mean(diffs)) * scale
    # Vectorized theo batch 500 để tiết kiệm RAM
    boots = []
    batch = 500
    for start in range(0, n_boot, batch):
        bs = min(batch, n_boot - start)
        idx = rng.integers(0, n, size=(bs, n))
        boots.append(diffs[idx].mean(axis=1) * scale)
    all_boots = np.concatenate(boots)
    lo, hi = np.percentile(all_boots, [2.5, 97.5])
    return mean_val, float(lo), float(hi)


def load_jsonl(filepath: str) -> List[Dict[str, Any]]:
    with open(filepath, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
