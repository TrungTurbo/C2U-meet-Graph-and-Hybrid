#!/usr/bin/env python3
"""
src/evaluate_multi.py
Giai đoạn 4 (Gói 4.3): Đánh giá đa khái niệm ArXiv K=4 (và K=2 đối chứng):
- So sánh C2U-Compose với Avg-Direction trên arxiv_multi_seed{1,2,3} x 4 encoder = 12 unit.
- Endpoint (Phương trình 6): mean_u [ min_c H5Drop_compose(u, c) - min_c H5Drop_avg(u, c) ]
  với khoảng tin cậy 95% từ 4.000 bootstrap trên các hiệu số theo unit.
- Chỉ chạy trên selection và confirm, không chạm test.
"""

import os
import sys
import argparse
from typing import List, Dict, Any
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from c2u_operator import build_c2u_profile, C2UComposeOperator, AvgDirectionOperator
from eval_utils import (
    resolve_cache_path,
    resolve_ckpt_path,
    load_acr_weights,
    get_stage_data,
    dense_topk_indices,
    paired_bootstrap_ci,
)


CONCEPTS_K4 = [
    "Software Engineering",
    "Artificial Intelligence",
    "Neurons and Cognition",
    "Number Theory",
]


def evaluate_multi_unit(
    cache_path: str,
    ckpt_path: str,
    seed: int,
    encoder: str,
    stage: str,
    concepts: List[str],
    gate_quantile_space: str = "query",
) -> Dict[str, Any]:
    data = torch.load(cache_path, map_location="cpu", weights_only=False)
    q_embs, q_is_target, q_labels, _ = get_stage_data(data, stage)
    index_embs = data["index_embs"].float()
    index_is_target = data["index_is_target"].long()
    index_labels = list(data["index_labels"])

    if ckpt_path and os.path.exists(ckpt_path):
        A, W, tau = load_acr_weights(ckpt_path, device="cpu")
    else:
        d = index_embs.shape[1]
        A = F.normalize(torch.randn(64, d), p=2, dim=-1)
        W = torch.eye(d)
        tau = 0.1

    prof_embs = data["profile_embs"].float()
    prof_labels = list(data["profile_labels"])

    profiles = []
    for cname in concepts:
        mask = torch.tensor([lbl == cname for lbl in prof_labels], dtype=torch.bool)
        if not mask.any():
            raise ValueError(f"Không tìm thấy profile cho khái niệm '{cname}' trong {cache_path}")
        profiles.append(
            build_c2u_profile(
                A=A,
                target_profile_embs=prof_embs[mask],
                calib_doc_embs=data["calib_doc_embs"],
                calib_query_embs=data["calib_query_embs"],
                W=W,
                tau=tau,
                beta=1.0,
                gate_quantile_space=gate_quantile_space,
                device="cpu",
            )
        )

    comp_op = C2UComposeOperator(profiles)
    avg_op = AvgDirectionOperator(profiles)

    q_comp = comp_op.edit_query_detailed(q_embs).q_edited
    q_avg = avg_op.edit_query_detailed(q_embs).q_edited

    top_raw = dense_topk_indices(q_embs, index_embs, max_k=5).numpy()
    top_comp = dense_topk_indices(q_comp, index_embs, max_k=5).numpy()
    top_avg = dense_topk_indices(q_avg, index_embs, max_k=5).numpy()

    idx_lbl_arr = np.array(index_labels, dtype=object)
    idx_t_arr = index_is_target.numpy()

    h5_drop_comp = {}
    h5_drop_avg = {}
    for cname in concepts:
        q_idx = [i for i, lbl in enumerate(q_labels) if lbl == cname]
        assert len(q_idx) > 0, f"Thiếu truy vấn cho khái niệm '{cname}'"
        # Hit@5 của khái niệm cname: có ít nhất 1 tài liệu mang nhãn cname trong top-5
        raw_h = np.mean([1.0 if np.any(idx_lbl_arr[top_raw[i]] == cname) else 0.0 for i in q_idx]) * 100.0
        cmp_h = np.mean([1.0 if np.any(idx_lbl_arr[top_comp[i]] == cname) else 0.0 for i in q_idx]) * 100.0
        avg_h = np.mean([1.0 if np.any(idx_lbl_arr[top_avg[i]] == cname) else 0.0 for i in q_idx]) * 100.0
        h5_drop_comp[cname] = raw_h - cmp_h
        h5_drop_avg[cname] = raw_h - avg_h

    min_comp = min(h5_drop_comp.values())
    min_avg = min(h5_drop_avg.values())

    # Control utility CN & TE trên các truy vấn control
    c_idx = [i for i in range(len(q_labels)) if int(q_is_target[i].item()) == 0]
    raw_cn = np.mean([1.0 if q_labels[i] in idx_lbl_arr[top_raw[i]] else 0.0 for i in c_idx]) * 100.0
    cmp_cn = np.mean([1.0 if q_labels[i] in idx_lbl_arr[top_comp[i]] else 0.0 for i in c_idx]) * 100.0
    avg_cn = np.mean([1.0 if q_labels[i] in idx_lbl_arr[top_avg[i]] else 0.0 for i in c_idx]) * 100.0
    raw_te = np.mean([np.mean(idx_t_arr[top_raw[i]] == 1) for i in c_idx]) * 100.0
    cmp_te = np.mean([np.mean(idx_t_arr[top_comp[i]] == 1) for i in c_idx]) * 100.0
    avg_te = np.mean([np.mean(idx_t_arr[top_avg[i]] == 1) for i in c_idx]) * 100.0

    return {
        "stage": stage, "K": len(concepts), "seed": seed, "encoder": encoder,
        "min_h5_drop_compose_pp": min_comp,
        "min_h5_drop_avg_pp": min_avg,
        "endpoint_diff_pp": min_comp - min_avg,
        "compose_cn_pp": raw_cn - cmp_cn,
        "avg_cn_pp": raw_cn - avg_cn,
        "compose_te_pp": cmp_te - raw_te,
        "avg_te_pp": avg_te - raw_te,
    }


def main():
    parser = argparse.ArgumentParser(description="GĐ4 Gói 4.3: Đánh giá đa khái niệm ArXiv K=4 và K=2")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--encoders", nargs="+", default=["minilm", "bge", "e5", "qwen3"])
    parser.add_argument("--stage", type=str, default="confirm", choices=["selection", "confirm"])
    parser.add_argument("--k_list", nargs="+", type=int, default=[4, 2])
    parser.add_argument("--gate_quantile_space", type=str, default="query", choices=["query", "doc"])
    parser.add_argument("--cache_dir", type=str, default="cache_embeddings")
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints")
    parser.add_argument("--output_dir", type=str, default="results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    rows = []
    for K in args.k_list:
        concepts = CONCEPTS_K4[:K]
        for s in args.seeds:
            split_name = f"arxiv_multi_seed{s}"
            for enc in args.encoders:
                c_path = resolve_cache_path(args.cache_dir, split_name, enc)
                if not c_path:
                    continue
                k_path = resolve_ckpt_path(args.ckpt_dir, split_name, enc)
                rows.append(
                    evaluate_multi_unit(
                        cache_path=c_path,
                        ckpt_path=k_path,
                        seed=s,
                        encoder=enc,
                        stage=args.stage,
                        concepts=concepts,
                        gate_quantile_space=args.gate_quantile_space,
                    )
                )

    if not rows:
        print("[Cảnh báo] Chưa tìm thấy file cache cho arxiv_multi_seed*.")
        return

    df = pd.DataFrame(rows)
    out_csv = os.path.join(args.output_dir, f"multi_{args.stage}.csv")
    df.to_csv(out_csv, index=False)
    print(df.to_string(index=False))

    for K in args.k_list:
        sub = df[df["K"] == K]
        if sub.empty:
            continue
        diffs = sub["endpoint_diff_pp"].to_numpy(dtype=np.float64)
        mean_ep, lo, hi = paired_bootstrap_ci(diffs, n_boot=4000, scale=1.0)
        print(f"\n[K={K} | {len(sub)} units] Endpoint (min Compose - min Avg) = {mean_ep:+.3f} pp | 95% CI [{lo:+.3f}; {hi:+.3f}]")


if __name__ == "__main__":
    main()
