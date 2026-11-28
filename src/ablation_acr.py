#!/usr/bin/env python3
"""
src/ablation_acr.py
Giai đoạn 4 (Gói 4.4): Ablation về vai trò thực tế của ACR tại beta = 1 trên selection (val):
So sánh 3 biến thể:
  1. C2U với ACR đã huấn luyện
  2. C2U với ACR chưa huấn luyện (A ~ N(0,1) chuẩn hóa hàng, W = I)
  3. Công thức dạng đóng: normalize(q + eta * k_q * g)
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from acr_model import get_dictionary_size_N
from c2u_operator import C2UOperator, _compute_kq_from_angle
from eval_utils import (
    resolve_cache_path,
    resolve_ckpt_path,
    load_acr_weights,
    get_stage_data,
    dense_topk_indices,
    compute_per_query_metrics,
)


def main():
    parser = argparse.ArgumentParser(description="GĐ4 Gói 4.4: Ablation vai trò ACR ở beta=1 trên selection")
    parser.add_argument("--datasets", nargs="+", default=["tofu", "amazon", "mnli", "arxiv"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--encoders", nargs="+", default=["minilm", "bge", "e5", "qwen3"])
    parser.add_argument("--gate_quantile_space", type=str, default="query", choices=["query", "doc"])
    parser.add_argument("--cache_dir", type=str, default="cache_embeddings")
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints")
    parser.add_argument("--output_dir", type=str, default="results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    rows = []

    for ds in args.datasets:
        for s in args.seeds:
            split_name = f"{ds}_seed{s}"
            for enc in args.encoders:
                c_path = resolve_cache_path(args.cache_dir, split_name, enc)
                k_path = resolve_ckpt_path(args.ckpt_dir, split_name, enc)
                if not c_path or not k_path:
                    continue

                data = torch.load(c_path, map_location="cpu", weights_only=False)
                q_embs, q_t, q_lbl, _ = get_stage_data(data, "selection")
                idx_embs = data["index_embs"].float()
                idx_t = data["index_is_target"].long()
                idx_lbl = list(data["index_labels"])

                A_tr, W_tr, tau = load_acr_weights(k_path, device="cpu")
                d = idx_embs.shape[1]
                N = get_dictionary_size_N(ds, d)

                gen = torch.Generator().manual_seed(s * 100 + 42)
                A_un = F.normalize(torch.randn(N, d, generator=gen), p=2, dim=-1)
                W_un = torch.eye(d)

                op_tr = C2UOperator(A=A_tr, target_profile_embs=data["profile_embs"],
                                    calibration_control_embs=data["calib_doc_embs"],
                                    calibration_query_embs=data["calib_query_embs"],
                                    W=W_tr, tau=tau, beta=1.0,
                                    gate_quantile_space=args.gate_quantile_space)
                op_un = C2UOperator(A=A_un, target_profile_embs=data["profile_embs"],
                                    calibration_control_embs=data["calib_doc_embs"],
                                    calibration_query_embs=data["calib_query_embs"],
                                    W=W_un, tau=tau, beta=1.0,
                                    gate_quantile_space=args.gate_quantile_space)

                res_tr = op_tr.edit_query_detailed(q_embs)
                res_un = op_un.edit_query_detailed(q_embs)

                # Dạng đóng trực tiếp với hướng g thuần (không đi qua A' và W)
                g = op_tr.profile.g
                s_gate = op_tr.profile.gate_fn(q_embs)
                k_q = _compute_kq_from_angle(q_embs, g, op_tr.theta_req_factor * s_gate, eps_plane=op_tr.eps_plane)
                q_norm = F.normalize(q_embs, dim=-1)
                q_cf = q_norm.clone()
                act = (s_gate > 0) & (res_tr.eta > 0)
                if act.any():
                    q_cf[act] = F.normalize(q_norm[act] + res_tr.eta[act].unsqueeze(-1) * k_q[act].unsqueeze(-1) * g.unsqueeze(0), dim=-1)

                raw_met = compute_per_query_metrics(dense_topk_indices(q_embs, idx_embs, 10), q_embs, idx_embs, idx_t, idx_lbl, q_t, q_lbl)
                diff_tr_un = (res_tr.q_edited - res_un.q_edited).abs().max().item()
                diff_tr_cf = (res_tr.q_edited - q_cf).abs().max().item()

                for name, q_var in [("ACR-Trained", res_tr.q_edited), ("ACR-Untrained", res_un.q_edited), ("Closed-Form", q_cf)]:
                    met = compute_per_query_metrics(dense_topk_indices(q_var, idx_embs, 10), q_var, idx_embs, idx_t, idx_lbl, q_t, q_lbl)
                    rows.append({
                        "dataset": ds, "seed": s, "encoder": enc, "variant": name,
                        "h5_drop_pp": float(np.mean(raw_met["t_hit5"] - met["t_hit5"])) * 100.0,
                        "cn_pp": float(np.mean(raw_met["c_own_hit5"] - met["c_own_hit5"])) * 100.0,
                        "te_pp": float(np.mean(met["c_te5"] - raw_met["c_te5"])) * 100.0,
                        "max_q_diff_trained_vs_untrained": diff_tr_un,
                        "max_q_diff_trained_vs_closedform": diff_tr_cf,
                    })

    if not rows:
        print("[Cảnh báo] Không tìm thấy cặp (cache, ckpt) nào.")
        return

    df = pd.DataFrame(rows)
    out_csv = os.path.join(args.output_dir, "ablation_acr_selection.csv")
    df.to_csv(out_csv, index=False)
    print(df.to_string(index=False))
    print(f"[Saved] {out_csv}")


if __name__ == "__main__":
    main()
