#!/usr/bin/env python3
"""
src/evaluate_dense.py
Giai đoạn 4 (Gói 4.1): Tái lập chuẩn C2U trên Dense RAG (selection & confirmation).
- Không bao giờ động vào test_queries.jsonl / test_q_embs.
- Chạy C2U và các baseline P1–P2 phân tách theo Access Class:
  * Profile-only: Raw, Random, Delta-Edit
  * Labeled calibration (400 mẫu): INLP, LEACE, RAG-Emb-Ctrl
  * Corpus/membership oracle (cận trên): Filter-and-Refill, Rank-Router
- Hỗ trợ chọn gate_quantile_space in {query, doc} trên selection (CN <= 0, TE <= 0)
  rồi xác nhận trên confirmation và ghi configs/dense_lock.yaml.
- Tổng hợp 4.000 paired bootstrap 95% CI theo dataset và báo cáo từng encoder.
"""

import os
import sys
import argparse
import yaml
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from concept_erasure import LeaceEraser

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from c2u_operator import C2UOperator
from eval_utils import (
    resolve_cache_path,
    resolve_ckpt_path,
    load_acr_weights,
    get_stage_data,
    compute_per_query_metrics,
    dense_topk_indices,
    paired_bootstrap_ci,
)


ACCESS_CLASSES = {
    "Raw": "profile_only",
    "C2U": "profile_only",
    "Random": "profile_only",
    "Delta-Edit": "profile_only",
    "INLP": "labeled_calib_400",
    "LEACE": "labeled_calib_400",
    "RAG-Emb-Ctrl": "labeled_calib_400",
    "Filter-and-Refill": "corpus_oracle",
    "Rank-Router": "corpus_oracle",
}


def apply_random_baseline(q_embs: torch.Tensor, target_mean_shift: float, seed: int = 42) -> torch.Tensor:
    """Random baseline cùng độ lớn dịch chuyển trung bình với C2U."""
    gen = torch.Generator(device=q_embs.device).manual_seed(seed)
    noise = torch.randn(q_embs.shape, generator=gen, device=q_embs.device, dtype=torch.float32)
    noise = F.normalize(noise, p=2, dim=-1)
    q_norm = F.normalize(q_embs.float(), p=2, dim=-1)
    # Tìm bước alpha sao cho ||q' - q|| tương đồng độ lệch C2U
    step = max(0.0, target_mean_shift)
    return F.normalize(q_norm + step * noise, p=2, dim=-1)


def apply_delta_edit(q_embs: torch.Tensor, op: C2UOperator, step_scale: float = 0.2) -> torch.Tensor:
    """Delta-Edit (profile-only): dịch chuyển tuyến tính đồng đều theo hướng g không qua cổng hiệu chuẩn."""
    q_norm = F.normalize(q_embs.float(), p=2, dim=-1)
    g = op.profile.g.unsqueeze(0)
    return F.normalize(q_norm + step_scale * g, p=2, dim=-1)


def fit_inlp_projection(X_np: np.ndarray, y_np: np.ndarray, max_iters: int = 10, seed: int = 42) -> np.ndarray:
    """Huấn luyện phép chiếu không gian rỗng lặp INLP trên 400 mẫu calibration_labeled."""
    d = X_np.shape[1]
    P = np.eye(d, dtype=np.float64)
    X_curr = X_np.astype(np.float64).copy()
    if len(np.unique(y_np)) < 2:
        return P
    for it in range(max_iters):
        clf = LogisticRegression(max_iter=500, random_state=seed + it, C=1.0)
        clf.fit(X_curr, y_np)
        acc = clf.score(X_curr, y_np)
        if acc <= 0.55 and it > 0:
            break
        w = clf.coef_[0]
        norm_w = np.linalg.norm(w)
        if norm_w <= 1e-12:
            break
        w = w / norm_w
        P_step = np.eye(d) - np.outer(w, w)
        X_curr = X_curr @ P_step
        P = P @ P_step
    return P


def apply_inlp(q_embs: torch.Tensor, calib_x: torch.Tensor, calib_y: torch.Tensor, seed: int = 42) -> torch.Tensor:
    P = fit_inlp_projection(calib_x.cpu().numpy(), calib_y.cpu().numpy(), max_iters=10, seed=seed)
    P_t = torch.tensor(P, dtype=torch.float32, device=q_embs.device)
    q_norm = F.normalize(q_embs.float(), p=2, dim=-1)
    return F.normalize(torch.matmul(q_norm, P_t), p=2, dim=-1)


def apply_leace(q_embs: torch.Tensor, calib_x: torch.Tensor, calib_y: torch.Tensor) -> torch.Tensor:
    x_cpu = F.normalize(calib_x.float(), p=2, dim=-1).cpu()
    z_cpu = calib_y.float().cpu().unsqueeze(-1)
    eraser = LeaceEraser.fit(x_cpu, z_cpu)
    q_cpu = F.normalize(q_embs.float(), p=2, dim=-1).cpu()
    erased = eraser(q_cpu).to(q_embs.device)
    return F.normalize(erased, p=2, dim=-1)


def apply_rag_emb_ctrl(
    q_embs: torch.Tensor,
    calib_x: torch.Tensor,
    calib_y: torch.Tensor,
    max_angle: float = 0.2,
    seed: int = 42,
) -> torch.Tensor:
    """RAG-Emb-Ctrl adaptation: lái theo hướng ngược bộ phân loại tuyến tính trên 400 mẫu có nhãn, giữ trần 0.2 rad."""
    X_np = F.normalize(calib_x.float(), p=2, dim=-1).cpu().numpy()
    y_np = calib_y.cpu().numpy()
    clf = LogisticRegression(max_iter=500, random_state=seed, C=1.0, class_weight="balanced")
    clf.fit(X_np, y_np)
    w = torch.tensor(-clf.coef_[0], dtype=torch.float32, device=q_embs.device)
    w = F.normalize(w, p=2, dim=-1)
    probs = torch.sigmoid(torch.tensor(clf.decision_function(F.normalize(q_embs.float(), p=2, dim=-1).cpu().numpy()),
                                       dtype=torch.float32, device=q_embs.device))
    # Áp dụng cổng mềm dựa trên xác suất >= 0.5 và giới hạn góc 0.2 rad
    gate = torch.clamp((probs - 0.5) * 2.0, 0.0, 1.0)
    step = torch.tan(torch.tensor(max_angle, device=q_embs.device)) * gate
    q_norm = F.normalize(q_embs.float(), p=2, dim=-1)
    return F.normalize(q_norm + step.unsqueeze(-1) * w.unsqueeze(0), p=2, dim=-1)


def oracle_filter_and_refill(
    q_embs: torch.Tensor,
    index_embs: torch.Tensor,
    index_is_target: torch.Tensor,
    max_k: int = 10,
) -> torch.Tensor:
    """Filter-and-Refill (corpus oracle): loại toàn bộ tài liệu target khỏi bảng xếp hạng và nạp bù control."""
    sims = torch.matmul(F.normalize(q_embs.float(), dim=-1), F.normalize(index_embs.float(), dim=-1).t())
    masked_sims = sims.clone()
    masked_sims[:, index_is_target == 1] = -1e9
    return torch.topk(masked_sims, k=min(max_k, index_embs.shape[0]), dim=-1).indices


def oracle_rank_router(
    q_embs: torch.Tensor,
    index_embs: torch.Tensor,
    index_is_target: torch.Tensor,
    calib_x: torch.Tensor,
    calib_y: torch.Tensor,
    max_k: int = 10,
    seed: int = 42,
) -> torch.Tensor:
    """Rank-Router (corpus oracle): định tuyến các truy vấn bị phát hiện sang chỉ mục chỉ gồm control."""
    X_np = F.normalize(calib_x.float(), p=2, dim=-1).cpu().numpy()
    y_np = calib_y.cpu().numpy()
    clf = LogisticRegression(max_iter=500, random_state=seed)
    clf.fit(X_np, y_np)
    q_np = F.normalize(q_embs.float(), p=2, dim=-1).cpu().numpy()
    is_flagged = torch.tensor(clf.predict(q_np) == 1, device=q_embs.device)

    sims = torch.matmul(F.normalize(q_embs.float(), dim=-1), F.normalize(index_embs.float(), dim=-1).t())
    routed_sims = sims.clone()
    mask_2d = is_flagged.unsqueeze(1) & (index_is_target == 1).unsqueeze(0)
    routed_sims[mask_2d] = -1e9
    return torch.topk(routed_sims, k=min(max_k, index_embs.shape[0]), dim=-1).indices


def evaluate_unit(
    cache_path: str,
    ckpt_path: Optional[str],
    dataset: str,
    seed: int,
    encoder: str,
    stage: str = "selection",
    gate_quantile_space: str = "query",
    methods: Optional[List[str]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, np.ndarray]]]:
    """Đánh giá một unit và trả về danh sách dòng tóm tắt + dữ liệu per-query để bootstrap."""
    data = torch.load(cache_path, map_location="cpu", weights_only=False)
    q_embs, q_is_target, q_labels, _ = get_stage_data(data, stage)
    index_embs = data["index_embs"].float()
    index_is_target = data["index_is_target"].long()
    index_labels = list(data["index_labels"])

    if ckpt_path and os.path.exists(ckpt_path):
        A, W, tau = load_acr_weights(ckpt_path, device="cpu")
    else:
        # Fallback hiếm nếu thiếu ckpt (chỉ nên dùng trong test nhanh)
        from acr_model import get_dictionary_size_N
        d = index_embs.shape[1]
        N = get_dictionary_size_N(dataset, d)
        gen = torch.Generator().manual_seed(42)
        A = F.normalize(torch.randn(N, d, generator=gen), p=2, dim=-1)
        W = torch.eye(d)
        tau = 0.1

    op = C2UOperator(
        A=A,
        target_profile_embs=data["profile_embs"],
        calibration_control_embs=data["calib_doc_embs"],
        calibration_query_embs=data["calib_query_embs"],
        W=W,
        tau=tau,
        beta=1.0,
        gate_quantile_space=gate_quantile_space,
        device="cpu",
    )

    if methods is None:
        methods = ["Raw", "C2U", "Random", "Delta-Edit", "INLP", "LEACE", "RAG-Emb-Ctrl", "Filter-and-Refill", "Rank-Router"]

    c2u_res = op.edit_query_detailed(q_embs)
    c2u_q = c2u_res.q_edited
    mean_shift_norm = torch.norm(c2u_q - F.normalize(q_embs, dim=-1), dim=-1).mean().item()

    calib_x = data["calib_labeled_embs"].float()
    calib_y = data["calib_labeled_is_target"].long()

    raw_top = dense_topk_indices(q_embs, index_embs, max_k=10)
    raw_metrics = compute_per_query_metrics(raw_top, q_embs, index_embs, index_is_target, index_labels, q_is_target, q_labels)

    rows = []
    per_query_by_method: Dict[str, Dict[str, np.ndarray]] = {}

    for m in methods:
        if m == "Raw":
            qm, topk = q_embs, raw_top
        elif m == "C2U":
            qm = c2u_q
            topk = dense_topk_indices(qm, index_embs, max_k=10)
        elif m == "Random":
            qm = apply_random_baseline(q_embs, mean_shift_norm, seed=seed * 100 + 7)
            topk = dense_topk_indices(qm, index_embs, max_k=10)
        elif m == "Delta-Edit":
            qm = apply_delta_edit(q_embs, op, step_scale=max(mean_shift_norm, 0.1))
            topk = dense_topk_indices(qm, index_embs, max_k=10)
        elif m == "INLP":
            qm = apply_inlp(q_embs, calib_x, calib_y, seed=seed)
            topk = dense_topk_indices(qm, index_embs, max_k=10)
        elif m == "LEACE":
            qm = apply_leace(q_embs, calib_x, calib_y)
            topk = dense_topk_indices(qm, index_embs, max_k=10)
        elif m == "RAG-Emb-Ctrl":
            qm = apply_rag_emb_ctrl(q_embs, calib_x, calib_y, seed=seed)
            topk = dense_topk_indices(qm, index_embs, max_k=10)
        elif m == "Filter-and-Refill":
            qm = q_embs
            topk = oracle_filter_and_refill(q_embs, index_embs, index_is_target, max_k=10)
        elif m == "Rank-Router":
            qm = q_embs
            topk = oracle_rank_router(q_embs, index_embs, index_is_target, calib_x, calib_y, max_k=10, seed=seed)
        else:
            raise ValueError(f"Unknown method: {m}")

        met = compute_per_query_metrics(topk, qm, index_embs, index_is_target, index_labels, q_is_target, q_labels)
        per_query_by_method[m] = met

        h5_drop = float(np.mean(raw_metrics["t_hit5"] - met["t_hit5"])) * 100.0
        tc = float(np.mean(raw_metrics["t_cos"] - met["t_cos"])) * 100.0
        cn = float(np.mean(raw_metrics["c_own_hit5"] - met["c_own_hit5"])) * 100.0
        feas = raw_metrics["c_feasible"]
        cn_feas = float(np.mean(raw_metrics["c_own_hit5"][feas] - met["c_own_hit5"][feas])) * 100.0 if feas.any() else 0.0
        p5_drop = float(np.mean(raw_metrics["c_own_p5"] - met["c_own_p5"])) * 100.0
        te = float(np.mean(met["c_te5"] - raw_metrics["c_te5"])) * 100.0
        raw_ctrl_h5 = float(np.mean(raw_metrics["c_own_hit5"])) * 100.0

        rows.append({
            "stage": stage,
            "dataset": dataset,
            "seed": seed,
            "encoder": encoder,
            "gate_quantile_space": gate_quantile_space,
            "gate_mode": op.profile.gate_mode,
            "residual_branch": op.profile.residual_branch,
            "method": m,
            "access_class": ACCESS_CLASSES[m],
            "raw_h5_pct": float(np.mean(raw_metrics["t_hit5"])) * 100.0,
            "method_h5_pct": float(np.mean(met["t_hit5"])) * 100.0,
            "h5_drop_pp": h5_drop,
            "tc_drop_x100": tc,
            "raw_ctrl_h5_pct": raw_ctrl_h5,
            "cn_pp": cn,
            "cn_feasible_pp": cn_feas,
            "ctrl_p5_drop_pp": p5_drop,
            "te_pp": te,
            "relaxed_global_control": (dataset == "arxiv"),
            "saturation_flag": (raw_ctrl_h5 >= 99.0),
        })

    return rows, per_query_by_method


def aggregate_bootstrap(
    all_pq: Dict[Tuple[str, str], Dict[str, List[np.ndarray]]],
    n_boot: int = 4000,
) -> pd.DataFrame:
    """Tổng hợp theo dataset (gộp encoder và seed) với 4.000 paired bootstrap."""
    summary_rows = []
    for (dataset, method), arrays in sorted(all_pq.items()):
        if method == "Raw":
            continue
        raw_arrays = all_pq[(dataset, "Raw")]
        t_h5_diff = np.concatenate(raw_arrays["t_hit5"]) - np.concatenate(arrays["t_hit5"])
        t_cos_diff = np.concatenate(raw_arrays["t_cos"]) - np.concatenate(arrays["t_cos"])
        c_cn_diff = np.concatenate(raw_arrays["c_own_hit5"]) - np.concatenate(arrays["c_own_hit5"])
        c_p5_diff = np.concatenate(raw_arrays["c_own_p5"]) - np.concatenate(arrays["c_own_p5"])
        c_te_diff = np.concatenate(arrays["c_te5"]) - np.concatenate(raw_arrays["c_te5"])
        feas = np.concatenate(raw_arrays["c_feasible"])
        c_feas_diff = (np.concatenate(raw_arrays["c_own_hit5"])[feas] - np.concatenate(arrays["c_own_hit5"])[feas]) if feas.any() else np.array([0.0])

        h5_m, h5_lo, h5_hi = paired_bootstrap_ci(t_h5_diff, n_boot=n_boot)
        tc_m, tc_lo, tc_hi = paired_bootstrap_ci(t_cos_diff, n_boot=n_boot)
        cn_m, cn_lo, cn_hi = paired_bootstrap_ci(c_cn_diff, n_boot=n_boot)
        cnf_m, cnf_lo, cnf_hi = paired_bootstrap_ci(c_feas_diff, n_boot=n_boot)
        p5_m, p5_lo, p5_hi = paired_bootstrap_ci(c_p5_diff, n_boot=n_boot)
        te_m, te_lo, te_hi = paired_bootstrap_ci(c_te_diff, n_boot=n_boot)

        summary_rows.append({
            "dataset": dataset,
            "method": method,
            "access_class": ACCESS_CLASSES[method],
            "h5_drop_pp": h5_m, "h5_ci_lo": h5_lo, "h5_ci_hi": h5_hi,
            "tc_x100": tc_m, "tc_ci_lo": tc_lo, "tc_ci_hi": tc_hi,
            "cn_pp": cn_m, "cn_ci_lo": cn_lo, "cn_ci_hi": cn_hi,
            "cn_feasible_pp": cnf_m, "cn_feas_ci_lo": cnf_lo, "cn_feas_ci_hi": cnf_hi,
            "p5_drop_pp": p5_m, "p5_ci_lo": p5_lo, "p5_ci_hi": p5_hi,
            "te_pp": te_m, "te_ci_lo": te_lo, "te_ci_hi": te_hi,
        })
    return pd.DataFrame(summary_rows)


def run_stage(
    datasets: List[str],
    seeds: List[int],
    encoders: List[str],
    cache_dir: str,
    ckpt_dir: str,
    stage: str,
    gate_quantile_space: str,
    methods: Optional[List[str]] = None,
    n_boot: int = 4000,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    unit_rows = []
    all_pq: Dict[Tuple[str, str], Dict[str, List[np.ndarray]]] = {}

    for ds in datasets:
        for s in seeds:
            split_name = f"{ds}_seed{s}"
            for enc in encoders:
                c_path = resolve_cache_path(cache_dir, split_name, enc)
                if not c_path:
                    continue
                k_path = resolve_ckpt_path(ckpt_dir, split_name, enc)
                rows, pq_map = evaluate_unit(
                    cache_path=c_path,
                    ckpt_path=k_path,
                    dataset=ds,
                    seed=s,
                    encoder=enc,
                    stage=stage,
                    gate_quantile_space=gate_quantile_space,
                    methods=methods,
                )
                unit_rows.extend(rows)
                for m_name, met in pq_map.items():
                    key = (ds, m_name)
                    if key not in all_pq:
                        all_pq[key] = {k: [] for k in met.keys()}
                    for k, arr in met.items():
                        all_pq[key][k].append(arr)

    df_units = pd.DataFrame(unit_rows)
    df_boot = aggregate_bootstrap(all_pq, n_boot=n_boot) if unit_rows else pd.DataFrame()
    return df_units, df_boot


def main():
    parser = argparse.ArgumentParser(description="GĐ4 Gói 4.1: Tái lập Dense C2U (selection & confirm)")
    parser.add_argument("--datasets", nargs="+", default=["tofu", "amazon", "mnli", "arxiv"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--encoders", nargs="+", default=["minilm", "bge", "e5", "qwen3"])
    parser.add_argument("--cache_dir", type=str, default="cache_embeddings")
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--lock_path", type=str, default="configs/dense_lock.yaml")
    parser.add_argument("--stage", type=str, default="both", choices=["selection", "confirm", "both"])
    parser.add_argument("--gate_quantile_space", type=str, default="auto", choices=["auto", "query", "doc"])
    parser.add_argument("--n_boot", type=int, default=4000)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.lock_path), exist_ok=True)

    chosen_space = args.gate_quantile_space
    if chosen_space == "auto":
        # Chọn gate_quantile_space trên selection với ràng buộc CN <= 0 và TE <= 0
        candidates = {}
        for sp in ["query", "doc"]:
            df_u, _ = run_stage(
                args.datasets, args.seeds, args.encoders, args.cache_dir, args.ckpt_dir,
                stage="selection", gate_quantile_space=sp, methods=["Raw", "C2U"], n_boot=args.n_boot
            )
            if not df_u.empty:
                c2u_u = df_u[df_u["method"] == "C2U"]
                mean_h5 = float(c2u_u["h5_drop_pp"].mean())
                max_cn = float(c2u_u.groupby("dataset")["cn_pp"].mean().max())
                max_te = float(c2u_u.groupby("dataset")["te_pp"].mean().max())
                candidates[sp] = {"h5_drop": mean_h5, "max_cn": max_cn, "max_te": max_te}
                print(f"[Select space={sp}] H5 Drop={mean_h5:+.2f} pp | max dataset CN={max_cn:+.4f} | max dataset TE={max_te:+.4f}")

        feasible = {sp: v for sp, v in candidates.items() if v["max_cn"] <= 1e-9 and v["max_te"] <= 1e-9}
        if feasible:
            chosen_space = max(feasible.keys(), key=lambda sp: feasible[sp]["h5_drop"])
        else:
            chosen_space = "query"
        print(f"[Lock] Chọn gate_quantile_space = '{chosen_space}'")

    stages_to_run = ["selection", "confirm"] if args.stage == "both" else [args.stage]
    for st in stages_to_run:
        df_units, df_boot = run_stage(
            args.datasets, args.seeds, args.encoders, args.cache_dir, args.ckpt_dir,
            stage=st, gate_quantile_space=chosen_space, n_boot=args.n_boot
        )
        if df_units.empty:
            print(f"[Cảnh báo] Không tìm thấy unit nào cho stage={st}")
            continue
        unit_csv = os.path.join(args.output_dir, f"dense_{st}.csv")
        boot_csv = os.path.join(args.output_dir, f"dense_{st}_bootstrap.csv")
        df_units.to_csv(unit_csv, index=False)
        df_boot.to_csv(boot_csv, index=False)
        print(f"\n=== DENSE {st.upper()} (gate_quantile_space={chosen_space}) ===")
        print(df_boot[["dataset", "method", "h5_drop_pp", "h5_ci_lo", "h5_ci_hi", "cn_pp", "te_pp"]].to_string(index=False))

    # Ghi configs/dense_lock.yaml
    lock_cfg = {
        "method": "C2U-Dense",
        "beta": 1.0,
        "theta_req_factor": 0.3,
        "trust_angle_cap_rad": 0.2,
        "trust_cos_floor": 0.98,
        "eps_plane_rad": 1e-6,
        "profile_size": 20,
        "gate_quantile_space": chosen_space,
        "datasets": args.datasets,
        "encoders": args.encoders,
        "seeds": args.seeds,
    }
    with open(args.lock_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(lock_cfg, f, sort_keys=False, allow_unicode=True)
    print(f"[Lock] Đã ghi cấu hình khóa Dense vào {args.lock_path}")


if __name__ == "__main__":
    main()
