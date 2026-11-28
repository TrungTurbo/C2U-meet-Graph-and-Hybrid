#!/usr/bin/env python3
"""
src/evaluate_transforms.py
Giai đoạn 4 (Gói 4.2): Kiểm tra biến đổi truy vấn một lượt (12 ô = 4 dataset x 3 điều kiện):
- Standard, Injection, Zou-noise (khóa trong configs/transforms.yaml).
- Chỉ đo trên selection (val) và confirmation (confirm), không chạm test.
- Chỉ số: Density Delta, CER Delta (B=512 token, tối đa 50 tài liệu, cho phép phần cuối), Attack utility (mất own-label Hit@10).
- Đếm số ô suppress (Density Delta < 0) và divergent (Density Delta < 0 và CER Delta > 0) trên giá trị chưa làm tròn.
"""

import os
import sys
import argparse
import yaml
from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from c2u_operator import C2UOperator
from encode_cache import get_encoder_config, encode_batch
from eval_utils import (
    resolve_cache_path,
    resolve_ckpt_path,
    load_acr_weights,
    load_jsonl,
    paired_bootstrap_ci,
)


def load_cer_tokenizer(preferred_name: str):
    """Tải tokenizer cố định cho CER (ưu tiên local cache, fallback regex tokenizer nếu offline)."""
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(preferred_name, local_files_only=True)
        return tok, preferred_name
    except Exception:
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(preferred_name)
            return tok, preferred_name
        except Exception:
            return None, "whitespace_fallback"


def count_tokens(text: str, tokenizer) -> int:
    if tokenizer is not None:
        return len(tokenizer.encode(text, add_special_tokens=False))
    return len(text.split())


def compute_cer_flags(
    sims: torch.Tensor,            # [B_ctrl, M]
    doc_tok_lens: List[int],
    index_is_target: np.ndarray,
    token_budget: int = 512,
    max_docs: int = 50,
    allow_partial: bool = True,
) -> np.ndarray:
    """Đóng gói tối đa max_docs trong ngân sách token_budget và kiểm tra có tài liệu target nào lọt vào không."""
    top_order = torch.topk(sims, k=min(max_docs, sims.shape[1]), dim=-1).indices.cpu().numpy()
    cer_flags = np.zeros(len(top_order), dtype=np.float64)
    for i, doc_indices in enumerate(top_order):
        used = 0
        has_target = False
        for d_idx in doc_indices:
            if used >= token_budget:
                break
            d_len = max(1, doc_tok_lens[int(d_idx)])
            if used + d_len <= token_budget:
                used += d_len
                if index_is_target[int(d_idx)] == 1:
                    has_target = True
            else:
                if allow_partial and (token_budget - used) > 0:
                    used = token_budget
                    if index_is_target[int(d_idx)] == 1:
                        has_target = True
                break
        cer_flags[i] = 1.0 if has_target else 0.0
    return cer_flags


def apply_text_transform(queries: List[str], cond_cfg: Dict[str, Any]) -> List[str]:
    ctype = cond_cfg.get("type", "identity")
    txt = cond_cfg.get("text", "")
    if ctype == "prefix":
        return [txt + q for q in queries]
    elif ctype == "suffix":
        return [q + txt for q in queries]
    return list(queries)


def main():
    parser = argparse.ArgumentParser(description="GĐ4 Gói 4.2: Biến đổi truy vấn (Standard/Injection/Zou-noise)")
    parser.add_argument("--datasets", nargs="+", default=["tofu", "amazon", "mnli", "arxiv"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--encoders", nargs="+", default=["minilm", "bge", "e5", "qwen3"])
    parser.add_argument("--stage", type=str, default="confirm", choices=["selection", "confirm"])
    parser.add_argument("--conditions", nargs="+", default=None, help="Danh sách điều kiện chạy (mặc định tất cả trong YAML)")
    parser.add_argument("--gate_quantile_space", type=str, default="query", choices=["query", "doc"])
    parser.add_argument("--transforms_cfg", type=str, default="configs/transforms.yaml")
    parser.add_argument("--splits_dir", type=str, default="processed_splits")
    parser.add_argument("--cache_dir", type=str, default="cache_embeddings")
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    with open(args.transforms_cfg, "r", encoding="utf-8") as f:
        t_cfg = yaml.safe_load(f)

    conditions = t_cfg["conditions"]
    if args.conditions:
        conditions = {k: v for k, v in conditions.items() if k in args.conditions}
    pack_cfg = t_cfg["cer_packing"]
    tokenizer, tok_name = load_cer_tokenizer(pack_cfg["preferred_tokenizer"])
    print(f"[CER Tokenizer] {tok_name} (budget={pack_cfg['token_budget']}, max_docs={pack_cfg['max_docs']})")

    from sentence_transformers import SentenceTransformer

    cell_records = []
    # Cache model theo encoder để không load lại
    loaded_models = {}

    for enc in args.encoders:
        enc_cfg = get_encoder_config(enc)
        for ds in args.datasets:
            for s in args.seeds:
                split_name = f"{ds}_seed{s}"
                c_path = resolve_cache_path(args.cache_dir, split_name, enc)
                if not c_path:
                    continue
                k_path = resolve_ckpt_path(args.ckpt_dir, split_name, enc)
                data = torch.load(c_path, map_location="cpu", weights_only=False)

                split_path = os.path.join(args.splits_dir, split_name)
                q_file = "val_queries.jsonl" if args.stage == "selection" else "confirm_queries.jsonl"
                raw_queries = [x["query"] for x in load_jsonl(os.path.join(split_path, q_file))]
                index_docs = [x["document"] for x in load_jsonl(os.path.join(split_path, "index_corpus.jsonl"))]
                doc_tok_lens = [count_tokens(doc, tokenizer) for doc in index_docs]

                if k_path and os.path.exists(k_path):
                    A, W, tau = load_acr_weights(k_path, device="cpu")
                else:
                    from acr_model import get_dictionary_size_N
                    d = data["index_embs"].shape[1]
                    N = get_dictionary_size_N(ds, d)
                    A = F.normalize(torch.randn(N, d), p=2, dim=-1)
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
                    gate_quantile_space=args.gate_quantile_space,
                    device="cpu",
                )

                index_embs = F.normalize(data["index_embs"].float(), dim=-1)
                idx_t = data["index_is_target"].numpy()
                idx_labels = list(data["index_labels"])
                q_t = data["val_q_is_target"].numpy() if args.stage == "selection" else data["confirm_q_is_target"].numpy()
                q_labels = list(data["val_q_labels"] if args.stage == "selection" else data["confirm_q_labels"])

                t_mask = (q_t == 1)
                c_mask = (q_t == 0)

                for cond_name, cond_cfg in conditions.items():
                    if cond_name == "standard":
                        q_raw_emb = (data["val_q_embs"] if args.stage == "selection" else data["confirm_q_embs"]).float()
                    else:
                        if enc not in loaded_models:
                            try:
                                loaded_models[enc] = SentenceTransformer(
                                    enc_cfg["model_name_or_path"], revision=enc_cfg["revision"],
                                    device=args.device, local_files_only=True
                                )
                            except Exception:
                                loaded_models[enc] = SentenceTransformer(
                                    enc_cfg["model_name_or_path"], revision=enc_cfg["revision"],
                                    device=args.device
                                )
                        trans_texts = apply_text_transform(raw_queries, cond_cfg)
                        enc_bs = 8 if "qwen3" in enc.lower() else 32
                        q_raw_emb = encode_batch(
                            loaded_models[enc],
                            trans_texts,
                            prefix=enc_cfg.get("query_prompt", ""),
                            prompt_name=enc_cfg.get("prompt_name"),
                            batch_size=enc_bs,
                            device=args.device,
                        ).cpu()

                    q_c2u_emb, _ = op.edit_query(q_raw_emb)

                    sims_raw = torch.matmul(F.normalize(q_raw_emb, dim=-1), index_embs.t())
                    sims_c2u = torch.matmul(F.normalize(q_c2u_emb, dim=-1), index_embs.t())

                    top5_raw = torch.topk(sims_raw[t_mask], k=5, dim=-1).indices.numpy()
                    top5_c2u = torch.topk(sims_c2u[t_mask], k=5, dim=-1).indices.numpy()
                    dens_raw = np.mean(idx_t[top5_raw] == 1, axis=1)
                    dens_c2u = np.mean(idx_t[top5_c2u] == 1, axis=1)

                    cer_raw = compute_cer_flags(
                        sims_raw[c_mask], doc_tok_lens, idx_t,
                        token_budget=pack_cfg["token_budget"], max_docs=pack_cfg["max_docs"],
                        allow_partial=pack_cfg["allow_partial_last_doc"]
                    )
                    cer_c2u = compute_cer_flags(
                        sims_c2u[c_mask], doc_tok_lens, idx_t,
                        token_budget=pack_cfg["token_budget"], max_docs=pack_cfg["max_docs"],
                        allow_partial=pack_cfg["allow_partial_last_doc"]
                    )

                    top10_raw = torch.topk(sims_raw[c_mask], k=10, dim=-1).indices.numpy()
                    top10_c2u = torch.topk(sims_c2u[c_mask], k=10, dim=-1).indices.numpy()
                    c_lbls = [q_labels[i] for i in np.where(c_mask)[0]]
                    h10_raw = np.array([1.0 if any(idx_labels[j] == c_lbls[i] and idx_t[j] == 0 for j in top10_raw[i]) else 0.0 for i in range(len(c_lbls))])
                    h10_c2u = np.array([1.0 if any(idx_labels[j] == c_lbls[i] and idx_t[j] == 0 for j in top10_c2u[i]) else 0.0 for i in range(len(c_lbls))])

                    cell_records.append({
                        "dataset": ds, "condition": cond_name, "seed": s, "encoder": enc,
                        "tokenizer": tok_name,
                        "dens_diff": dens_c2u - dens_raw,
                        "cer_diff": cer_c2u - cer_raw,
                        "att_util_diff": h10_raw - h10_c2u,
                    })

        # Giải phóng encoder khỏi VRAM ngay sau khi xong hết các dataset/seed của encoder này
        if enc in loaded_models:
            del loaded_models[enc]
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if not cell_records:
        print("[Cảnh báo] Không có unit nào để chạy evaluate_transforms.")
        return

    summary = []
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for r in cell_records:
        grouped.setdefault((r["dataset"], r["condition"]), []).append(r)

    n_suppress, n_divergent = 0, 0
    for (ds, cond), items in sorted(grouped.items()):
        d_arr = np.concatenate([x["dens_diff"] for x in items])
        c_arr = np.concatenate([x["cer_diff"] for x in items])
        u_arr = np.concatenate([x["att_util_diff"] for x in items])

        d_m, d_lo, d_hi = paired_bootstrap_ci(d_arr)
        c_m, c_lo, c_hi = paired_bootstrap_ci(c_arr)
        u_m, u_lo, u_hi = paired_bootstrap_ci(u_arr)

        is_sup = d_m < 0.0
        is_div = is_sup and (c_m > 0.0)
        n_suppress += int(is_sup)
        n_divergent += int(is_div)

        summary.append({
            "stage": args.stage, "dataset": ds, "condition": cond, "tokenizer": items[0]["tokenizer"],
            "density_delta_pp": d_m, "density_ci_lo": d_lo, "density_ci_hi": d_hi,
            "cer_delta_pp": c_m, "cer_ci_lo": c_lo, "cer_ci_hi": c_hi,
            "attack_util_h10_drop_pp": u_m, "att_ci_lo": u_lo, "att_ci_hi": u_hi,
            "suppress": is_sup, "divergent": is_div,
        })

    df_sum = pd.DataFrame(summary)
    out_csv = os.path.join(args.output_dir, f"transforms_{args.stage}.csv")
    df_sum.to_csv(out_csv, index=False)
    print(df_sum.to_string(index=False))
    print(f"\n[Tổng kết {len(summary)} ô] Suppress: {n_suppress}/{len(summary)} | Divergent: {n_divergent}/{len(summary)}")
    print(f"[Saved] {out_csv}")


if __name__ == "__main__":
    main()
