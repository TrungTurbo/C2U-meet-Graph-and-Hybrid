#!/usr/bin/env python3
"""
evaluate.py
Đánh giá hiệu năng Top-5 Retrieval của Raw Query vs C2U Query theo Mục 4.E:
- Target Hit@5 Drop (càng cao càng tốt, ^): Mức giảm tỷ lệ tài liệu Target lọt vào Top-5 của test_target
- Control Utility Drop / CN (v, mục tiêu <= 0): Mức suy giảm độ chính xác trên test_control
- Target Exposure / TE (v, mục tiêu <= 0): Mức thay đổi tỷ lệ tài liệu Target vô tình lọt vào Top-5 của test_control
Kho chỉ mục dùng chung gồm 100 target docs + 300 control docs.
"""

import os
import sys

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import json
import argparse
import pandas as pd
import torch
import torch.nn.functional as F

try:
    from acr_model import ACRModel
    from c2u_operator import C2UOperator
except ImportError:
    from src.acr_model import ACRModel
    from src.c2u_operator import C2UOperator


def evaluate_retrieval(
    q_embs: torch.Tensor,
    corpus_embs: torch.Tensor,
    corpus_is_target: torch.Tensor,
    gt_docs: torch.Tensor = None,
    k: int = 5
) -> dict:
    """
    Truy xuất Top-K qua cosine similarity:
    - target_hit: tỷ lệ query có ít nhất 1 target doc trong top-k
    - target_exposure: tỷ lệ tài liệu target trong top-k (hoặc số lượng target doc trong top-k)
    - utility: tỷ lệ truy xuất đúng tài liệu gốc (ground truth hit)
    """
    q_norm = F.normalize(q_embs, p=2, dim=-1)
    c_norm = F.normalize(corpus_embs, p=2, dim=-1)
    
    # Cosine similarity matrix: (num_q, num_corpus)
    sims = torch.matmul(q_norm, c_norm.t())
    
    # Top-K
    _, topk_idx = torch.topk(sims, k=k, dim=-1)  # (num_q, k)
    
    # Lấy cờ is_target
    topk_is_target = corpus_is_target[topk_idx]  # (num_q, k)
    
    # Target Hit: có >= 1 tài liệu target trong top-k
    target_hits = (topk_is_target.sum(dim=-1) > 0).float()
    target_hit_rate = target_hits.mean().item()
    
    # Target Exposure: tỷ lệ tài liệu target trong top-k (số lượng / k)
    target_exposure = (topk_is_target.float().sum(dim=-1) / float(k)).mean().item()
    
    # Utility: kiểm tra top-k docs retrieved có khớp doc ground truth không
    utility_rate = 0.0
    if gt_docs is not None:
        gt_norm = F.normalize(gt_docs, p=2, dim=-1)
        topk_docs = c_norm[topk_idx]  # (num_q, k, d)
        gt_exp = gt_norm.unsqueeze(1) # (num_q, 1, d)
        cos_to_gt = torch.sum(topk_docs * gt_exp, dim=-1)  # (num_q, k)
        # Cosine >= 0.98 xem như chính là document đó
        found_gt = (cos_to_gt >= 0.98).sum(dim=-1) > 0
        utility_rate = found_gt.float().mean().item()
        
    return {
        "target_hit": target_hit_rate,
        "target_exposure": target_exposure,
        "utility": utility_rate
    }


def run_evaluation(
    cache_dir: str,
    checkpoint_path: str,
    output_csv: str = "results.csv",
    dataset_name: str = "tofu",
    model_name: str = "all-MiniLM-L6-v2",
    k: int = 5,
    device: str = "cpu"
):
    print(f"\n[Evaluate] Đánh giá C2U trên tập '{dataset_name}' với mô hình '{model_name}'...")
    
    # 1. Nạp cache embeddings
    corpus_emb = torch.load(os.path.join(cache_dir, "corpus_emb.pt"), map_location=device)
    target_prof_emb = torch.load(os.path.join(cache_dir, "target_profile_emb.pt"), map_location=device)
    cal_ctrl_emb = torch.load(os.path.join(cache_dir, "calibration_control_emb.pt"), map_location=device)
    
    tt_q_emb = torch.load(os.path.join(cache_dir, "test_target_q_emb.pt"), map_location=device)
    tt_doc_emb = torch.load(os.path.join(cache_dir, "test_target_doc_emb.pt"), map_location=device)
    tc_q_emb = torch.load(os.path.join(cache_dir, "test_control_q_emb.pt"), map_location=device)
    tc_doc_emb = torch.load(os.path.join(cache_dir, "test_control_doc_emb.pt"), map_location=device)
    
    with open(os.path.join(cache_dir, "corpus_meta.json"), "r", encoding="utf-8") as f:
        corpus_meta = json.load(f)
    corpus_is_target = torch.tensor([x.get("is_target", 0) for x in corpus_meta], dtype=torch.long, device=device)
    
    # 2. Nạp Checkpoint ACR
    print(f"[Evaluate] Đang nạp ACR checkpoint từ {checkpoint_path}...")
    ckpt = torch.load(checkpoint_path, map_location=device)
    d = ckpt.get("d", corpus_emb.shape[-1])
    N = ckpt.get("N", 768)
    tau = ckpt.get("tau", 0.1)
    
    acr = ACRModel(d=d, N=N, tau=tau).to(device)
    acr.load_state_dict(ckpt["model_state_dict"])
    acr.eval()
    
    learned_A = acr.get_normalized_dictionary().detach()
    
    # 3. Khởi tạo toán tử C2U
    operator = C2UOperator(
        A=learned_A,
        target_profile_embs=target_prof_emb,
        calibration_control_embs=cal_ctrl_emb,
        tau=tau,
        device=device
    )
    
    # 4. Áp dụng C2U chỉnh sửa query
    tt_q_c2u, tt_gate = operator.edit_query(tt_q_emb)
    tc_q_c2u, tc_gate = operator.edit_query(tc_q_emb)
    
    avg_gate_t = tt_gate.mean().item()
    avg_gate_c = tc_gate.mean().item()
    print(f"  Cổng kích hoạt trung bình s(q) - Target: {avg_gate_t:.4f}")
    print(f"  Cổng kích hoạt trung bình s(q) - Control: {avg_gate_c:.4f}")
    
    # 5. Đánh giá Top-K Retrieval
    # 5a. test_target: đo lường mức độ unlearning target
    raw_target = evaluate_retrieval(tt_q_emb, corpus_emb, corpus_is_target, tt_doc_emb, k=k)
    c2u_target = evaluate_retrieval(tt_q_c2u, corpus_emb, corpus_is_target, tt_doc_emb, k=k)
    
    # 5b. test_control: đo lường mức độ bảo toàn control utility và target exposure trên control
    raw_control = evaluate_retrieval(tc_q_emb, corpus_emb, corpus_is_target, tc_doc_emb, k=k)
    c2u_control = evaluate_retrieval(tc_q_c2u, corpus_emb, corpus_is_target, tc_doc_emb, k=k)
    
    # Tính các chỉ số Drop theo định nghĩa bài báo:
    # Target Hit@5 Drop (^): Mức giảm tỷ lệ tài liệu target lọt vào Top-5 của test_target
    target_hit_drop = raw_target["target_hit"] - c2u_target["target_hit"]
    
    # Control Utility Drop / CN (v, <= 0): Mức suy giảm độ chính xác trên test_control
    control_utility_drop = raw_control["utility"] - c2u_control["utility"]
    
    # Target Exposure / TE (v, <= 0): Mức thay đổi tỷ lệ tài liệu target vô tình lọt vào Top-5 của test_control
    # C2U TE - Raw TE (mục tiêu <= 0 tức không làm tăng target trong control queries)
    te_drop = c2u_control["target_exposure"] - raw_control["target_exposure"]
    
    results = [
        {
            "Dataset": dataset_name,
            "Model": model_name,
            "Method": "Raw",
            f"Target_Hit@{k}": raw_target["target_hit"],
            f"Target_Hit@{k}_Drop": 0.0,
            "Control_Utility": raw_control["utility"],
            "Control_Utility_Drop": 0.0,
            "Control_TE": raw_control["target_exposure"],
            "TE_Change": 0.0,
            "Avg_Gate_s": 0.0
        },
        {
            "Dataset": dataset_name,
            "Model": model_name,
            "Method": "C2U",
            f"Target_Hit@{k}": c2u_target["target_hit"],
            f"Target_Hit@{k}_Drop": target_hit_drop,
            "Control_Utility": c2u_control["utility"],
            "Control_Utility_Drop": control_utility_drop,
            "Control_TE": c2u_control["target_exposure"],
            "TE_Change": te_drop,
            "Avg_Gate_s": avg_gate_t
        }
    ]
    
    df = pd.DataFrame(results)
    
    # Ghi file CSV
    if os.path.exists(output_csv):
        df.to_csv(output_csv, mode="a", header=False, index=False)
    else:
        df.to_csv(output_csv, mode="w", header=True, index=False)
        
    print("\n" + "=" * 80)
    print(f"KẾT QUẢ ĐÁNH GIÁ C2U (Top-{k} Retrieval): {dataset_name.upper()} | {model_name}")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80)
    print(f"[*] Target Hit@{k} Drop (càng cao càng tốt)      : {target_hit_drop * 100:.2f}%")
    print(f"[*] Control Utility Drop (CN, mục tiêu <= 0)     : {control_utility_drop * 100:.2f}%")
    print(f"[*] Target Exposure Change (TE, mục tiêu <= 0)   : {te_drop * 100:.2f}%\n")


def main():
    parser = argparse.ArgumentParser(description="Đánh giá Raw vs C2U Top-5 Retrieval.")
    parser.add_argument("--cache_dir", type=str, default="cache_emb/tofu", help="Thư mục cache embeddings")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/tofu/acr_checkpoint.pt", help="Đường dẫn ACR checkpoint")
    parser.add_argument("--output_csv", type=str, default="results.csv", help="Đường dẫn file CSV kết quả")
    parser.add_argument("--dataset_name", type=str, default="tofu", help="Tên tập dữ liệu")
    parser.add_argument("--model_name", type=str, default="all-MiniLM-L6-v2", help="Tên mô hình Encoder")
    parser.add_argument("--k", type=int, default=5, help="Top-K")
    parser.add_argument("--device", type=str, default=None, help="Device: cuda or cpu")
    args = parser.parse_args()
    
    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    run_evaluation(
        cache_dir=args.cache_dir,
        checkpoint_path=args.checkpoint,
        output_csv=args.output_csv,
        dataset_name=args.dataset_name,
        model_name=args.model_name,
        k=args.k,
        device=device
    )


if __name__ == "__main__":
    main()
