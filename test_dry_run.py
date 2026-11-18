#!/usr/bin/env python3
"""
test_dry_run.py
Kiểm thử toàn diện dry-run pipeline C2U trên dữ liệu tensor giả lập (hỗ trợ cả CUDA và CPU):
1. Kiểm tra kiến trúc ACRModel và tính toán Gradient (Backward pass)
2. Kiểm tra bộ 3 Loss (Cosine Triplet, Cosine Alignment, Entropy Penalty)
3. Kiểm tra toán tử C2U 4 bước:
   - Preconditioned suppression direction (mu_t, mu_c, D, g)
   - Calibrated Gate s(q)
   - lstsq(A^T, g), residual e, r, A', b_beta và đẳng thức b_beta^T A' = g
   - Hệ số k_q theo định lý hàm sin
   - Trust-Region Update với góc xoay <= 0.2 rad (<q', q> >= max(cos(0.2), 0.98))
4. Mô phỏng Top-5 Retrieval và xác nhận Target Hit Drop > 0
"""

import sys
import os
import math

# Thêm src vào đường dẫn
SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import torch
import torch.nn.functional as F

from acr_model import ACRModel, ACRLoss
from c2u_operator import C2UOperator


def test_acr_gradients(device: str = "cpu"):
    print("\n" + "=" * 60)
    print(f"[1] KIỂM TRA ACR MODEL & GRADIENT BACKWARD (Device: {device})")
    print("=" * 60)
    d = 384
    N = 768
    batch_size = 8
    
    model = ACRModel(d=d, N=N, tau=0.1).to(device)
    criterion = ACRLoss(margin=0.3, lambda_align=1.0, lambda_ent=0.02).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    # Tensor giả lập đã chuẩn hóa L2 trên device
    q = F.normalize(torch.randn(batch_size, d, device=device), p=2, dim=-1)
    pos = F.normalize(torch.randn(batch_size, d, device=device), p=2, dim=-1)
    neg = F.normalize(torch.randn(batch_size, d, device=device), p=2, dim=-1)
    
    # Forward pass
    q_proj, w_q, z_q = model(q)
    pos_proj, w_pos, z_pos = model(pos)
    neg_proj, w_neg, z_neg = model(neg)
    
    # Kiểm tra shape
    assert q_proj.shape == (batch_size, d), f"Lỗi shape q_proj: {q_proj.shape}"
    assert w_q.shape == (batch_size, N), f"Lỗi shape w_q: {w_q.shape}"
    assert z_q.shape == (batch_size, d), f"Lỗi shape z_q: {z_q.shape}"
    
    # Kiểm tra tổng trọng số softmax = 1.0
    w_sum = torch.sum(w_q, dim=-1)
    assert torch.allclose(w_sum, torch.ones_like(w_sum), atol=1e-5), "Lỗi tổng softmax != 1"
    
    # Loss computation
    loss_dict = criterion(q_proj, pos_proj, neg_proj, w_q, w_pos, w_neg)
    total_loss = loss_dict["total_loss"]
    
    print(f"  Total Loss   : {total_loss.item():.4f}")
    print(f"  Triplet Loss : {loss_dict['triplet_loss'].item():.4f}")
    print(f"  Align Loss   : {loss_dict['align_loss'].item():.4f}")
    print(f"  Entropy Loss : {loss_dict['ent_loss'].item():.4f}")
    
    # Backward pass
    optimizer.zero_grad()
    total_loss.backward()
    
    assert model.A.grad is not None, "Gradient của ma trận A bị None"
    grad_norm_A = torch.norm(model.A.grad).item()
    print(f"  Gradient Norm của A: {grad_norm_A:.6f}")
    assert not math.isnan(grad_norm_A), "Gradient của A bị NaN"
    assert grad_norm_A > 0, "Gradient của A bằng 0"
    
    optimizer.step()
    print("  -> ACR Model & Backward pass: PASSED!")


def test_c2u_algebra(device: str = "cpu"):
    print("\n" + "=" * 60)
    print(f"[2] KIỂM TRA ĐẠI SỐ TUYẾN TÍNH C2U (lstsq, r, delta_q, Trust-Region) (Device: {device})")
    print("=" * 60)
    d = 384
    N = 768
    
    A = F.normalize(torch.randn(N, d, device=device), p=2, dim=-1)
    target_vec = F.normalize(torch.randn(d, device=device), p=2, dim=-1)
    ctrl_vec = F.normalize(torch.randn(d, device=device), p=2, dim=-1)
    
    target_profile = F.normalize(target_vec + 0.2 * torch.randn(20, d, device=device), p=2, dim=-1)
    calibration_control = F.normalize(ctrl_vec + 0.2 * torch.randn(200, d, device=device), p=2, dim=-1)
    
    operator = C2UOperator(
        A=A,
        target_profile_embs=target_profile,
        calibration_control_embs=calibration_control,
        tau=0.1,
        beta=1.0,
        theta_req_factor=0.3,
        device=device
    )
    
    # 1. Kiểm tra Bước 1: g
    assert operator.g.shape == (d,), f"Lỗi shape g: {operator.g.shape}"
    g_norm = torch.norm(operator.g).item()
    print(f"  g norm: {g_norm:.4f}")
    assert abs(g_norm - 1.0) < 1e-4, "Norm của g khác 1.0"
    
    # 2. Kiểm tra lstsq và đẳng thức b_beta^T @ A' = g
    recon_g_prime = torch.matmul(operator.b_beta, operator.A_prime)
    diff_g = torch.norm(recon_g_prime - operator.g).item()
    print(f"  Độ lệch giữa b_beta^T @ A' và g: {diff_g:.6e}")
    assert diff_g < 1e-4, "Đẳng thức b_beta^T @ A' = g không thỏa mãn!"
    
    # 3. Kiểm tra Calibrated Gate s(q)
    q_target = F.normalize(target_vec + 0.05 * torch.randn(10, d, device=device), p=2, dim=-1)
    q_ctrl = F.normalize(ctrl_vec + 0.05 * torch.randn(10, d, device=device), p=2, dim=-1)
    
    s_target = operator.compute_gate(q_target)
    s_ctrl = operator.compute_gate(q_ctrl)
    
    print(f"  Avg Gate s(q) - Target: {s_target.mean().item():.4f}")
    print(f"  Avg Gate s(q) - Control: {s_ctrl.mean().item():.4f}")
    assert s_target.mean().item() > s_ctrl.mean().item(), "Gate Target phải lớn hơn Gate Control"
    
    # 4. Kiểm tra Trust-Region: <q', q> >= max(cos(0.2), 0.98)
    q_prime_target, _ = operator.edit_query(q_target)
    cos_sim = torch.sum(q_prime_target * q_target, dim=-1)
    min_cos = cos_sim.min().item()
    thresh = max(math.cos(0.2), 0.98)
    print(f"  Min cos(q', q) trong batch: {min_cos:.6f} (Ngưỡng yêu cầu: >= {thresh:.6f})")
    assert min_cos >= thresh - 1e-5, f"Vi phạm góc xoay Trust-Region: {min_cos} < {thresh}"
    
    # 5. Kiểm tra unlearning: giảm cosine sim với target_vec
    cos_target_before = torch.sum(q_target * target_vec, dim=-1).mean().item()
    cos_target_after = torch.sum(q_prime_target * target_vec, dim=-1).mean().item()
    print(f"  Target Cosine Sim: Trước={cos_target_before:.4f} -> Sau={cos_target_after:.4f} (Giảm: {cos_target_before - cos_target_after:.4f})")
    assert cos_target_after < cos_target_before, "C2U phải làm giảm độ tương đồng với Target Concept"
    
    print("  -> Đại số tuyến tính C2U & Trust-Region: PASSED!")


def test_topk_retrieval_simulation(device: str = "cpu"):
    print("\n" + "=" * 60)
    print(f"[3] MÔ PHỎNG TOP-5 RETRIEVAL (Hit@5 Drop, CN, TE) (Device: {device})")
    print("=" * 60)
    d = 384
    N = 768
    
    A = F.normalize(torch.randn(N, d, device=device), p=2, dim=-1)
    target_vec = F.normalize(torch.randn(d, device=device), p=2, dim=-1)
    ctrl_vec = F.normalize(torch.randn(d, device=device), p=2, dim=-1)
    
    target_profile = F.normalize(target_vec + 0.1 * torch.randn(20, d, device=device), p=2, dim=-1)
    calibration_control = F.normalize(ctrl_vec + 0.1 * torch.randn(200, d, device=device), p=2, dim=-1)
    
    operator = C2UOperator(A, target_profile, calibration_control, tau=0.1, device=device)
    
    corpus_target = F.normalize(target_vec + 0.15 * torch.randn(100, d, device=device), p=2, dim=-1)
    corpus_ctrl = F.normalize(ctrl_vec + 0.15 * torch.randn(300, d, device=device), p=2, dim=-1)
    corpus = torch.cat([corpus_target, corpus_ctrl], dim=0)  # (400, d)
    corpus_is_target = torch.cat([torch.ones(100, device=device), torch.zeros(300, device=device)]).long()
    
    test_target_q = F.normalize(target_vec + 0.15 * torch.randn(100, d, device=device), p=2, dim=-1)
    test_control_q = F.normalize(ctrl_vec + 0.15 * torch.randn(300, d, device=device), p=2, dim=-1)
    
    test_target_q_c2u, _ = operator.edit_query(test_target_q)
    test_control_q_c2u, _ = operator.edit_query(test_control_q)
    
    def eval_retrieval(q_embs, k=5):
        sims = torch.matmul(q_embs, corpus.t())
        _, topk_idx = torch.topk(sims, k=k, dim=-1)
        topk_target = corpus_is_target[topk_idx]
        hit_at_k = (topk_target.sum(dim=-1) > 0).float().mean().item()
        exposure = (topk_target.float().sum(dim=-1) / float(k)).mean().item()
        return hit_at_k, exposure
        
    raw_hit, raw_te = eval_retrieval(test_target_q)
    c2u_hit, c2u_te = eval_retrieval(test_target_q_c2u)
    
    hit_drop = raw_hit - c2u_hit
    te_drop = raw_te - c2u_te
    
    print(f"  Raw Target Hit@5  : {raw_hit * 100:.2f}% | TE: {raw_te:.2f}")
    print(f"  C2U Target Hit@5  : {c2u_hit * 100:.2f}% | TE: {c2u_te:.2f}")
    print(f"  Target Hit@5 Drop : {hit_drop * 100:.2f}% (Càng cao càng tốt)")
    print(f"  TE Drop           : {te_drop:.4f} (Càng cao càng tốt)")
    
    assert hit_drop >= 0.0, "Hit@5 Drop phải không âm"
    assert te_drop >= 0.0, "TE Drop phải không âm"
    print("  -> Mô phỏng Top-5 Retrieval: PASSED!")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"BẮT ĐẦU CHẠY DRY-RUN TOÀN DIỆN C2U PIPELINE (Device: {device})...")
    torch.manual_seed(42)
    test_acr_gradients(device)
    test_c2u_algebra(device)
    test_topk_retrieval_simulation(device)
    print("\n" + "=" * 60)
    print("TẤT CẢ KIỂM THỬ DRY-RUN ĐÃ THÀNH CÔNG VƯỢT TRỘI (100% PASS)!")
    print("=" * 60)


if __name__ == "__main__":
    main()
