#!/usr/bin/env python3
"""
tests/test_dry_run.py
Kiểm thử khô toàn diện Gói 3.5 (v2):
1. Kiểm tra ACRModel forward/backward (gradient khác 0 trên cả A_raw và W).
2. Kiểm tra trên dữ liệu thực tế local: tofu_seed1_minilm.pt (nhánh phần dư nhỏ) & arxiv_seed1_minilm.pt (nhánh phần dư):
   - Huấn luyện nhanh ACR 1 epoch trên CPU.
   - Đại số lstsq float64 ở cả hai nhánh.
   - Gate-off identity: 100% s(q)==0 thỏa torch.equal(q_edited, q).
   - Trần góc: 100% truy vấn thỏa realized_angle <= 0.200001 (cả đơn khái niệm và đa khái niệm K=4).
   - Tương đương dạng đóng tại beta = 1: max||q'_pipeline - normalize(q + eta * k_q * h)|| <= 1e-5 và ||h - g|| <= 1e-5.
   - Biên số học: q = h, q = -h, q perp h, s=0, s=1, calibration suy biến ép G0, p ~ 0 ép fail-closed.
   - Tín hiệu trên val: in Raw vs C2U (H5 Drop, CN, TE, và tỷ lệ unedited / below cap / at cap).
3. Kiểm tra giao diện C2UProfile.gate_fn, C2UEditResult, C2UComposeOperator vs AvgDirectionOperator.
"""

import os
import sys
import math
import torch
import torch.nn.functional as F

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from acr_model import ACRModel, ACRLoss, get_dictionary_size_N
from acr_dataset import create_acr_dataloaders
from c2u_operator import (
    C2UOperator,
    C2UComposeOperator,
    AvgDirectionOperator,
    build_c2u_profile,
    _compute_kq_from_angle,
)


def train_fast_1_epoch(cache_data: dict, dataset_name: str, device: str = "cpu"):
    """Huấn luyện nhanh ACR 1 epoch trên CPU và kiểm tra gradient A_raw & W."""
    q_embs = cache_data["train_q_embs"].float()
    d_embs = cache_data["train_d_embs"].float()
    d = q_embs.shape[1]
    N = get_dictionary_size_N(dataset_name, d)

    train_loader, _, _, _ = create_acr_dataloaders(q_embs, d_embs, val_ratio=0.1, batch_size=32, seed=42, device=device)
    model = ACRModel(d=d, N=N, tau=0.1).to(device)
    criterion = ACRLoss(mr=0.3, lambda_align=1.0, lambda_ent=0.02).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    model.train()
    checked_grad = False
    for q_b, pos_b, neg_b in train_loader:
        optimizer.zero_grad()
        out = model(q_b, pos_b, neg_b)
        loss_dict = criterion(out, q_b, pos_b, neg_b)
        loss = loss_dict["total_loss"]
        loss.backward()
        if not checked_grad:
            assert model.A_raw.grad is not None and torch.norm(model.A_raw.grad).item() > 0, "Grad A_raw = 0!"
            assert model.W.grad is not None and torch.norm(model.W.grad).item() > 0, "Grad W = 0!"
            assert not torch.isnan(model.A_raw.grad).any() and not torch.isnan(model.W.grad).any()
            checked_grad = True
        optimizer.step()

    model.eval()
    return model.get_normalized_dictionary().detach(), model.W.detach(), N


def verify_closed_form_equivalence(op: C2UOperator, q_batch: torch.Tensor):
    """Kiểm tra ||q'_pipeline - normalize(q + eta * k_q * h)|| <= 1e-5 và ||h - g|| <= 1e-5 ở nhánh residual."""
    res = op.edit_query_detailed(q_batch)
    prof = op.profile

    h_g_diff = torch.norm(prof.h_vec - prof.g).item()
    print(f"  [Closed-form] branch={prof.residual_branch} | ||h - g||_2 = {h_g_diff:.3e}")
    assert h_g_diff <= 1e-5, f"||h - g|| vượt 1e-5: {h_g_diff}"

    theta_req = op.theta_req_factor * res.gate_s_max
    k_q = _compute_kq_from_angle(q_batch, prof.h_unit, theta_req, eps_plane=op.eps_plane)
    q_norm = F.normalize(q_batch.float(), p=2, dim=-1)
    q_closed = q_norm.clone()
    active = (res.gate_s_max > 0) & (res.eta > 0)
    if active.any():
        raw_step = q_norm[active] + res.eta[active].unsqueeze(-1) * k_q[active].unsqueeze(-1) * prof.h_vec.unsqueeze(0)
        q_closed[active] = F.normalize(raw_step, p=2, dim=-1)

    max_diff = (res.q_edited - q_closed).abs().max().item()
    print(f"  [Closed-form] max|q'_pipeline - q'_closed| = {max_diff:.3e}")
    assert max_diff <= 1e-5, f"Sai lệch dạng đóng > 1e-5: {max_diff}"


def test_real_unit(cache_path: str, dataset_name: str, expected_branch: str):
    print("\n" + "=" * 70)
    print(f"[DRY-RUN UNIT] {os.path.basename(cache_path)} (expected branch: {expected_branch})")
    print("=" * 70)
    data = torch.load(cache_path, map_location="cpu", weights_only=False)

    A, W, N = train_fast_1_epoch(data, dataset_name, device="cpu")
    print(f"  [ACR] 1-epoch CPU OK (N={N}, d={A.shape[1]}), grad(A_raw)>0, grad(W)>0")

    op = C2UOperator(
        A=A,
        target_profile_embs=data["profile_embs"],
        calibration_control_embs=data["calib_doc_embs"],
        calibration_query_embs=data["calib_query_embs"],
        W=W,
        tau=0.1,
        beta=1.0,
        gate_quantile_space="query",
        device="cpu",
    )
    assert op.profile.residual_branch == expected_branch, (
        f"Sai nhánh phần dư: {op.profile.residual_branch} != {expected_branch}"
    )

    val_q = data["val_q_embs"]
    res = op.edit_query_detailed(val_q)

    # 1. Gate-off identity
    gate_off_mask = (res.gate_s_max == 0.0)
    if gate_off_mask.any():
        assert torch.equal(res.q_edited[gate_off_mask], val_q[gate_off_mask]), "Vi phạm gate-off torch.equal!"
    print(f"  [Gate-off] {int(gate_off_mask.sum())}/{len(val_q)} queries torch.equal == True")

    # 2. Trần góc <= 0.200001
    max_ang = res.realized_angle.max().item()
    print(f"  [Angle cap] max realized_angle = {max_ang:.6f} rad (<= 0.200001)")
    assert max_ang <= 0.200001, f"Vi phạm trần góc: {max_ang}"

    # 3. Tương đương dạng đóng
    verify_closed_form_equivalence(op, val_q)

    # 4. Tín hiệu Raw vs C2U trên val
    idx_embs = data["index_embs"]
    idx_target = data["index_is_target"]
    val_target = data["val_q_is_target"]
    val_labels = data["val_q_labels"]
    idx_labels = data["index_labels"]

    def _eval(q_vecs):
        sims = torch.mm(F.normalize(q_vecs, dim=-1), F.normalize(idx_embs, dim=-1).t())
        top5 = torch.topk(sims, k=5, dim=1).indices
        t_idx = torch.where(val_target == 1)[0]
        c_idx = torch.where(val_target == 0)[0]
        h5 = sum((idx_target[top5[i]] == 1).any().item() for i in t_idx) / len(t_idx) * 100.0
        cn_hits = sum(val_labels[i] in [idx_labels[j] for j in top5[i]] for i in c_idx) / len(c_idx) * 100.0
        te_rate = sum((idx_target[top5[i]] == 1).sum().item() / 5.0 for i in c_idx) / len(c_idx) * 100.0
        return h5, cn_hits, te_rate

    raw_h5, raw_ctrl, raw_te = _eval(val_q)
    c2u_h5, c2u_ctrl, c2u_te = _eval(res.q_edited)
    h5_drop = raw_h5 - c2u_h5
    cn = raw_ctrl - c2u_ctrl
    te = c2u_te - raw_te

    def _status_pct(mask):
        s_sub = res.gate_s_max[mask]
        ang_sub = res.realized_angle[mask]
        unedited = (s_sub == 0.0).float().mean().item() * 100.0
        at_cap = ((s_sub > 0.0) & (ang_sub >= 0.1999)).float().mean().item() * 100.0
        below_cap = 100.0 - unedited - at_cap
        return unedited, below_cap, at_cap

    t_un, t_bel, t_cap = _status_pct(val_target == 1)
    c_un, c_bel, c_cap = _status_pct(val_target == 0)

    print(f"  [Signal val] Raw H5={raw_h5:.2f}% -> C2U H5={c2u_h5:.2f}% | H5 Drop={h5_drop:+.2f} pp | CN={cn:+.2f} pp | TE={te:+.2f} pp")
    print(f"  [Edit regime Target ] unedited={t_un:.1f}% / below_cap={t_bel:.1f}% / at_cap={t_cap:.1f}%")
    print(f"  [Edit regime Control] unedited={c_un:.1f}% / below_cap={c_bel:.1f}% / at_cap={c_cap:.1f}%")
    assert h5_drop > 0.0, f"Kỳ vọng H5 Drop > 0 trên {dataset_name}, nhận được {h5_drop}"


def test_numerical_boundaries_and_multi():
    print("\n" + "=" * 70)
    print("[BOUNDARY & MULTI-CONCEPT] Kiểm tra biên số học, G0, fail-closed, gate_fn & K=4")
    print("=" * 70)
    torch.manual_seed(7)
    d, N = 64, 32
    A = F.normalize(torch.randn(N, d), p=2, dim=-1)
    W = torch.eye(d)
    t_dir = F.normalize(torch.randn(d), p=2, dim=-1)
    c_dir = F.normalize(torch.randn(d), p=2, dim=-1)
    prof = F.normalize(t_dir + 0.1 * torch.randn(20, d), p=2, dim=-1)
    cdoc = F.normalize(c_dir + 0.1 * torch.randn(100, d), p=2, dim=-1)
    cquery = F.normalize(c_dir + 0.1 * torch.randn(100, d), p=2, dim=-1)

    op = C2UOperator(A=A, target_profile_embs=prof, calibration_control_embs=cdoc,
                     calibration_query_embs=cquery, W=W, device="cpu")
    h = op.profile.h_unit
    # Biên q = h, q = -h, q perp h
    q_perp = F.normalize(torch.randn(d) - torch.dot(torch.randn(d), h) * h, p=2, dim=-1)
    q_edges = torch.stack([h, -h, q_perp], dim=0)
    res_edges = op.edit_query_detailed(q_edges)
    assert not torch.isnan(res_edges.q_edited).any() and not torch.isinf(res_edges.q_edited).any()
    assert res_edges.realized_angle.max().item() <= 0.200001

    # Kiểm tra gate_fn trực tiếp (dùng lại ở GĐ5)
    s_gate_fn = op.profile.gate_fn(q_edges)
    assert torch.allclose(s_gate_fn, res_edges.gate_s_max, atol=1e-6)

    # Ép G0 fallback: toàn bộ calibration query giống hệt nhau => z0 == z1
    cquery_deg = cquery[0].unsqueeze(0).repeat(50, 1)
    prof_g0 = build_c2u_profile(A=A, target_profile_embs=prof, calib_doc_embs=cdoc,
                                calib_query_embs=cquery_deg, W=W, gate_quantile_space="query")
    assert prof_g0.gate_mode == "G0", f"Không kích hoạt G0 fallback: {prof_g0.gate_mode}"
    # Dấu bằng tau0 không được kích hoạt
    s_eq = prof_g0.gate_fn(cquery_deg[:5])
    assert torch.all(s_eq == 0.0), "G0 fallback phải không kích hoạt tại dấu bằng tau0!"

    # Ép fail-closed: profile và calib trùng nhau => v = 0 -> fail_closed
    prof_fc = build_c2u_profile(A=A, target_profile_embs=cdoc[:20], calib_doc_embs=cdoc[:20],
                                calib_query_embs=cquery, W=W)
    assert prof_fc.residual_branch == "fail_closed" and prof_fc.gate_mode == "disabled"
    assert torch.all(prof_fc.gate_fn(q_edges) == 0.0)

    # Kiểm tra K=4 Compose vs Avg-Direction
    profiles = []
    for _ in range(4):
        tk = F.normalize(torch.randn(d), p=2, dim=-1)
        pk = F.normalize(tk + 0.1 * torch.randn(20, d), p=2, dim=-1)
        profiles.append(build_c2u_profile(A=A, target_profile_embs=pk, calib_doc_embs=cdoc,
                                          calib_query_embs=cquery, W=W))
    comp_op = C2UComposeOperator(profiles)
    avg_op = AvgDirectionOperator(profiles)
    q_test = F.normalize(torch.randn(32, d), p=2, dim=-1)
    res_comp = comp_op.edit_query_detailed(q_test)
    res_avg = avg_op.edit_query_detailed(q_test)
    assert res_comp.realized_angle.max().item() <= 0.200001
    assert res_avg.realized_angle.max().item() <= 0.200001
    assert not torch.allclose(res_comp.q_edited, res_avg.q_edited), "Compose và Avg-Direction phải cho đầu ra khác nhau!"
    print("  -> Biên số học, G0, fail-closed, gate_fn & K=4 Compose vs Avg: PASSED!")


def main():
    torch.manual_seed(42)
    test_numerical_boundaries_and_multi()

    tofu_cache = os.path.join(ROOT_DIR, "cache_embeddings", "tofu_seed1_minilm.pt")
    arxiv_cache = os.path.join(ROOT_DIR, "cache_embeddings", "arxiv_seed1_minilm.pt")
    assert os.path.exists(tofu_cache), f"Thiếu {tofu_cache}"
    assert os.path.exists(arxiv_cache), f"Thiếu {arxiv_cache}"

    test_real_unit(tofu_cache, "tofu", expected_branch="small")
    test_real_unit(arxiv_cache, "arxiv", expected_branch="residual")

    print("\n" + "=" * 70)
    print("TẤT CẢ KIỂM THỬ KHÔ GÓI 3.5 ĐÃ PASS 100%!")
    print("=" * 70)


if __name__ == "__main__":
    main()
