#!/usr/bin/env python3
"""
src/c2u_operator.py
Toán tử hiệu chỉnh truy vấn trực tuyến C2U (v2 - Giai đoạn 3):
- Khởi tạo hồ sơ khái niệm (C2UProfile) trên float64:
  * Hướng triệt tiêu tiền điều kiện g = normalize(-D^{-1} v)
  * Phân vị cổng (z0, z1) theo gate_quantile_space in {"query", "doc"}
  * G1 ramp và G0 fallback (thống kê thứ tự ceil(0.98 * n), kích hoạt nghiêm ngặt > tau0)
  * lstsq(A^T, g) trên float64, đủ 2 nhánh phần dư (||e|| > 1e-6 vs <= 1e-6) và fail-closed
- Suy luận trực tuyến:
  * Gate-off sao chép bit-exact từ mảng float32 gốc
  * eps_plane = 1e-6 rad
  * Ép kiểu FP32 trước khi trừ (w'_q - [w_q; 0])
  * Vùng tin cậy bằng bisection 30 vòng trên float64, giữ cận dưới khả thi
- Tổ hợp đa khái niệm (C2UComposeOperator) và bộ so sánh (AvgDirectionOperator).
"""

from dataclasses import dataclass
import math
from typing import List, Optional, Tuple
import torch
import torch.nn.functional as F


TRUST_COS_THRESH = max(math.cos(float(torch.tensor(0.2, dtype=torch.float32).item())), 0.98)


@dataclass
class C2UProfile:
    """Hồ sơ khái niệm tính một lần cho mỗi (unit, khái niệm)."""
    g: torch.Tensor                  # [d] float32, hướng triệt tiêu chuẩn hóa
    mu_t: torch.Tensor               # [d] float32, tâm tài liệu profile
    mu_c: torch.Tensor               # [d] float32, tâm tài liệu calibration control
    mu_t_norm: torch.Tensor          # [d] float32, hướng đơn vị của mu_t
    mu_c_norm: torch.Tensor          # [d] float32, hướng đơn vị của mu_c
    z0: float
    z1: float
    tau0: float
    gate_mode: str                   # "G1" | "G0" | "disabled"
    gate_quantile_space: str         # "query" | "doc"
    A_prime: torch.Tensor            # [N+1, d] float32
    b_beta: torch.Tensor             # [N+1] float32
    h_vec: torch.Tensor              # [d] float32, A_prime^T @ b_beta
    h_unit: torch.Tensor             # [d] float32, h_vec / ||h_vec||
    residual_branch: str             # "residual" | "small" | "fail_closed"
    A_base: torch.Tensor             # [N, d] float32, dùng tính w_q qua head W
    W: Optional[torch.Tensor] = None # [d, d] float32, projection head đóng băng
    tau: float = 0.1

    def compute_z(self, X: torch.Tensor) -> torch.Tensor:
        """Tính điểm chiếu thô z(x) = x^T normalize(mu_t) - x^T normalize(mu_c)."""
        x_norm = F.normalize(X.to(dtype=torch.float32, device=self.mu_t_norm.device), p=2, dim=-1)
        return torch.matmul(x_norm, self.mu_t_norm) - torch.matmul(x_norm, self.mu_c_norm)

    def gate_fn(self, X: torch.Tensor) -> torch.Tensor:
        """Tính cổng kích hoạt s(x) in [0, 1] cho bất kỳ batch vector nào (tái sử dụng ở GĐ5)."""
        z = self.compute_z(X)
        if self.gate_mode == "disabled" or self.residual_branch == "fail_closed":
            return torch.zeros_like(z, dtype=torch.float32)
        if self.gate_mode == "G0":
            return (z > self.tau0).to(dtype=torch.float32)
        denom = self.z1 - self.z0
        if not math.isfinite(denom) or denom <= 0.0:
            return torch.zeros_like(z, dtype=torch.float32)
        return torch.clamp((z - self.z0) / denom, 0.0, 1.0).to(dtype=torch.float32)


@dataclass
class C2UEditResult:
    """Kết quả chỉnh sửa truy vấn trực tuyến với đầy đủ tín hiệu điều phối cho GĐ4–GĐ6."""
    q_edited: torch.Tensor           # [B, d] float32, đã chuẩn hóa L2
    gate_s: torch.Tensor             # [B, K] float32
    gate_s_max: torch.Tensor         # [B] float32
    z_score: torch.Tensor            # [B, K] float32
    direction_g: torch.Tensor        # [K, d] float32
    mu_t: torch.Tensor               # [K, d] float32
    realized_angle: torch.Tensor     # [B] float32, arccos(<q', q>)
    eta: torch.Tensor                # [B] float32
    branch: List[str]                # "gate_off" | "zero_delta" | "full_step" | "bisect"


def build_c2u_profile(
    A: torch.Tensor,
    target_profile_embs: torch.Tensor,
    calib_doc_embs: torch.Tensor,
    calib_query_embs: Optional[torch.Tensor] = None,
    W: Optional[torch.Tensor] = None,
    tau: float = 0.1,
    beta: float = 1.0,
    gate_quantile_space: str = "query",
    device: str = "cpu",
) -> C2UProfile:
    """Khởi tạo offline C2UProfile trên float64 theo Gói 3.1."""
    if gate_quantile_space not in ("query", "doc"):
        raise ValueError(f"gate_quantile_space phải là 'query' hoặc 'doc', nhận được: {gate_quantile_space}")

    A_f32 = F.normalize(A.to(device=device, dtype=torch.float32), p=2, dim=-1)
    N, d = A_f32.shape
    W_f32 = W.to(device=device, dtype=torch.float32) if W is not None else None

    prof64 = F.normalize(target_profile_embs.to(device=device, dtype=torch.float64), p=2, dim=-1)
    cdoc64 = F.normalize(calib_doc_embs.to(device=device, dtype=torch.float64), p=2, dim=-1)
    if gate_quantile_space == "query" and calib_query_embs is not None:
        cgate64 = F.normalize(calib_query_embs.to(device=device, dtype=torch.float64), p=2, dim=-1)
    else:
        cgate64 = cdoc64

    mu_t64 = prof64.mean(dim=0)
    mu_c64 = cdoc64.mean(dim=0)
    mu_t_norm_val = torch.linalg.norm(mu_t64).item()
    mu_c_norm_val = torch.linalg.norm(mu_c64).item()

    def _disabled_profile(reason: str) -> C2UProfile:
        zeros_d = torch.zeros(d, device=device, dtype=torch.float32)
        return C2UProfile(
            g=zeros_d,
            mu_t=mu_t64.float(),
            mu_c=mu_c64.float(),
            mu_t_norm=zeros_d,
            mu_c_norm=zeros_d,
            z0=0.0,
            z1=0.0,
            tau0=float("inf"),
            gate_mode="disabled",
            gate_quantile_space=gate_quantile_space,
            A_prime=torch.cat([A_f32, zeros_d.unsqueeze(0)], dim=0),
            b_beta=torch.zeros(N + 1, device=device, dtype=torch.float32),
            h_vec=zeros_d,
            h_unit=zeros_d,
            residual_branch=reason,
            A_base=A_f32,
            W=W_f32,
            tau=tau,
        )

    if (not math.isfinite(mu_t_norm_val) or mu_t_norm_val <= 1e-12
            or not math.isfinite(mu_c_norm_val) or mu_c_norm_val <= 1e-12):
        return _disabled_profile("fail_closed")

    mu_t_unit64 = mu_t64 / mu_t_norm_val
    mu_c_unit64 = mu_c64 / mu_c_norm_val

    # 1. Hướng triệt tiêu tiền điều kiện g = normalize(-D^{-1} v)
    v_diff = mu_t64 - mu_c64
    v_norm = torch.linalg.norm(v_diff).item()
    if not math.isfinite(v_norm) or v_norm <= 1e-12:
        return _disabled_profile("fail_closed")
    v64 = v_diff / v_norm

    m = (cdoc64 ** 2).mean(dim=0)
    m_bar = m.mean()
    D_diag = 0.9 * m + 0.1 * m_bar + 0.001
    if not torch.isfinite(D_diag).all() or (D_diag <= 0).any():
        return _disabled_profile("fail_closed")

    g_raw = -(v64 / D_diag)
    g_raw_norm = torch.linalg.norm(g_raw).item()
    if not math.isfinite(g_raw_norm) or g_raw_norm <= 1e-12:
        return _disabled_profile("fail_closed")
    g64 = g_raw / g_raw_norm
    if not torch.isfinite(g64).all():
        return _disabled_profile("fail_closed")

    # 2. Ngưỡng cổng (G1 ramp hoặc G0 fallback) — dùng cùng phép tính FP32 với compute_z()
    mu_t_unit32 = mu_t_unit64.float()
    mu_c_unit32 = mu_c_unit64.float()
    cgate32 = F.normalize(cgate64.float(), p=2, dim=-1)
    z_calib = torch.matmul(cgate32, mu_t_unit32) - torch.matmul(cgate32, mu_c_unit32)
    if not torch.isfinite(z_calib).all() or z_calib.numel() == 0:
        return _disabled_profile("fail_closed")

    z0 = float(torch.quantile(z_calib, 0.50).item())
    z1 = float(torch.quantile(z_calib, 0.95).item())
    tau0 = float("inf")
    if math.isfinite(z0) and math.isfinite(z1) and z1 > z0:
        gate_mode = "G1"
    else:
        gate_mode = "G0"
        z_sorted, _ = torch.sort(z_calib)
        n = z_sorted.numel()
        order_idx = min(n - 1, max(0, math.ceil(0.98 * n) - 1))
        tau0 = float(z_sorted[order_idx].item())

    # 3. Bổ sung nguyên tử phần dư trên float64
    A64 = F.normalize(A_f32.to(torch.float64), p=2, dim=-1)
    At64 = A64.t()
    # Giải trên CPU/device tùy lstsq (float64 lstsq ổn định trên cả CPU và CUDA)
    lstsq_res = torch.linalg.lstsq(At64, g64.unsqueeze(-1))
    a64 = lstsq_res.solution.squeeze(-1)
    if not torch.isfinite(a64).all():
        return _disabled_profile("fail_closed")

    p64 = torch.matmul(At64, a64)
    e64 = g64 - p64
    e_norm = torch.linalg.norm(e64).item()
    p_norm = torch.linalg.norm(p64).item()

    if not math.isfinite(e_norm) or not math.isfinite(p_norm):
        return _disabled_profile("fail_closed")

    if e_norm > 1e-6:
        residual_branch = "residual"
        r64 = e64 / e_norm
        b_tail = float(beta) * e_norm
    else:
        if p_norm <= 1e-12:
            return _disabled_profile("fail_closed")
        residual_branch = "small"
        r64 = p64 / p_norm
        b_tail = 0.0

    A_prime64 = torch.cat([A64, r64.unsqueeze(0)], dim=0)
    b_beta64 = torch.cat([a64, torch.tensor([b_tail], device=device, dtype=torch.float64)], dim=0)
    h64 = torch.matmul(A_prime64.t(), b_beta64)
    h_norm = torch.linalg.norm(h64).item()
    if not math.isfinite(h_norm) or h_norm <= 1e-12:
        return _disabled_profile("fail_closed")
    h_unit64 = h64 / h_norm

    return C2UProfile(
        g=g64.float(),
        mu_t=mu_t64.float(),
        mu_c=mu_c64.float(),
        mu_t_norm=mu_t_unit64.float(),
        mu_c_norm=mu_c_unit64.float(),
        z0=z0,
        z1=z1,
        tau0=tau0,
        gate_mode=gate_mode,
        gate_quantile_space=gate_quantile_space,
        A_prime=A_prime64.float(),
        b_beta=b_beta64.float(),
        h_vec=h64.float(),
        h_unit=h_unit64.float(),
        residual_branch=residual_branch,
        A_base=A_f32,
        W=W_f32,
        tau=tau,
    )


def _compute_wq0(q_f32: torch.Tensor, profile: C2UProfile) -> torch.Tensor:
    """Tính trọng số [w_q; 0] từ head song tuyến đóng băng ở FP32."""
    B = q_f32.shape[0]
    N = profile.A_base.shape[0]
    zeros_tail = torch.zeros(B, 1, device=q_f32.device, dtype=torch.float32)
    if profile.W is None:
        return torch.zeros(B, N + 1, device=q_f32.device, dtype=torch.float32)
    q_norm = F.normalize(q_f32, p=2, dim=-1)
    h_proj = torch.matmul(q_norm, profile.W)
    sim = torch.matmul(h_proj, profile.A_base.t())
    w_q = F.softmax(sim / profile.tau, dim=-1)
    return torch.cat([w_q, zeros_tail], dim=-1)


def _compute_kq_from_angle(
    q_f32: torch.Tensor,
    h_unit_f32: torch.Tensor,
    theta_req: torch.Tensor,
    eps_plane: float = 1e-6,
) -> torch.Tensor:
    """Tính hệ số k_q = sin(theta) / sin(phi - theta) với tính toán góc trên float64."""
    q64 = F.normalize(q_f32.to(torch.float64), p=2, dim=-1)
    h64 = F.normalize(h_unit_f32.to(torch.float64), p=2, dim=-1)
    cos_phi = torch.clamp(torch.sum(q64 * h64, dim=-1), -1.0, 1.0)
    phi = torch.acos(cos_phi)

    req64 = theta_req.to(torch.float64)
    theta = torch.clamp(torch.minimum(req64, phi - eps_plane), min=0.0)
    sin_theta = torch.sin(theta)
    sin_diff = torch.sin(phi - theta)

    valid = (req64 > 0.0) & (theta > 0.0) & (sin_diff > 1e-12) & torch.isfinite(sin_diff)
    safe_diff = torch.where(valid, sin_diff, torch.ones_like(sin_diff))
    k_q = torch.where(valid, sin_theta / safe_diff, torch.zeros_like(sin_theta))
    return k_q.to(torch.float32)


def _trust_region_bisection(
    q_orig_f32: torch.Tensor,
    delta_f32: torch.Tensor,
    is_gate_off: torch.Tensor,
    trust_cos_thresh: float = TRUST_COS_THRESH,
    n_bisect: int = 30,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
    """
    Bước vùng tin cậy trên float64:
    - Gate-off hoặc delta=0: sao chép trực tiếp từ q_orig_f32 (bit-exact torch.equal).
    - Thử eta = 1.0; nếu vi phạm <q', q> >= trust_cos_thresh thì bisection 30 vòng và giữ cận dưới.
    """
    B = q_orig_f32.shape[0]
    q_out = q_orig_f32.clone()
    etas = torch.zeros(B, device=q_orig_f32.device, dtype=torch.float32)
    angles = torch.zeros(B, device=q_orig_f32.device, dtype=torch.float32)
    branches: List[str] = ["gate_off"] * B

    q64_all = q_orig_f32.to(torch.float64)
    q64_norms = torch.linalg.norm(q64_all, dim=-1, keepdim=True)
    q64_unit_all = torch.where(q64_norms > 1e-12, q64_all / q64_norms, q64_all)
    d64_all = delta_f32.to(torch.float64)

    for i in range(B):
        if bool(is_gate_off[i].item()):
            branches[i] = "gate_off"
            continue
        d_vec = d64_all[i]
        if torch.all(d_vec == 0.0) or not torch.isfinite(d_vec).all():
            branches[i] = "zero_delta"
            continue

        q_u = q64_unit_all[i]

        def _try_eta(eta_val: float) -> Tuple[bool, torch.Tensor]:
            cand = q_u + eta_val * d_vec
            c_norm = torch.linalg.norm(cand).item()
            if not math.isfinite(c_norm) or c_norm <= 1e-12:
                return False, q_u
            cand_u = cand / c_norm
            if not torch.isfinite(cand_u).all():
                return False, q_u
            cos_val = float(torch.sum(cand_u * q_u).item())
            return cos_val >= trust_cos_thresh, cand_u

        ok_full, cand_full = _try_eta(1.0)
        if ok_full:
            q_out[i] = cand_full.float()
            etas[i] = 1.0
            cos_val = max(-1.0, min(1.0, float(torch.sum(cand_full.float().to(torch.float64) * q_u).item())))
            angles[i] = float(math.acos(cos_val))
            branches[i] = "full_step"
            continue

        lo = 0.0
        hi = 1.0
        best_u = q_u
        for _ in range(n_bisect):
            mid = 0.5 * (lo + hi)
            ok_mid, cand_mid = _try_eta(mid)
            if ok_mid:
                lo = mid
                best_u = cand_mid
            else:
                hi = mid

        if lo == 0.0:
            # Không có bước dương khả thi -> giữ nguyên q gốc
            branches[i] = "zero_delta"
            etas[i] = 0.0
            angles[i] = 0.0
        else:
            q_out[i] = best_u.float()
            etas[i] = float(lo)
            cos_val = max(-1.0, min(1.0, float(torch.sum(best_u.float().to(torch.float64) * q_u).item())))
            angles[i] = float(math.acos(cos_val))
            branches[i] = "bisect"

    return q_out, etas, angles, branches


class C2UOperator:
    """Toán tử chỉnh sửa truy vấn trực tuyến C2U đơn khái niệm (v2)."""

    def __init__(
        self,
        A: torch.Tensor,
        target_profile_embs: torch.Tensor,
        calibration_control_embs: torch.Tensor,
        calibration_query_embs: Optional[torch.Tensor] = None,
        W: Optional[torch.Tensor] = None,
        tau: float = 0.1,
        beta: float = 1.0,
        theta_req_factor: float = 0.3,
        eps_plane: float = 1e-6,
        trust_cos_thresh: float = TRUST_COS_THRESH,
        gate_quantile_space: str = "query",
        device: str = "cpu",
    ):
        self.device = device
        self.tau = tau
        self.beta = beta
        self.theta_req_factor = theta_req_factor
        self.eps_plane = eps_plane
        self.trust_cos_thresh = trust_cos_thresh

        self.profile = build_c2u_profile(
            A=A,
            target_profile_embs=target_profile_embs,
            calib_doc_embs=calibration_control_embs,
            calib_query_embs=calibration_query_embs,
            W=W,
            tau=tau,
            beta=beta,
            gate_quantile_space=gate_quantile_space,
            device=device,
        )

    def edit_query_detailed(self, q: torch.Tensor) -> C2UEditResult:
        """Chạy đầy đủ quy trình C2U và trả về C2UEditResult."""
        q_in = q.to(device=self.device, dtype=torch.float32)
        if q_in.dim() == 1:
            q_in = q_in.unsqueeze(0)

        z = self.profile.compute_z(q_in)
        s = self.profile.gate_fn(q_in)

        if self.profile.gate_mode == "disabled" or self.profile.residual_branch == "fail_closed":
            delta_q = torch.zeros_like(q_in)
            is_gate_off = torch.ones(q_in.shape[0], device=self.device, dtype=torch.bool)
        else:
            theta_req = self.theta_req_factor * s
            k_q = _compute_kq_from_angle(q_in, self.profile.h_unit, theta_req, eps_plane=self.eps_plane)
            w_q0 = _compute_wq0(q_in, self.profile)
            w_prime = w_q0 + k_q.unsqueeze(-1) * self.profile.b_beta.unsqueeze(0)
            # Ép về FP32 trước khi trừ (A.6)
            delta_q = torch.matmul(w_prime.float() - w_q0.float(), self.profile.A_prime.float())
            is_gate_off = (s <= 0.0)

        q_edited, etas, angles, branches = _trust_region_bisection(
            q_orig_f32=q_in,
            delta_f32=delta_q,
            is_gate_off=is_gate_off,
            trust_cos_thresh=self.trust_cos_thresh,
            n_bisect=30,
        )

        return C2UEditResult(
            q_edited=q_edited,
            gate_s=s.unsqueeze(-1),
            gate_s_max=s,
            z_score=z.unsqueeze(-1),
            direction_g=self.profile.g.unsqueeze(0),
            mu_t=self.profile.mu_t.unsqueeze(0),
            realized_angle=angles,
            eta=etas,
            branch=branches,
        )

    def edit_query(self, q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Giao diện tương thích trả về (q_edited, s)."""
        squeeze_out = (q.dim() == 1)
        res = self.edit_query_detailed(q)
        if squeeze_out:
            return res.q_edited.squeeze(0), res.gate_s_max.squeeze(0)
        return res.q_edited, res.gate_s_max


class C2UComposeOperator:
    """Tổ hợp đa khái niệm C2U-Compose theo Mục 4.5 (Gói 3.3)."""

    def __init__(
        self,
        profiles: List[C2UProfile],
        theta_req_factor: float = 0.3,
        budget_cap: float = 0.2,
        eps_plane: float = 1e-6,
        trust_cos_thresh: float = TRUST_COS_THRESH,
    ):
        assert len(profiles) > 0, "Cần ít nhất 1 C2UProfile"
        self.profiles = profiles
        self.theta_req_factor = theta_req_factor
        self.budget_cap = budget_cap
        self.eps_plane = eps_plane
        self.trust_cos_thresh = trust_cos_thresh
        self.device = profiles[0].g.device

    def edit_query_detailed(self, q: torch.Tensor) -> C2UEditResult:
        q_in = q.to(device=self.device, dtype=torch.float32)
        if q_in.dim() == 1:
            q_in = q_in.unsqueeze(0)
        B = q_in.shape[0]
        K = len(self.profiles)

        z_list = [p.compute_z(q_in) for p in self.profiles]
        s_list = [p.gate_fn(q_in) for p in self.profiles]
        z_mat = torch.stack(z_list, dim=-1)  # [B, K]
        s_mat = torch.stack(s_list, dim=-1)  # [B, K]
        s_max = s_mat.max(dim=-1).values     # [B]

        # Ngân sách góc chung B_q = min(0.3 * max_c s_c(q), 0.2)
        t_raw = self.theta_req_factor * s_mat  # [B, K]
        t_sum = t_raw.sum(dim=-1, keepdim=True)
        B_q = torch.minimum(self.theta_req_factor * s_max, torch.full_like(s_max, self.budget_cap)).unsqueeze(-1)
        scale = torch.where(t_sum > B_q, B_q / torch.clamp(t_sum, min=1e-12), torch.ones_like(t_sum))
        t_scaled = t_raw * scale

        delta_total = torch.zeros_like(q_in)
        for c_idx, prof in enumerate(self.profiles):
            if prof.gate_mode == "disabled" or prof.residual_branch == "fail_closed":
                continue
            t_c = t_scaled[:, c_idx]
            k_qc = _compute_kq_from_angle(q_in, prof.h_unit, t_c, eps_plane=self.eps_plane)
            w_q0 = _compute_wq0(q_in, prof)
            w_prime = w_q0 + k_qc.unsqueeze(-1) * prof.b_beta.unsqueeze(0)
            delta_c = torch.matmul(w_prime.float() - w_q0.float(), prof.A_prime.float())
            delta_total = delta_total + delta_c

        is_gate_off = (s_max <= 0.0)
        q_edited, etas, angles, branches = _trust_region_bisection(
            q_orig_f32=q_in,
            delta_f32=delta_total,
            is_gate_off=is_gate_off,
            trust_cos_thresh=self.trust_cos_thresh,
            n_bisect=30,
        )

        return C2UEditResult(
            q_edited=q_edited,
            gate_s=s_mat,
            gate_s_max=s_max,
            z_score=z_mat,
            direction_g=torch.stack([p.g for p in self.profiles], dim=0),
            mu_t=torch.stack([p.mu_t for p in self.profiles], dim=0),
            realized_angle=angles,
            eta=etas,
            branch=branches,
        )


class AvgDirectionOperator:
    """Bộ so sánh Avg-Direction cho đa khái niệm (Gói 3.3)."""

    def __init__(
        self,
        profiles: List[C2UProfile],
        theta_req_factor: float = 0.3,
        eps_plane: float = 1e-6,
        trust_cos_thresh: float = TRUST_COS_THRESH,
    ):
        assert len(profiles) > 0, "Cần ít nhất 1 C2UProfile"
        self.profiles = profiles
        self.theta_req_factor = theta_req_factor
        self.eps_plane = eps_plane
        self.trust_cos_thresh = trust_cos_thresh
        self.device = profiles[0].g.device

        active_g = [p.g.to(torch.float64) for p in profiles if p.residual_branch != "fail_closed"]
        if active_g:
            g_sum = torch.stack(active_g, dim=0).sum(dim=0)
            gn = torch.linalg.norm(g_sum).item()
            self.g_avg = (g_sum / gn).float() if (math.isfinite(gn) and gn > 1e-12) else torch.zeros_like(profiles[0].g)
        else:
            self.g_avg = torch.zeros_like(profiles[0].g)

    def edit_query_detailed(self, q: torch.Tensor) -> C2UEditResult:
        q_in = q.to(device=self.device, dtype=torch.float32)
        if q_in.dim() == 1:
            q_in = q_in.unsqueeze(0)

        z_mat = torch.stack([p.compute_z(q_in) for p in self.profiles], dim=-1)
        s_mat = torch.stack([p.gate_fn(q_in) for p in self.profiles], dim=-1)
        s_max = s_mat.max(dim=-1).values

        if float(torch.linalg.norm(self.g_avg).item()) <= 1e-12:
            delta_q = torch.zeros_like(q_in)
            is_gate_off = torch.ones(q_in.shape[0], device=self.device, dtype=torch.bool)
        else:
            theta_req = self.theta_req_factor * s_max
            k_q = _compute_kq_from_angle(q_in, self.g_avg, theta_req, eps_plane=self.eps_plane)
            delta_q = k_q.unsqueeze(-1) * self.g_avg.unsqueeze(0)
            is_gate_off = (s_max <= 0.0)

        q_edited, etas, angles, branches = _trust_region_bisection(
            q_orig_f32=q_in,
            delta_f32=delta_q,
            is_gate_off=is_gate_off,
            trust_cos_thresh=self.trust_cos_thresh,
            n_bisect=30,
        )

        return C2UEditResult(
            q_edited=q_edited,
            gate_s=s_mat,
            gate_s_max=s_max,
            z_score=z_mat,
            direction_g=torch.stack([p.g for p in self.profiles], dim=0),
            mu_t=torch.stack([p.mu_t for p in self.profiles], dim=0),
            realized_angle=angles,
            eta=etas,
            branch=branches,
        )
