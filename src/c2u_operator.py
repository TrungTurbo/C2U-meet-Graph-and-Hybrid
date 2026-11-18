#!/usr/bin/env python3
"""
c2u_operator.py
Cài đặt toán học 4 bước chỉnh sửa truy vấn C2U theo đặc tả chuẩn:
1. Hướng triệt tiêu tiền điều kiện:
   v = normalize(mu_t - mu_c)
   D_ii = 0.9 * E_ctrl[x_i^2] + 0.1 * m_bar + 0.001
   g = normalize(-D^{-1} v)
2. Cổng kích hoạt hiệu chuẩn:
   z(q) = q^T normalize(mu_t) - q^T normalize(mu_c)
   s(q) = clip((z(q) - z_0) / (z_1 - z_0), 0, 1) với z_0 (50%), z_1 (95%) trên calibration_control
3. Bổ sung nguyên tử phần dư & Delta:
   a = lstsq(A^T, g), e = g - A^T a, r = e / ||e||
   A' = [A; r], b_beta = [a; beta * ||e||] với beta = 1
   theta_req = 0.3 * s(q), theta kha thi, k_q = sin(theta) / sin(phi - theta)
   w'_q = [w_q; 0] + k_q * b_beta, delta_q = (w'_q - [w_q; 0])^T A' = k_q * g
4. Cập nhật vùng tin cậy (Trust-Region Update):
   q' = normalize(q + eta * delta_q) sao cho <q', q> >= max(cos(0.2), 0.98)
"""

import math
import torch
import torch.nn.functional as F
from typing import Tuple, Optional


class C2UOperator:
    """
    Toán tử suy luận C2U (Compositional Concept Unlearning Operator).
    """
    def __init__(
        self,
        A: torch.Tensor,
        target_profile_embs: torch.Tensor,
        calibration_control_embs: torch.Tensor,
        tau: float = 0.1,
        beta: float = 1.0,
        theta_req_factor: float = 0.3,
        trust_cos_thresh: float = max(math.cos(0.2), 0.98),
        device: str = "cpu"
    ):
        self.device = device
        self.tau = tau
        self.beta = beta
        self.theta_req_factor = theta_req_factor
        self.trust_cos_thresh = trust_cos_thresh
        
        # Chuyển tensors sang device
        self.A = F.normalize(A.to(device), p=2, dim=-1)  # (N, d)
        self.target_embs = F.normalize(target_profile_embs.to(device), p=2, dim=-1)  # (20, d)
        self.ctrl_embs = F.normalize(calibration_control_embs.to(device), p=2, dim=-1)  # (M, d)
        
        self.N, self.d = self.A.shape
        
        # Tiền tính toán Bước 1, 2 và Bước 3 (phần dư r, A', b_beta)
        self._precompute()
        
    def _precompute(self):
        """Tiền tính toán các đại lượng tĩnh phục vụ suy luận thời gian thực."""
        # 1. Hướng triệt tiêu tiền điều kiện
        mu_t = torch.mean(self.target_embs, dim=0)
        mu_c = torch.mean(self.ctrl_embs, dim=0)
        self.mu_t_norm = F.normalize(mu_t, p=2, dim=-1)
        self.mu_c_norm = F.normalize(mu_c, p=2, dim=-1)
        
        v = F.normalize(mu_t - mu_c, p=2, dim=-1)  # (d,)
        
        # Ma trận đường chéo hiệp phương sai D_ii
        e_x2 = torch.mean(self.ctrl_embs ** 2, dim=0)  # (d,)
        m_bar = torch.mean(e_x2).item()
        D_diag = 0.9 * e_x2 + 0.1 * m_bar + 0.001       # (d,)
        
        # Hướng g = normalize(-D^{-1} v)
        minus_d_inv_v = - (v / D_diag)
        self.g = F.normalize(minus_d_inv_v, p=2, dim=-1)  # (d,)
        
        # 2. Cổng kích hoạt hiệu chuẩn trên calibration_control
        z_ctrl = torch.matmul(self.ctrl_embs, self.mu_t_norm) - torch.matmul(self.ctrl_embs, self.mu_c_norm)
        self.z_0 = torch.quantile(z_ctrl, 0.50).item()
        self.z_1 = torch.quantile(z_ctrl, 0.95).item()
        if self.z_1 <= self.z_0:
            self.z_1 = self.z_0 + 1e-4
            
        # 3. Phân rã nguyên tử phần dư qua lstsq(A^T, g)
        # A in R^{N x d} => A^T in R^{d x N}, g in R^d
        # Giải hệ A^T @ a = g với a in R^N
        # torch.linalg.lstsq nhận A_matrix (d, N), B_matrix (d, 1)
        At = self.A.t()  # (d, N)
        g_col = self.g.unsqueeze(-1)  # (d, 1)
        lstsq_res = torch.linalg.lstsq(At, g_col)
        self.a = lstsq_res.solution.squeeze(-1)  # (N,)
        
        # Phần dư e = g - A^T @ a = g - a @ A
        recon_g = torch.matmul(self.a, self.A)  # (d,)
        e = self.g - recon_g
        e_norm = torch.norm(e).item()
        
        if e_norm > 1e-6:
            self.r = e / e_norm
        else:
            self.r = self.g
            e_norm = 0.0
            
        # Ngân hàng mở rộng A' = [A; r] in R^{(N+1) x d}
        self.A_prime = torch.cat([self.A, self.r.unsqueeze(0)], dim=0)  # (N+1, d)
        
        # Hệ số b_beta = [a; beta * ||e||] in R^{N+1}
        self.b_beta = torch.cat([self.a, torch.tensor([self.beta * e_norm], device=self.device)], dim=0)  # (N+1,)
        
    def compute_gate(self, q: torch.Tensor) -> torch.Tensor:
        """
        Bước 2: Cổng kích hoạt s(q) in [0, 1].
        q: shape (..., d)
        """
        q_norm = F.normalize(q, p=2, dim=-1)
        z = torch.matmul(q_norm, self.mu_t_norm) - torch.matmul(q_norm, self.mu_c_norm)
        s = torch.clamp((z - self.z_0) / (self.z_1 - self.z_0), 0.0, 1.0)
        return s
        
    def compute_shift(self, q: torch.Tensor, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Bước 3: Tính delta_q = (w'_q - [w_q; 0])^T A' theo theta_req = 0.3 * s(q).
        q: shape (..., d)
        s: shape (...)
        Trả về: delta_q (..., d), k_q (...)
        """
        q_norm = F.normalize(q, p=2, dim=-1)
        
        # sim(q, g) -> phi = arccos(<q, g>)
        # g có norm = 1, q_norm có norm = 1
        cos_phi = torch.sum(q_norm * self.g, dim=-1)  # (...)
        cos_phi = torch.clamp(cos_phi, -1.0 + 1e-7, 1.0 - 1e-7)
        phi = torch.acos(cos_phi)  # (...)
        
        # theta_req = 0.3 * s(q)
        theta_req = self.theta_req_factor * s  # (...)
        
        # Góc khả thi theta: đảm bảo phi - theta > 1e-4 để tránh kỳ dị định lý sin
        eps_angle = 1e-4
        max_feasible_theta = torch.clamp(phi - eps_angle, min=0.0)
        theta = torch.minimum(theta_req, max_feasible_theta)
        
        # Hệ số k_q = sin(theta) / sin(phi - theta)
        sin_theta = torch.sin(theta)
        sin_diff = torch.sin(phi - theta)
        sin_diff = torch.clamp(sin_diff, min=1e-6)
        
        # Khi s(q) = 0 => theta = 0 => k_q = 0
        k_q = torch.where(s > 1e-6, sin_theta / sin_diff, torch.zeros_like(s))  # (...)
        
        # delta_q = (w'_q - [w_q; 0])^T A' = k_q * (b_beta^T A') = k_q * g
        delta_q = k_q.unsqueeze(-1) * self.g  # (..., d)
        return delta_q, k_q
        
    def trust_region_update(self, q: torch.Tensor, delta_q: torch.Tensor) -> torch.Tensor:
        """
        Bước 4: Cập nhật vùng tin cậy q' = normalize(q + eta * delta_q)
        đảm bảo <q', q> >= max(cos(0.2), 0.98) ~ 0.980066.
        """
        q_norm = F.normalize(q, p=2, dim=-1)
        
        # Thử với eta = 1.0 trước
        eta = 1.0
        q_prime = F.normalize(q_norm + eta * delta_q, p=2, dim=-1)
        cos_sim = torch.sum(q_prime * q_norm, dim=-1, keepdim=True)  # (..., 1)
        
        # Backtracking nếu cos_sim < trust_cos_thresh
        max_iters = 15
        for _ in range(max_iters):
            mask = cos_sim < self.trust_cos_thresh
            if not mask.any():
                break
            eta *= 0.8
            q_prime = F.normalize(q_norm + eta * delta_q, p=2, dim=-1)
            cos_sim = torch.sum(q_prime * q_norm, dim=-1, keepdim=True)
            
        return q_prime
        
    def edit_query(self, q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quy trình trọn vẹn 4 bước chỉnh sửa query:
        q -> s(q) -> delta_q -> q'
        Trả về (q_prime, s)
        """
        s = self.compute_gate(q)
        delta_q, _ = self.compute_shift(q, s)
        q_prime = self.trust_region_update(q, delta_q)
        return q_prime, s
