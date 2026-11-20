#!/usr/bin/env python3
"""
src/acr_model.py
Kiến trúc ACR (Atomic Concept Representation) & 3 hàm Loss theo Giai đoạn 2 (v2):
- Ma trận từ điển A_raw in R^{N x d}, khởi tạo N(0, 1), chuẩn hóa mỗi hàng thành vector đơn vị.
- Head song tuyến sim(h_x, a_i) = h_x^T W a_i với W in R^{d x d}, khởi tạo W = I.
- Trọng số nguyên tử: w_x = softmax(sim(h_x, A) / tau) với tau = 0.1.
- Vector tái tạo: z_x = (w_x A) / (||w_x A||_2 + eps).
- 3 hàm Loss:
    * L_tri (margin m_r = 0.3)
    * L_align (lambda_align = 1.0)
    * L_ent (lambda_ent = 0.02, eps = 1e-12)
- Kích thước N chuẩn theo Mục A.1:
    * ArXiv: N = 64
    * TOFU / Amazon / MNLI: MiniLM (N=768), bge/e5 (N=1536), Qwen3 (N=2048)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any


def get_dictionary_size_N(dataset_name: str, dimension: int) -> int:
    """Xác định số lượng nguyên tử N chuẩn xác theo bài báo (Mục A.1)."""
    ds_lower = dataset_name.lower()
    if "arxiv" in ds_lower:
        return 64
    if dimension == 384:
        return 768
    elif dimension == 768:
        return 1536
    elif dimension >= 1024:
        return 2048
    return 768


class ACRModel(nn.Module):
    """
    Mô hình biểu diễn khái niệm nguyên tử ACR.
    Gồm ma trận từ điển A in R^{N x d} và ma trận chiếu song tuyến W in R^{d x d}.
    """
    def __init__(self, d: int, N: int, tau: float = 0.1):
        super().__init__()
        self.d = d
        self.N = N
        self.tau = tau

        # 1. Khởi tạo A_raw từ phân phối chuẩn N(0, 1) (Mục A.3)
        self.A_raw = nn.Parameter(torch.randn(N, d))

        # 2. Khởi tạo Head song tuyến W in R^{d x d}, khởi tạo W = I (Identity)
        self.W = nn.Parameter(torch.eye(d))

    def get_normalized_dictionary(self) -> torch.Tensor:
        """Chuẩn hóa mỗi hàng của A thành vector đơn vị (unit directions)."""
        return F.normalize(self.A_raw, p=2, dim=-1)

    def forward_single(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Tính trọng số nguyên tử w_x và vector tái tạo z_x cho batch h.
        Input: h shape [B, d]
        Output:
          w: shape [B, N]
          z: shape [B, d]
        """
        # A có kích thước [N, d], chuẩn hóa L2 từng hàng
        A = self.get_normalized_dictionary()

        # sim(h, a_i) = h^T W a_i -> [B, d] @ [d, d] -> [B, d] @ [d, N] -> [B, N]
        h_proj = torch.matmul(h, self.W)
        sim = torch.matmul(h_proj, A.t())  # [B, N]

        # Trọng số Softmax w_x với nhiệt độ tau = 0.1
        w = F.softmax(sim / self.tau, dim=-1)  # [B, N]

        # Tái tạo z_x = normalize(w @ A)
        z_unnorm = torch.matmul(w, A)  # [B, d]
        z = z_unnorm / (torch.norm(z_unnorm, p=2, dim=-1, keepdim=True) + 1e-8)

        return w, z

    def forward(
        self,
        q: torch.Tensor,
        d_pos: torch.Tensor,
        d_neg: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Forward bộ ba (q, d^+, d^-)."""
        w_q, z_q = self.forward_single(q)
        w_pos, z_pos = self.forward_single(d_pos)
        w_neg, z_neg = self.forward_single(d_neg)

        return {
            "w_q": w_q, "z_q": z_q,
            "w_pos": w_pos, "z_pos": z_pos,
            "w_neg": w_neg, "z_neg": z_neg
        }


class ACRLoss(nn.Module):
    """
    Bộ 3 hàm Loss của ACR:
    L = L_tri + lambda_align * L_align + lambda_ent * L_ent
    """
    def __init__(
        self,
        mr: float = 0.3,
        lambda_align: float = 1.0,
        lambda_ent: float = 0.02,
        eps_log: float = 1e-12
    ):
        super().__init__()
        self.mr = mr
        self.lambda_align = lambda_align
        self.lambda_ent = lambda_ent
        self.eps_log = eps_log

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        q_raw: torch.Tensor,
        d_pos_raw: torch.Tensor,
        d_neg_raw: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        z_q = outputs["z_q"]
        z_pos = outputs["z_pos"]
        z_neg = outputs["z_neg"]

        w_q = outputs["w_q"]
        w_pos = outputs["w_pos"]
        w_neg = outputs["w_neg"]

        # 1. Triplet Loss: max(0, mr - cos(z_q, z_pos) + cos(z_q, z_neg))
        cos_pos = (z_q * z_pos).sum(dim=-1)
        cos_neg = (z_q * z_neg).sum(dim=-1)
        loss_tri_elements = F.relu(self.mr - cos_pos + cos_neg)
        loss_tri = loss_tri_elements.mean()

        # Triplet accuracy (tỷ lệ cos_pos > cos_neg)
        triplet_acc = (cos_pos > cos_neg).float().mean()

        # 2. Alignment Loss: sum_{x in {q, pos, neg}} (1 - cos(z_x, e_x^base))
        # Vì z_x và e_x^base đều là vector đơn vị, cos(z_x, e_x^base) = z_x . e_x^base
        cos_align_q = (z_q * q_raw).sum(dim=-1)
        cos_align_pos = (z_pos * d_pos_raw).sum(dim=-1)
        cos_align_neg = (z_neg * d_neg_raw).sum(dim=-1)

        loss_align = ((1.0 - cos_align_q) + (1.0 - cos_align_pos) + (1.0 - cos_align_neg)).mean()
        mean_align_cos = (cos_align_q + cos_align_pos + cos_align_neg).mean() / 3.0

        # 3. Entropy Loss: - sum_k w_k log(w_k + eps)
        def entropy(w):
            return - (w * torch.log(w + self.eps_log)).sum(dim=-1)

        ent_q = entropy(w_q)
        ent_pos = entropy(w_pos)
        ent_neg = entropy(w_neg)
        loss_ent = (ent_q + ent_pos + ent_neg).mean()

        # Tổng hợp Loss
        total_loss = loss_tri + (self.lambda_align * loss_align) + (self.lambda_ent * loss_ent)

        return {
            "total_loss": total_loss,
            "loss_tri": loss_tri,
            "loss_align": loss_align,
            "loss_ent": loss_ent,
            "triplet_acc": triplet_acc,
            "mean_align_cos": mean_align_cos
        }
