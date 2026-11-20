#!/usr/bin/env python3
"""
src/acr_dataset.py
Dữ liệu bộ ba huấn luyện ACR (Gói 2.1):
- Nguồn: chỉ dùng rổ `train_acr`. Embedding lấy sẵn từ cache (train_q_embs, train_d_embs).
- Tuyệt đối KHÔNG lấy d^- từ index_corpus (chống rò rỉ dữ liệu).
- Sinh bộ ba (q, d^+, d^-) ngẫu nhiên nội bộ train_acr mỗi epoch bằng torch.Generator có seed.
- Chia train/val: 90/10 có seed cố định.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional


class ACRTripletDataset(Dataset):
    """
    Dataset sinh bộ ba (q, d^+, d^-) trực tiếp từ embeddings đã cache trong train_acr.
    q: vector truy vấn thứ i (train_q_embs[i])
    d^+: vector tài liệu dương ghép cặp (train_d_embs[i])
    d^-: vector tài liệu âm ngẫu nhiên khác d^+, chỉ lấy trong nội bộ train_d_embs.
    """
    def __init__(
        self,
        q_embs: torch.Tensor,
        d_embs: torch.Tensor,
        seed: int = 42
    ):
        assert q_embs.shape[0] == d_embs.shape[0], "Số lượng query và document trong train_acr phải bằng nhau"
        self.q_embs = q_embs.float()
        self.d_embs = d_embs.float()
        self.num_samples = q_embs.shape[0]
        self.seed = seed
        self.epoch = 0
        self.rng = torch.Generator()
        self.rng.manual_seed(seed)
        self.resample_negatives()

    def set_epoch(self, epoch: int):
        """Cập nhật epoch và lấy mẫu lại d^- để tăng tính đa dạng theo bài báo."""
        self.epoch = epoch
        self.rng.manual_seed(self.seed + epoch * 1000)
        self.resample_negatives()

    def resample_negatives(self):
        """Lấy mẫu d^- ngẫu nhiên cho mỗi mẫu i sao cho d^- != d^+ (ngẫu nhiên trong rổ train_acr)."""
        # Sinh hoán vị hoặc chỉ số ngẫu nhiên khác i
        offsets = torch.randint(1, self.num_samples, (self.num_samples,), generator=self.rng)
        self.neg_indices = (torch.arange(self.num_samples) + offsets) % self.num_samples

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        neg_idx = self.neg_indices[idx].item()
        return self.q_embs[idx], self.d_embs[idx], self.d_embs[neg_idx]


def create_acr_dataloaders(
    q_embs: torch.Tensor,
    d_embs: torch.Tensor,
    val_ratio: float = 0.1,
    batch_size: int = 32,
    seed: int = 42,
    device: str = "cpu"
) -> Tuple[DataLoader, DataLoader, ACRTripletDataset, ACRTripletDataset]:
    """
    Chia 90/10 train/val và tạo DataLoader cho ACR training.
    """
    num_samples = q_embs.shape[0]
    num_val = int(num_samples * val_ratio)
    num_train = num_samples - num_val

    # Chia train/val cố định bằng generator có seed
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(num_samples, generator=gen)
    train_idx = perm[:num_train]
    val_idx = perm[num_train:]

    train_dataset = ACRTripletDataset(
        q_embs[train_idx].to(device),
        d_embs[train_idx].to(device),
        seed=seed
    )
    val_dataset = ACRTripletDataset(
        q_embs[val_idx].to(device),
        d_embs[val_idx].to(device),
        seed=seed + 999
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    return train_loader, val_loader, train_dataset, val_dataset
