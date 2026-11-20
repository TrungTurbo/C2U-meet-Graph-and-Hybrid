#!/usr/bin/env python3
"""
src/train_acr.py
Huấn luyện Offline Từ điển Nguyên tử ACR (Gói 2.3):
- Dữ liệu: Cache embeddings (train_q_embs, train_d_embs) từ GĐ1.
- Tự động xác định số lượng nguyên tử N theo Mục A.1.
- Optimizer AdamW (lr=1e-3), ReduceLROnPlateau (factor=0.5, patience=1 theo val total loss).
- Early stopping (patience=2), hỗ trợ max_epochs=1 (mốc tái lập A.6) hoặc max_epochs=10.
- Lưu checkpoint checkpoints/acr_<dataset>_seed<k>_<encoder>.pt với A đã chuẩn hóa hàng.
- Báo cáo đầy đủ 3 thành phần loss, triplet accuracy và mean alignment cosine.
"""

import os
import sys

# Đảm bảo import được các module cùng thư mục src dù gọi từ thư mục nào
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import argparse
import random
import hashlib
import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

try:
    from acr_model import ACRModel, ACRLoss, get_dictionary_size_N
    from acr_dataset import create_acr_dataloaders
except ImportError:
    from src.acr_model import ACRModel, ACRLoss, get_dictionary_size_N
    from src.acr_dataset import create_acr_dataloaders


def compute_file_sha256(filepath: str) -> str:
    """Tính SHA256 của file."""
    if not os.path.exists(filepath):
        return ""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def set_seed(seed: int = 42):
    """Cố định seed trên toàn bộ thư viện phục vụ tính tái lập."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def train_single_unit(
    cache_path: str,
    output_dir: str = "checkpoints",
    max_epochs: int = 10,
    lr: float = 1e-3,
    batch_size: int = 32,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    seed: int = 42
) -> str:
    """Huấn luyện ACR cho một unit embedding cache hoàn chỉnh."""
    os.makedirs(output_dir, exist_ok=True)
    set_seed(seed)

    print(f"\n" + "=" * 65)
    print(f"[ACR Train] Đang tải cache embedding: {cache_path}")
    print("=" * 65)

    data = torch.load(cache_path, map_location="cpu", weights_only=False)
    q_embs = data["train_q_embs"].float()
    d_embs = data["train_d_embs"].float()
    d = q_embs.shape[1]

    # Phân tích tên split và encoder từ tên file cache
    base_name = os.path.splitext(os.path.basename(cache_path))[0]
    parts = base_name.split("_")
    # ví dụ: tofu_seed1_minilm hoặc tofu_seed1_all-MiniLM-L6-v2
    dataset_name = parts[0]
    seed_tag = parts[1] if len(parts) > 1 else "seed1"
    encoder_tag = "_".join(parts[2:]) if len(parts) > 2 else "minilm"

    N = get_dictionary_size_N(dataset_name, d)
    print(f"  - Chiều biểu diễn d: {d}")
    print(f"  - Số lượng nguyên tử N: {N}")
    print(f"  - Thiết bị: {device}, Số lượng cặp huấn luyện: {len(q_embs)}")

    # Tạo DataLoader
    train_loader, val_loader, train_ds, val_ds = create_acr_dataloaders(
        q_embs=q_embs,
        d_embs=d_embs,
        val_ratio=0.1,
        batch_size=batch_size,
        seed=seed,
        device=device
    )

    model = ACRModel(d=d, N=N, tau=0.1).to(device)
    criterion = ACRLoss(mr=0.3, lambda_align=1.0, lambda_ent=0.02).to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)

    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None
    no_improve_count = 0
    history = []

    # SHA256 file dữ liệu gốc
    split_dir = os.path.join("processed_splits", f"{dataset_name}_{seed_tag}")
    train_jsonl_path = os.path.join(split_dir, "train_acr.jsonl")
    train_sha256 = compute_file_sha256(train_jsonl_path)

    for epoch in range(max_epochs):
        train_ds.set_epoch(epoch)
        model.train()
        train_totals = {"total": 0.0, "tri": 0.0, "align": 0.0, "ent": 0.0, "acc": 0.0, "align_cos": 0.0}
        n_train_batches = 0

        for q_b, pos_b, neg_b in train_loader:
            optimizer.zero_grad()
            outputs = model(q_b, pos_b, neg_b)
            loss_dict = criterion(outputs, q_b, pos_b, neg_b)
            loss = loss_dict["total_loss"]
            loss.backward()
            optimizer.step()

            train_totals["total"] += loss.item()
            train_totals["tri"] += loss_dict["loss_tri"].item()
            train_totals["align"] += loss_dict["loss_align"].item()
            train_totals["ent"] += loss_dict["loss_ent"].item()
            train_totals["acc"] += loss_dict["triplet_acc"].item()
            train_totals["align_cos"] += loss_dict["mean_align_cos"].item()
            n_train_batches += 1

        # Validation
        model.eval()
        val_totals = {"total": 0.0, "tri": 0.0, "align": 0.0, "ent": 0.0, "acc": 0.0, "align_cos": 0.0}
        n_val_batches = 0

        with torch.no_grad():
            for q_b, pos_b, neg_b in val_loader:
                outputs = model(q_b, pos_b, neg_b)
                loss_dict = criterion(outputs, q_b, pos_b, neg_b)
                val_totals["total"] += loss_dict["total_loss"].item()
                val_totals["tri"] += loss_dict["loss_tri"].item()
                val_totals["align"] += loss_dict["loss_align"].item()
                val_totals["ent"] += loss_dict["loss_ent"].item()
                val_totals["acc"] += loss_dict["triplet_acc"].item()
                val_totals["align_cos"] += loss_dict["mean_align_cos"].item()
                n_val_batches += 1

        train_summary = {k: v / max(1, n_train_batches) for k, v in train_totals.items()}
        val_summary = {k: v / max(1, n_val_batches) for k, v in val_totals.items()}
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"  Epoch [{epoch+1:02d}/{max_epochs:02d}] "
            f"Train Loss: {train_summary['total']:.4f} (tri={train_summary['tri']:.4f}, align={train_summary['align']:.4f}, ent={train_summary['ent']:.4f}) | "
            f"Val Loss: {val_summary['total']:.4f} (tri={val_summary['tri']:.4f}, acc={val_summary['acc']*100:.1f}%, align_cos={val_summary['align_cos']:.3f}) | "
            f"LR: {current_lr:.1e}"
        )

        history.append({
            "epoch": epoch + 1,
            "train": train_summary,
            "val": val_summary,
            "lr": current_lr
        })

        scheduler.step(val_summary["total"])

        # Theo dõi checkpoint tốt nhất theo val total loss
        if val_summary["total"] < best_val_loss:
            best_val_loss = val_summary["total"]
            best_epoch = epoch + 1
            no_improve_count = 0
            best_state = {
                "A_normalized": model.get_normalized_dictionary().detach().cpu(),
                "W": model.W.detach().cpu()
            }
        else:
            no_improve_count += 1
            if no_improve_count >= 2 and epoch >= 2:
                print(f"  -> Dừng sớm (Early Stopping) tại epoch {epoch+1} vì Val Loss không giảm sau 2 epoch liên tiếp.")
                break

    # Nghiệm thu chuẩn: A có hàng chuẩn 1
    A_final = best_state["A_normalized"]
    norms = torch.norm(A_final, p=2, dim=-1)
    diff = (norms - 1.0).abs().max().item()
    assert diff < 1e-4, f"Lỗi: Các hàng của A chưa đạt chuẩn đơn vị! Max diff = {diff}"
    assert not torch.isnan(A_final).any(), "Phát hiện NaN trong A!"

    # Lưu checkpoint
    ckpt_filename = f"acr_{dataset_name}_{seed_tag}_{encoder_tag}.pt"
    ckpt_path = os.path.join(output_dir, ckpt_filename)

    checkpoint_data = {
        "A": A_final,
        "W": best_state["W"],
        "config": {
            "dataset": dataset_name,
            "seed_tag": seed_tag,
            "encoder_tag": encoder_tag,
            "d": d,
            "N": N,
            "tau": 0.1,
            "mr": 0.3,
            "lambda_align": 1.0,
            "lambda_ent": 0.02,
            "lr": lr,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "train_acr_sha256": train_sha256
        },
        "history": history
    }
    torch.save(checkpoint_data, ckpt_path)
    print(f"  -> Đã lưu checkpoint thành công: {ckpt_path} (Best Epoch: {best_epoch}, Val Loss: {best_val_loss:.4f})")
    return ckpt_path


def main():
    parser = argparse.ArgumentParser(description="Huấn luyện ACR offline theo chuẩn Giai đoạn 2 (v2)")
    parser.add_argument("--cache_path", type=str, required=True, help="Đường dẫn file cache .pt từ GĐ1")
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--max_epochs", type=int, default=10, help="Số epoch tối đa (1 cho mốc tái lập, 10 cho đầy đủ)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    train_single_unit(
        cache_path=args.cache_path,
        output_dir=args.output_dir,
        max_epochs=args.max_epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        device=args.device,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
