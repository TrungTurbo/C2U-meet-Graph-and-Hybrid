#!/usr/bin/env python3
"""
train_acr.py
Huấn luyện ACR (Atomic Concept Representation) theo Mục 4.C:
- Dictionary A in R^{N x d} (N=768 cho MiniLM, N=1536 cho bge/e5, N=64 cho ArXiv)
- Projection Head f_theta
- AdamW (lr=1e-3, batch_size=32)
- Scheduler ReduceLROnPlateau (factor 0.5, patience 1)
- Train/Val Split 90/10, Early Stopping (patience=3)
- Lưu checkpoint acr_checkpoint.pt
"""

import os
import argparse
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
from acr_model import ACRModel, ACRLoss


def train_acr(
    cache_dir: str,
    output_dir: str,
    d: int = 384,
    N: int = 768,
    tau: float = 0.1,
    epochs: int = 15,
    batch_size: int = 32,
    lr: float = 1e-3,
    val_split: float = 0.1,
    early_stopping_patience: int = 3,
    device: str = "cuda"
):
    os.makedirs(output_dir, exist_ok=True)
    triplets_path = os.path.join(cache_dir, "train_triplets_emb.pt")
    if not os.path.exists(triplets_path):
        raise FileNotFoundError(f"Không tìm thấy file triplets cache tại: {triplets_path}")
        
    print(f"\n[ACR Training] Đang nạp cached triplets từ {triplets_path}...")
    data = torch.load(triplets_path, map_location="cpu")
    q_emb = data["q_emb"]
    pos_emb = data["pos_emb"]
    neg_emb = data["neg_emb"]
    
    # Tự động điều chỉnh d nếu cần
    actual_d = q_emb.shape[-1]
    if actual_d != d:
        print(f"[ACR Training] Tự động cập nhật d={actual_d} từ embeddings (thay vì d={d})")
        d = actual_d
        if N != 64:  # Giữ N=64 cho ArXiv nếu được chỉ định
            N = 1536 if d == 768 else 768
            
    print(f"[ACR Training] Cấu hình: d={d}, N={N}, tau={tau}, epochs={epochs}, batch_size={batch_size}, lr={lr}")
    
    full_dataset = TensorDataset(q_emb, pos_emb, neg_emb)
    total_samples = len(full_dataset)
    val_size = int(total_samples * val_split)
    train_size = total_samples - val_size
    
    train_ds, val_ds = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    print(f"  Train samples: {train_size} | Validation samples: {val_size}")
    
    model = ACRModel(d=d, N=N, tau=tau).to(device)
    criterion = ACRLoss(margin=0.3, lambda_align=1.0, lambda_ent=0.02)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    # Scheduler: ReduceLROnPlateau factor 0.5, patience 1
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)
    
    best_val_loss = float("inf")
    patience_counter = 0
    checkpoint_path = os.path.join(output_dir, "acr_checkpoint.pt")
    
    for epoch in range(1, epochs + 1):
        # 1. Training Phase
        model.train()
        train_loss = 0.0
        train_trip = 0.0
        train_align = 0.0
        train_ent = 0.0
        steps = 0
        
        for b_q, b_pos, b_neg in train_loader:
            b_q = b_q.to(device)
            b_pos = b_pos.to(device)
            b_neg = b_neg.to(device)
            
            optimizer.zero_grad()
            q_proj, w_q, _ = model(b_q)
            pos_proj, w_pos, _ = model(b_pos)
            neg_proj, w_neg, _ = model(b_neg)
            
            loss_dict = criterion(q_proj, pos_proj, neg_proj, w_q, w_pos, w_neg)
            loss = loss_dict["total_loss"]
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_trip += loss_dict["triplet_loss"].item()
            train_align += loss_dict["align_loss"].item()
            train_ent += loss_dict["ent_loss"].item()
            steps += 1
            
        avg_train_loss = train_loss / max(1, steps)
        avg_train_trip = train_trip / max(1, steps)
        avg_train_align = train_align / max(1, steps)
        avg_train_ent = train_ent / max(1, steps)
        
        # 2. Validation Phase
        model.eval()
        val_loss = 0.0
        val_steps = 0
        with torch.no_grad():
            for b_q, b_pos, b_neg in val_loader:
                b_q = b_q.to(device)
                b_pos = b_pos.to(device)
                b_neg = b_neg.to(device)
                
                q_proj, w_q, _ = model(b_q)
                pos_proj, w_pos, _ = model(b_pos)
                neg_proj, w_neg, _ = model(b_neg)
                
                loss_dict = criterion(q_proj, pos_proj, neg_proj, w_q, w_pos, w_neg)
                val_loss += loss_dict["total_loss"].item()
                val_steps += 1
                
        avg_val_loss = val_loss / max(1, val_steps)
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        
        print(f"Epoch [{epoch:02d}/{epochs:02d}] "
              f"Train Loss: {avg_train_loss:.4f} (Trip: {avg_train_trip:.4f}, Align: {avg_train_align:.4f}, Ent: {avg_train_ent:.4f}) | "
              f"Val Loss: {avg_val_loss:.4f} | LR: {current_lr:.6f}")
              
        # Early stopping logic
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "d": d,
                "N": N,
                "tau": tau,
                "best_val_loss": best_val_loss,
                "epoch": epoch
            }, checkpoint_path)
            print(f"  -> Lưu checkpoint mới (Best Val Loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"[ACR Training] Kích hoạt Early Stopping tại epoch {epoch}!")
                break
                
    print(f"\n[ACR Training] Hoàn tất huấn luyện. Checkpoint tối ưu tại: {checkpoint_path}")


def main():
    parser = argparse.ArgumentParser(description="Huấn luyện ACR Model trên triplet embeddings.")
    parser.add_argument("--cache_dir", type=str, default="cache_emb/tofu", help="Thư mục chứa cache embeddings")
    parser.add_argument("--output_dir", type=str, default="checkpoints/tofu", help="Thư mục lưu checkpoint")
    parser.add_argument("--d", type=int, default=384, help="Chiều biểu diễn embedding d")
    parser.add_argument("--N", type=int, default=768, help="Số nguyên tử từ điển N")
    parser.add_argument("--tau", type=float, default=0.1, help="Nhiệt độ softmax tau")
    parser.add_argument("--epochs", type=int, default=15, help="Số epoch tối đa")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Tốc độ học")
    parser.add_argument("--device", type=str, default=None, help="Device: cuda or cpu")
    args = parser.parse_args()
    
    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    train_acr(
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        d=args.d,
        N=args.N,
        tau=args.tau,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device
    )


if __name__ == "__main__":
    main()
