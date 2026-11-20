#!/usr/bin/env python3
"""
run_kaggle.py
Kịch bản thực thi toàn diện trên Kaggle GPU (T4 hoặc P100):
1. Tự động kiểm tra GPU và cài đặt thư viện cần thiết.
2. Kiểm tra tính toàn vẹn và rời rạc của 15 split (Gói 1.4).
3. Mã hóa văn bản đóng băng (Frozen Pretrained Encoders) cho 48 unit ra cache .pt.
4. Huấn luyện từ điển khái niệm nguyên tử ACR (Giai đoạn 2) cho toàn bộ 48 unit.
5. Xuất báo cáo tóm tắt các checkpoint và loss.
"""

import os
import sys
import time
import argparse
import torch

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(ROOT_DIR, "src")
TESTS_DIR = os.path.join(ROOT_DIR, "tests")
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if TESTS_DIR not in sys.path:
    sys.path.insert(0, TESTS_DIR)

from src.encode_cache import encode_unit
from src.train_acr import train_single_unit
try:
    from tests.test_data_integrity import test_splits_integrity
except ImportError:
    from test_data_integrity import test_splits_integrity


def run_pipeline(
    datasets=None,
    encoders=None,
    seeds=None,
    max_epochs=10,
    device=None
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if datasets is None:
        datasets = ["tofu", "amazon", "mnli", "arxiv"]
    if encoders is None:
        encoders = ["all-MiniLM-L6-v2", "bge-base-en-v1.5", "e5-base-v2", "Qwen3-Embedding-0.6B"]
    if seeds is None:
        seeds = [1, 2, 3]

    print("=" * 70)
    print("PIPELINE C2U - GIAI ĐOẠN 1 & 2 TRÊN KAGGLE GPU")
    print(f"Thiết bị: {device} | CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Model: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    print(f"Danh sách Dataset ({len(datasets)}): {datasets}")
    print(f"Danh sách Encoders ({len(encoders)}): {encoders}")
    print(f"Danh sách Seeds ({len(seeds)}): {seeds}")
    print("=" * 70)

    # BƯỚC 1: Kiểm thử toàn vẹn dữ liệu
    print("\n>>> BƯỚC 1: Kiểm tra toàn vẹn và rời rạc dữ liệu 15 split...")
    test_splits_integrity()

    # BƯỚC 2: Mã hóa vector đóng băng & Huấn luyện ACR cho từng unit
    print("\n>>> BƯỚC 2 & 3: Mã hóa embedding đóng băng & Huấn luyện ACR...")
    os.makedirs("cache_embeddings", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    completed_units = 0
    total_units = len(datasets) * len(seeds) * len(encoders)
    t0_all = time.time()

    summary_results = []

    for ds in datasets:
        for s in seeds:
            split_dir = os.path.join("processed_splits", f"{ds}_seed{s}")
            if not os.path.exists(split_dir):
                print(f"[Cảnh báo] Bỏ qua split không tồn tại: {split_dir}")
                continue

            for enc in encoders:
                unit_name = f"{ds}_seed{s}_{enc}"
                print(f"\n------------------------------------------------------------")
                print(f"[{completed_units + 1}/{total_units}] Đang xử lý Unit: {unit_name}")
                print(f"------------------------------------------------------------")
                t0_unit = time.time()

                # 2.1 Mã hóa cache .pt (nếu chưa có)
                short_enc = {
                    "all-MiniLM-L6-v2": "minilm",
                    "bge-base-en-v1.5": "bge",
                    "e5-base-v2": "e5",
                    "Qwen3-Embedding-0.6B": "qwen3"
                }.get(enc, enc)

                cache_file_canonical = os.path.join("cache_embeddings", f"{ds}_seed{s}_{enc.replace('/', '_')}.pt")
                cache_file_short = os.path.join("cache_embeddings", f"{ds}_seed{s}_{short_enc}.pt")

                if os.path.exists(cache_file_short):
                    cache_path = cache_file_short
                    print(f"  [Đã có cache] {cache_path}")
                elif os.path.exists(cache_file_canonical):
                    cache_path = cache_file_canonical
                    print(f"  [Đã có cache] {cache_path}")
                else:
                    cache_path = encode_unit(
                        split_dir=split_dir,
                        encoder_key=enc,
                        output_dir="cache_embeddings",
                        device=device
                    )

                # 2.2 Huấn luyện ACR (nếu chưa có checkpoint)
                ckpt_file_canonical = os.path.join("checkpoints", f"acr_{ds}_seed{s}_{enc.replace('/', '_')}.pt")
                ckpt_file_short = os.path.join("checkpoints", f"acr_{ds}_seed{s}_{short_enc}.pt")

                if os.path.exists(ckpt_file_short):
                    ckpt_path = ckpt_file_short
                    print(f"  [Đã có checkpoint] Bỏ qua huấn luyện: {ckpt_path}")
                elif os.path.exists(ckpt_file_canonical):
                    ckpt_path = ckpt_file_canonical
                    print(f"  [Đã có checkpoint] Bỏ qua huấn luyện: {ckpt_path}")
                else:
                    ckpt_path = train_single_unit(
                        cache_path=cache_path,
                        output_dir="checkpoints",
                        max_epochs=max_epochs,
                        device=device,
                        seed=s * 100 + 42
                    )

                # Dọn dẹp GPU sau mỗi unit
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                import gc
                gc.collect()

                elapsed_unit = time.time() - t0_unit
                completed_units += 1

                # Đọc kết quả từ checkpoint
                ckpt_data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                cfg = ckpt_data["config"]
                summary_results.append({
                    "unit": unit_name,
                    "dataset": ds,
                    "seed": s,
                    "encoder": enc,
                    "d": cfg["d"],
                    "N": cfg["N"],
                    "best_epoch": cfg["best_epoch"],
                    "best_val_loss": cfg["best_val_loss"],
                    "time_sec": elapsed_unit
                })

    print("\n" + "=" * 70)
    print(f"HOÀN THÀNH TOÀN BỘ {completed_units}/{total_units} UNITS TRONG {time.time() - t0_all:.1f} GIÂY!")
    print("=" * 70)
    print(f"{'Unit':<35} | {'d':<5} | {'N':<5} | {'Epoch':<6} | {'Val Loss':<10} | {'Time (s)':<8}")
    print("-" * 75)
    for res in summary_results:
        print(f"{res['unit']:<35} | {res['d']:<5} | {res['N']:<5} | {res['best_epoch']:<6} | {res['best_val_loss']:<10.4f} | {res['time_sec']:<8.1f}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Chạy Pipeline C2U Giai đoạn 1 & 2 trên Kaggle GPU")
    parser.add_argument("--dataset", type=str, default="all", help="Dataset (tofu, amazon, mnli, arxiv, hoặc all)")
    parser.add_argument("--encoder", type=str, default="all", help="Encoder (minilm, bge, e5, qwen3, hoặc all)")
    parser.add_argument("--seed", type=int, default=1, help="Seed (1, 2, 3, hoặc 0 để chạy cả 3)")
    parser.add_argument("--max_epochs", type=int, default=10, help="Số epoch tối đa cho ACR (1 để test nhanh)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    ds_map = {"tofu": ["tofu"], "amazon": ["amazon"], "mnli": ["mnli"], "arxiv": ["arxiv"], "all": ["tofu", "amazon", "mnli", "arxiv"]}
    enc_map = {
        "minilm": ["all-MiniLM-L6-v2"],
        "bge": ["bge-base-en-v1.5"],
        "e5": ["e5-base-v2"],
        "qwen3": ["Qwen3-Embedding-0.6B"],
        "all": ["all-MiniLM-L6-v2", "bge-base-en-v1.5", "e5-base-v2", "Qwen3-Embedding-0.6B"]
    }
    seeds = [1, 2, 3] if args.seed == 0 else [args.seed]

    run_pipeline(
        datasets=ds_map.get(args.dataset, ["tofu"]),
        encoders=enc_map.get(args.encoder, ["all-MiniLM-L6-v2"]),
        seeds=seeds,
        max_epochs=args.max_epochs,
        device=args.device
    )


if __name__ == "__main__":
    main()
