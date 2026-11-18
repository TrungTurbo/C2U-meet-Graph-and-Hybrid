#!/usr/bin/env python3
"""
run_kaggle_pipeline.py
Kịch bản tổng hợp chạy 1 lệnh duy nhất trên Kaggle Notebook (GPU T4/P100):
1. Tự động kiểm tra môi trường GPU & dependencies
2. Encode văn bản ra cache .pt qua SentenceTransformers
3. Huấn luyện ACR Model (Dictionary A + Projection Head)
4. Chạy toán tử C2U 4 bước và Đánh giá Top-5 Retrieval (Raw vs C2U)
5. Xuất báo cáo CSV & in bảng tóm tắt
"""

import os
import sys
import subprocess
import argparse

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(CURRENT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


def check_and_install_packages():
    """Kiểm tra và cài đặt dependencies nếu môi trường Kaggle chưa có."""
    try:
        import sentence_transformers
    except ImportError:
        print("[Env] Đang cài đặt sentence-transformers...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "sentence-transformers", "scikit-learn"])


def run_pipeline(
    dataset: str = "tofu",
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    data_dir: str = None,
    work_dir: str = None,
    epochs: int = 15,
    batch_size: int = 64
):
    check_and_install_packages()
    
    import torch
    from encode_cache import process_dataset_cache
    from train_acr import train_acr
    from evaluate import run_evaluation
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("\n" + "=" * 80)
    print(f"BẮT ĐẦU C2U PIPELINE | Tập dữ liệu: {dataset.upper()} | Mô hình: {model_name}")
    print(f"Device: {device} | PyTorch: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 80)
    
    # Xác định đường dẫn thư mục làm việc
    if work_dir is None:
        if os.path.exists("/kaggle/working"):
            work_dir = "/kaggle/working"
        else:
            work_dir = os.path.join(CURRENT_DIR, "kaggle_output")
    os.makedirs(work_dir, exist_ok=True)
    
    # Xác định đường dẫn dữ liệu input
    if data_dir is None:
        kaggle_candidate_1 = f"/kaggle/input/c2u-splits/{dataset}"
        kaggle_candidate_2 = f"/kaggle/working/processed_splits/{dataset}"
        local_candidate = os.path.join(CURRENT_DIR, "processed_splits", dataset)
        
        if os.path.exists(kaggle_candidate_1):
            data_dir = kaggle_candidate_1
        elif os.path.exists(kaggle_candidate_2):
            data_dir = kaggle_candidate_2
        elif os.path.exists(local_candidate):
            data_dir = local_candidate
        else:
            data_dir = local_candidate
            
    print(f"[*] Thư mục dữ liệu (Input) : {data_dir}")
    print(f"[*] Thư mục đầu ra (Output) : {work_dir}")
    
    model_tag = model_name.split("/")[-1]
    cache_dir = os.path.join(work_dir, "cache_emb", dataset, model_tag)
    ckpt_dir = os.path.join(work_dir, "checkpoints", dataset, model_tag)
    results_csv = os.path.join(work_dir, "c2u_final_results.csv")
    
    # GIAI ĐOẠN 1: ENCODE CACHE
    print("\n" + "---" * 20)
    print("GIAI ĐOẠN 1: TRÍCH XUẤT EMBEDDINGS & CACHE .PT")
    print("---" * 20)
    process_dataset_cache(
        split_dir=data_dir,
        output_dir=cache_dir,
        model_name=model_name,
        batch_size=batch_size,
        device=device
    )
    
    # GIAI ĐOẠN 2: HUẤN LUYỆN TỪ ĐIỂN ACR
    print("\n" + "---" * 20)
    print("GIAI ĐOẠN 2: HUẤN LUYỆN TỪ ĐIỂN NGUYÊN TỬ ACR")
    print("---" * 20)
    
    # Xác định d và N theo đặc tả:
    # N=768 cho MiniLM, N=1536 cho bge/e5, N=64 cho ArXiv
    if dataset == "arxiv":
        N = 64
        d = 768 if ("768" in model_name or "bge" in model_name or "e5" in model_name) else 384
    else:
        d = 768 if ("768" in model_name or "bge" in model_name or "e5" in model_name) else 384
        N = 1536 if d == 768 else 768
        
    print(f"[*] Cấu hình ACR: d={d}, N={N}")
    train_acr(
        cache_dir=cache_dir,
        output_dir=ckpt_dir,
        d=d,
        N=N,
        tau=0.1,
        epochs=epochs,
        batch_size=32,
        lr=1e-3,
        val_split=0.1,
        early_stopping_patience=3,
        device=device
    )
    
    # GIAI ĐOẠN 3: ĐÁNH GIÁ C2U VS RAW
    print("\n" + "---" * 20)
    print("GIAI ĐOẠN 3: ĐÁNH GIÁ TOP-5 RETRIEVAL (RAW vs C2U)")
    print("---" * 20)
    checkpoint_file = os.path.join(ckpt_dir, "acr_checkpoint.pt")
    run_evaluation(
        cache_dir=cache_dir,
        checkpoint_path=checkpoint_file,
        output_csv=results_csv,
        dataset_name=dataset,
        model_name=model_tag,
        k=5,
        device=device
    )
    
    print("\n" + "=" * 80)
    print(f"[HOÀN TẤT] Thực nghiệm thành công! Kết quả đã ghi vào: {results_csv}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Kịch bản điều phối toàn trình C2U trên Kaggle GPU.")
    parser.add_argument("--dataset", type=str, default="tofu", choices=["tofu", "mnli", "amazon", "arxiv", "all"], help="Tập dữ liệu")
    parser.add_argument("--model_name", type=str, default="sentence-transformers/all-MiniLM-L6-v2", help="Tên mô hình Encoder")
    parser.add_argument("--data_dir", type=str, default=None, help="Đường dẫn thư mục dữ liệu splits")
    parser.add_argument("--work_dir", type=str, default=None, help="Đường dẫn thư mục đầu ra")
    parser.add_argument("--epochs", type=int, default=15, help="Số epochs tối đa")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size khi encode")
    args = parser.parse_args()
    
    datasets_to_run = ["tofu", "mnli", "amazon"] if args.dataset == "all" else [args.dataset]
    for ds in datasets_to_run:
        run_pipeline(
            dataset=ds,
            model_name=args.model_name,
            data_dir=args.data_dir,
            work_dir=args.work_dir,
            epochs=args.epochs,
            batch_size=args.batch_size
        )


if __name__ == "__main__":
    main()
