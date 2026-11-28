# HƯỚNG DẪN THỰC THI TOÀN TRÌNH C2U IN RAG TRÊN KAGGLE GPU (T4/P100)

Tài liệu này hướng dẫn quy trình chạy toàn bộ **48 unit đơn khái niệm + 12 unit ArXiv đa khái niệm ($K=4$)** trên Kaggle GPU.

---

## 1. Chuẩn Bị & Tải Gói Dữ Liệu Lên Kaggle

File **`c2u_full_bundle.zip`** đóng gói sẵn:
1. **`processed_splits/`**: 15 thư mục split đã chuẩn hóa không rò rỉ (12 single + 3 `arxiv_multi`). Không cần tải dataset thô 3.8GB lên Kaggle.
2. **`src/`**: Mã nguồn GĐ1–GĐ4 v2 (`split_units.py`, `encode_cache.py`, `acr_dataset.py`, `acr_model.py`, `train_acr.py`, `c2u_operator.py`, `eval_utils.py`, `evaluate_dense.py`, `evaluate_transforms.py`, `evaluate_multi.py`, `ablation_acr.py`).
3. **`configs/`**: Các file YAML từ khóa, danh mục đã khóa và `transforms.yaml`.
4. **`manifests/`**: SHA256 từng file split và `encoders.json` ghim revision.
5. **`tests/`**: `test_data_integrity.py` và `test_dry_run.py`.
6. **`run_kaggle.py`**: Kịch bản điều phối GĐ1 cache & GĐ2 train ACR.

### Cách tải lên Kaggle:
1. Vào [Kaggle](https://www.kaggle.com/) -> **Datasets** -> **New Dataset**.
2. Đặt Title: `c2u-bundle`, kéo thả `c2u_full_bundle.zip` -> **Create**.
3. Mở Notebook mới:
   - Accelerator: **GPU T4 x 2** hoặc **GPU P100**.
   - Bật **Internet: On** (để tải weights encoder).
   - **Add Input** -> chọn dataset `c2u-bundle`.

---

## 2. Các Cells Thực Thi Trong Kaggle Notebook

### CELL 1: Kiểm Tra GPU & Cài Thư Viện
```python
import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Model: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")

!pip install -q sentence-transformers concept-erasure rank_bm25 scikit-learn
```

### CELL 2: Sao Chép Mã Nguồn & Phục Hồi Backup (Nếu Có)
```python
import os, shutil, glob, zipfile

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

src_input_dir = None
for root, dirs, files in os.walk("/kaggle/input"):
    if "src" in dirs and "processed_splits" in dirs:
        src_input_dir = root
        break

assert src_input_dir, "Không tìm thấy thư mục chứa src và processed_splits trong /kaggle/input!"
print(f"Đã tìm thấy dữ liệu tại: {src_input_dir}")

items_to_copy = ["src", "configs", "manifests", "processed_splits", "tests", "run_kaggle.py", "requirements.txt"]
for item in items_to_copy:
    src_path = os.path.join(src_input_dir, item)
    dst_path = os.path.join("/kaggle/working", item)
    if os.path.exists(src_path):
        if os.path.isdir(src_path):
            shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
        else:
            shutil.copy2(src_path, dst_path)
        print(f"  -> Đã copy {item}")

os.makedirs("/kaggle/working/cache_embeddings", exist_ok=True)
os.makedirs("/kaggle/working/checkpoints", exist_ok=True)
os.makedirs("/kaggle/working/results", exist_ok=True)

backup_zips = glob.glob("/kaggle/working/*results*.zip") + glob.glob("/kaggle/input/**/*results*.zip", recursive=True)
if backup_zips:
    target_zip = backup_zips[0]
    print(f"\n-> Phát hiện file backup: {target_zip}. Đang giải nén...")
    with zipfile.ZipFile(target_zip, 'r') as zf:
        zf.extractall("/kaggle/working")
    print(f"-> Đã phục hồi {len(glob.glob('/kaggle/working/checkpoints/*.pt'))} checkpoints.")
```

*(Lưu ý: nếu trong backup có sẵn file `*_qwen3.pt` tạo từ phiên bản cũ trước khi vá double-prompt, hãy xóa riêng các file `*_qwen3.pt` để `encode_cache.py` tạo lại đúng chuẩn.)*

### CELL 3: Kiểm Tra Toàn Vẹn Dữ Liệu (Gói 1.4)
```python
%cd /kaggle/working
!python tests/test_data_integrity.py
```

### CELL 4: Chạy GĐ1 Cache & GĐ2 Huấn Luyện ACR (60 Units)
```python
%cd /kaggle/working
# Chạy 4 dataset đơn (48 units) + arxiv_multi (12 units) = 60 units
# Thêm --repro_epoch1 nếu muốn lưu thêm checkpoint mốc tái lập 1 epoch (_ep1.pt)
!python run_kaggle.py --dataset all --seed 0 --encoder all --max_epochs 10
```

### CELL 5: Chạy Đánh Giá GĐ4 v2 (Chỉ Trên `selection` & `confirm`, Khóa `test`)
```python
%cd /kaggle/working
!python src/evaluate_dense.py --datasets tofu amazon mnli arxiv --seeds 1 2 3 --encoders minilm bge e5 qwen3 --stage both --gate_quantile_space auto
!python src/evaluate_transforms.py --datasets tofu amazon mnli arxiv --seeds 1 2 3 --encoders minilm bge e5 qwen3 --stage confirm --gate_quantile_space query
!python src/evaluate_multi.py --seeds 1 2 3 --encoders minilm bge e5 qwen3 --stage confirm --gate_quantile_space query
!python src/ablation_acr.py --datasets tofu amazon mnli arxiv --seeds 1 2 3 --encoders minilm bge e5 qwen3 --gate_quantile_space query
```

### CELL 6: Đóng Gói Kết Quả, Checkpoints & Cache Để Tải Về
```python
%cd /kaggle/working
import os, zipfile

output_zip = "/kaggle/working/c2u_results_and_checkpoints.zip"
with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for target_dir in ["checkpoints", "cache_embeddings", "results", "configs", "manifests"]:
        if os.path.exists(target_dir):
            for root, dirs, files in os.walk(target_dir):
                for f in files:
                    fp = os.path.join(root, f)
                    zf.write(fp, os.path.relpath(fp, "/kaggle/working"))

print(f"[Done] Đã đóng gói vào: {output_zip} ({os.path.getsize(output_zip) / (1024*1024):.2f} MB)")
```
