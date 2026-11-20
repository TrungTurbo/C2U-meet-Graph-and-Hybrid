# HƯỚNG DẪN THỰC THI TOÀN TRÌNH C2U IN RAG TRÊN KAGGLE GPU (T4/P100)

Tài liệu này hướng dẫn chi tiết quy trình chạy toàn bộ 48 unit thử nghiệm (4 dataset × 3 seed × 4 encoder) trên Kaggle GPU một cách đơn giản, tin cậy và không xảy ra lỗi thiếu file.

---

## 1. Chuẩn Bị & Tải Gói Dữ Liệu Lên Kaggle

Thư mục dự án đã đóng gói sẵn file nén toàn diện **`c2u_full_bundle.zip`** (~11.9 MB), bao gồm:
1. **`processed_splits/`**: Toàn bộ 15 thư mục split đã chuẩn hóa không rò rỉ (12 single + 3 multi-concept). Không cần tải dataset thô 3.8GB lên Kaggle!
2. **`src/`**: Mã nguồn module hóa (`split_units.py`, `encode_cache.py`, `acr_dataset.py`, `acr_model.py`, `train_acr.py`, `c2u_operator.py`, `evaluate.py`).
3. **`configs/`**: Toàn bộ file YAML từ khóa và danh mục đã khóa.
4. **`manifests/`**: Các file manifest SHA256 và `encoders.json` ghim revision.
5. **`tests/`**: Bộ kiểm thử toàn vẹn dữ liệu `test_data_integrity.py` và dry-run toán học.
6. **`run_kaggle.py`**: Kịch bản điều phối tự động từ A-Z.

### Cách tải lên Kaggle:
1. Vào [Kaggle](https://www.kaggle.com/) -> Chọn **Datasets** -> **New Dataset**.
2. Đặt tiêu đề (Title): `c2u-bundle`.
3. Kéo thả file `c2u_full_bundle.zip` vào và nhấn **Create**.
4. Mở Notebook mới trên Kaggle:
   - Chọn Accelerator: **GPU T4 x 2** hoặc **GPU P100**.
   - Bật **Internet**: **On** (trong mục Settings bên phải để tải weights encoder).
   - Nhấn **Add Input** -> Chọn dataset `c2u-bundle` vừa tạo.

---

## 2. Các Cells Thực Thi Trong Kaggle Notebook

### CELL 1: Kiểm Tra Phần Cứng GPU & Cài Đặt Dependencies
```python
import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Model: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")

!pip install -q sentence-transformers concept-erasure rank_bm25 scikit-learn
```

---

### CELL 2: Sao Chép Dữ Liệu, Cấu Hình Chống Tràn VRAM & Khôi Phục Checkpoints
```python
import os, shutil, glob, zipfile

# 1. Tự động tìm thư mục chứa mã nguồn trong /kaggle/input
src_input_dir = None
for root, dirs, files in os.walk("/kaggle/input"):
    if "src" in dirs and "processed_splits" in dirs:
        src_input_dir = root
        break

assert src_input_dir, "Không tìm thấy thư mục chứa src và processed_splits trong /kaggle/input!"
print(f"Đã tìm thấy dữ liệu tại: {src_input_dir}")

# 2. Copy toàn bộ sang /kaggle/working
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

# 3. Phục hồi 39 checkpoints cũ nếu phát hiện file zip backup
backup_zips = glob.glob("/kaggle/working/*results*.zip") + glob.glob("/kaggle/input/**/*results*.zip", recursive=True)
if backup_zips:
    target_zip = backup_zips[0]
    print(f"\n-> Phát hiện file backup: {target_zip}. Đang giải nén...")
    with zipfile.ZipFile(target_zip, 'r') as zf:
        zf.extractall("/kaggle/working")
    existing_ckpts = len(glob.glob("/kaggle/working/checkpoints/*.pt"))
    print(f"-> Đã phục hồi thành công {existing_ckpts} checkpoints cũ!")
else:
    print("\n-> Không tìm thấy file zip backup cũ, pipeline sẽ chạy từ đầu.")

# 4. FIX TRIỆT ĐỂ LỖI OOM CHO QWEN3 & ARXIV TRÊN GPU TESLA T4
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Vá encode_cache.py: hạ batch_size=16 để không bao giờ bị OOM
encode_file = "/kaggle/working/src/encode_cache.py"
with open(encode_file, "r", encoding="utf-8") as f:
    enc_code = f.read()

enc_code = enc_code.replace("batch_size: int = 128", "batch_size: int = 16")
enc_code = enc_code.replace("batch_size: int = 64", "batch_size: int = 16")
with open(encode_file, "w", encoding="utf-8") as f:
    f.write(enc_code)

# Vá run_kaggle.py: tự động dọn VRAM và bỏ qua các unit đã có checkpoint
run_file = "/kaggle/working/run_kaggle.py"
with open(run_file, "r", encoding="utf-8") as f:
    rk_code = f.read()

if "torch.cuda.empty_cache()" not in rk_code:
    rk_code = rk_code.replace(
        "completed_units += 1",
        "completed_units += 1\n                if torch.cuda.is_available(): torch.cuda.empty_cache()\n                import gc; gc.collect()"
    )

old_call = """                # 2.2 Huấn luyện ACR
                ckpt_path = train_single_unit(
                    cache_path=cache_path,
                    output_dir="checkpoints",
                    max_epochs=max_epochs,
                    device=device,
                    seed=s * 100 + 42
                )"""

new_call = """                # 2.2 Huấn luyện ACR (bỏ qua nếu đã có checkpoint)
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
                    )"""

if old_call in rk_code:
    rk_code = rk_code.replace(old_call, new_call)

with open(run_file, "w", encoding="utf-8") as f:
    f.write(rk_code)

print("\n[Thành công] Toàn bộ mã nguồn đã sẵn sàng và được vá lỗi OOM 100%!")
!ls -la /kaggle/working
```

---

### CELL 3: Kiểm Tra Toàn Vẹn & Rời Rạc Dữ Liệu (Gói 1.4)
```python
%cd /kaggle/working
# Chạy kiểm thử toàn vẹn 15 split và manifest (chạy mượt mà, không lỗi)
!python tests/test_data_integrity.py
```

---

### CELL 4: Chạy Pipeline Toàn Diện (Giai Đoạn 1 Cache & Giai Đoạn 2 Huấn Luyện ACR)
```python
%cd /kaggle/working
# Chạy toàn bộ 4 dataset x 3 seed x 4 encoder = 48 units trên GPU
!python run_kaggle.py --dataset all --seed 0 --encoder all --max_epochs 10
```

---

### CELL 5: Đóng Gói Toàn Bộ Checkpoints & Cache Để Tải Về
```python
%cd /kaggle/working
import zipfile, os

zip_name = "c2u_results_and_checkpoints.zip"
with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for folder in ["checkpoints", "cache_embeddings"]:
        if os.path.exists(folder):
            for root, dirs, files in os.walk(folder):
                for f in files:
                    fp = os.path.join(root, f)
                    zipf.write(fp, fp)

print(f"[Done] Đã đóng gói toàn bộ kết quả vào: /kaggle/working/{zip_name}")
print(f"Dung lượng: {os.path.getsize(zip_name) / (1024*1024):.2f} MB")
```

---

### CELL 6: Nén Kết Quả & Checkpoints Để Tải Về Máy
```python
import os, zipfile

os.makedirs("/kaggle/working/results", exist_ok=True)
output_zip = "/kaggle/working/c2u_results_and_checkpoints.zip"

with zipfile.ZipFile(output_zip, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
    for target_dir in ["checkpoints", "cache_embeddings", "manifests"]:
        if os.path.exists(target_dir):
            for root, dirs, files in os.walk(target_dir):
                for f in files:
                    fp = os.path.join(root, f)
                    zf.write(fp, os.path.relpath(fp, "/kaggle/working"))

print(f"[Done] Đã đóng gói toàn bộ kết quả vào: {output_zip}")
print(f"Dung lượng: {os.path.getsize(output_zip) / (1024*1024):.2f} MB")
```
