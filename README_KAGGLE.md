# HƯỚNG DẪN THỰC THI TOÀN TRÌNH C2U IN RAG TRÊN KAGGLE GPU (T4/P100)

Tài liệu này cung cấp quy trình chuẩn 6 bước (6 Cells) để thực nghiệm trên Kaggle Notebook không xảy ra bất kỳ lỗi thiếu file, thiếu thư viện hay sai đường dẫn nào.

---

## 1. Chuẩn Bị & Tải Gói Dự Án Lên Kaggle

Thư mục dự án đã tạo sẵn gói nén toàn diện **`c2u_full_bundle.zip`** (**1.6 MB**), chứa đầy đủ:
- Toàn bộ mã nguồn: `src/` (`acr_model.py`, `c2u_operator.py`, `encode_cache.py`, `train_acr.py`, `evaluate.py`, `split_units.py`)
- Script điều phối: `run_kaggle_pipeline.py`
- Script kiểm thử dry-run: `test_dry_run.py`
- Toàn bộ dữ liệu splits chuẩn: `processed_splits/` (`tofu`, `mnli`, `amazon`)

### Các bước tải lên Kaggle:
1. Đăng nhập [Kaggle](https://www.kaggle.com/) -> Chọn **Datasets** -> **New Dataset**.
2. Đặt tên dataset: **`c2u-bundle`**.
3. Kéo thả file `c2u_full_bundle.zip` vào và bấm **Create**.
4. Mở **New Notebook**, chọn **Accelerator**: **GPU T4 x2** (hoặc GPU P100), chọn **Add Data** -> Chọn dataset **`c2u-bundle`** vừa tạo.
   *(Nếu muốn chạy cả ArXiv, chọn Add Data thêm dataset `Cornell-University/arxiv`).*

> [!TIP]
> **Cách nhanh nhất:** Bạn có thể vào menu Kaggle Notebook: **File** -> **Import Notebook** và tải file **`c2u_kaggle_complete.ipynb`** có sẵn trong thư mục dự án lên, toàn bộ 6 cell dưới đây sẽ xuất hiện tự động!

---

## 2. Chi Tiết 6 Cells Chạy Trong Kaggle Notebook

### CELL 1: Kiểm Tra Phần Cứng GPU & Cài Đặt Dependencies
Đảm bảo kernel nhận đúng GPU và cài các thư viện phục vụ SentenceTransformers & vẽ biểu đồ:
```python
import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Model: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")

!pip install -q sentence-transformers tabulate seaborn
```

---

### CELL 2: Giải Nén Mã Nguồn & Dữ Liệu Vào Môi Trường Thực Thi
Giải nén trọn gói cả code và splits từ `/kaggle/input/c2u-bundle` vào `/kaggle/working`:
```python
import os, zipfile

bundle_paths = [
    "/kaggle/input/c2u-bundle/c2u_full_bundle.zip",
    "/kaggle/input/c2u-splits/c2u_full_bundle.zip",
    "/kaggle/input/c2u-splits/c2u_splits.zip"
]

extracted = False
for path in bundle_paths:
    if os.path.exists(path):
        print(f"[Setup] Đang giải nén gói: {path}...")
        with zipfile.ZipFile(path, "r") as zf:
            zf.extractall("/kaggle/working")
        extracted = True
        break

if not extracted:
    # Nếu tải lên dạng thư mục không nén
    !cp -r /kaggle/input/c2u-bundle/* /kaggle/working/ 2>/dev/null || true

!ls -lh /kaggle/working
```

---

### CELL 3: Chạy Kiểm Thử Sanity Check (Dry-run) Trực Tiếp Trên GPU
Kiểm tra trước toàn bộ phép toán đại số tuyến tính (`lstsq`, `r`, góc xoay Trust-Region $\le 0.2\text{ rad}$, gradient ACR) trong vòng 5 giây trước khi chạy tốn thời gian:
```bash
!python /kaggle/working/test_dry_run.py
```

---

### CELL 4: Chạy Pipeline Thực Nghiệm C2U (Huấn Luyện & Đánh Giá)
Bạn có thể dễ dàng thay đổi cấu hình dataset hoặc encoder:
- **Datasets:** `"tofu"`, `"mnli"`, `"amazon"`, `"arxiv"`, hoặc `"all"`.
- **Encoders:**
  - `sentence-transformers/all-MiniLM-L6-v2` ($d=384, N=768$)
  - `BAAI/bge-base-en-v1.5` ($d=768, N=1536$)
  - `intfloat/e5-base-v2` ($d=768, N=1536$)

```python
DATASET = "tofu"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EPOCHS = 15
BATCH_SIZE = 128

!python /kaggle/working/run_kaggle_pipeline.py \
    --dataset {DATASET} \
    --model_name {MODEL_NAME} \
    --work_dir /kaggle/working \
    --epochs {EPOCHS} \
    --batch_size {BATCH_SIZE}
```

> **Ghi chú về ArXiv:** Nếu đặt `DATASET = "arxiv"`, pipeline sẽ tự động cấu hình $N=64$ theo đúng đặc tả của bài báo.

---

### CELL 5: Trực Quan Hóa Kết Quả Thực Nghiệm Qua Biểu Đồ
Hiển thị bảng dữ liệu Pandas và tự động vẽ 3 biểu đồ trực quan đối sánh các chỉ số:
```python
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

csv_path = "/kaggle/working/c2u_final_results.csv"
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    print("=== BẢNG KẾT QUẢ C2U RETRIEVAL ===")
    display(df)

    # Thiết lập giao diện biểu đồ
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    sns.set_theme(style="whitegrid")

    # 1. Target Hit@5 Drop (Càng cao càng unlearn tốt)
    sns.barplot(data=df[df["Method"] == "C2U"], x="Dataset", y="Target_Hit@5_Drop", hue="Model", ax=axes[0], palette="Blues_d")
    axes[0].set_title("Target Hit@5 Drop (Càng cao càng tốt)", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Mức giảm tỷ lệ Target (%)")

    # 2. Control Utility Drop (Mục tiêu <= 0, bảo toàn tiện ích)
    sns.barplot(data=df[df["Method"] == "C2U"], x="Dataset", y="Control_Utility_Drop", hue="Model", ax=axes[1], palette="Greens_d")
    axes[1].axhline(0, color="red", linestyle="--", alpha=0.7)
    axes[1].set_title("Control Utility Drop (Mục tiêu <= 0)", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Mức suy giảm tiện ích Control")

    # 3. Target Exposure Change (Mục tiêu <= 0)
    sns.barplot(data=df[df["Method"] == "C2U"], x="Dataset", y="TE_Change", hue="Model", ax=axes[2], palette="Oranges_d")
    axes[2].axhline(0, color="red", linestyle="--", alpha=0.7)
    axes[2].set_title("Target Exposure Change in Control (<= 0)", fontsize=12, fontweight="bold")
    axes[2].set_ylabel("Thay đổi độ phơi bày")

    plt.tight_layout()
    plt.savefig("/kaggle/working/c2u_evaluation_plot.png", dpi=300)
    plt.show()
    print("[Plot] Đã lưu biểu đồ vào /kaggle/working/c2u_evaluation_plot.png")
else:
    print("Chưa tìm thấy file c2u_final_results.csv. Hãy chạy Cell 4 trước!")
```

---

### CELL 6: Tải Kết Quả CSV & Biểu Đồ Về Máy
Cung cấp link tải trực tiếp ngay trong giao diện Notebook:
```python
from IPython.display import FileLink, display

print("Click vào liên kết dưới đây để tải kết quả về máy tính:")
display(FileLink("c2u_final_results.csv"))
if os.path.exists("c2u_evaluation_plot.png"):
    display(FileLink("c2u_evaluation_plot.png"))
```
