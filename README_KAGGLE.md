# HƯỚNG DẪN THỰC THI C2U IN RAG TRÊN KAGGLE (GPU T4/P100)

## 1. Tổng quan các bộ dữ liệu thực nghiệm (Đã loại bỏ IMDb)
Theo Mục B.3 và Bảng 9 của bài báo gốc, tập IMDb bị suy biến mẫu câu nên cho mức giảm Hit@5 bằng 0.0000 trên 3/4 encoder. Dự án tập trung vào 4 bộ chuẩn:
- **TOFU:** Lọc chủ đề nhạy cảm (Family, Location, Birth) làm Target Concept, giữ Author ID làm Control Attribute.
- **MNLI:** Thể loại du lịch (`travel`) làm Target Concept, các thể loại khác làm Control.
- **Amazon Polarity:** Đánh giá tiêu cực (`sentiment=0`) làm Target Concept, tích cực làm Control.
- **ArXiv:** Sử dụng trực tiếp dataset `Cornell-University/arxiv` trên Kaggle với $N=64$.

---

## 2. Chuẩn bị file tải lên Kaggle

File `c2u_splits.zip` (1.6 MB) đã được tạo sẵn tại thư mục gốc của dự án, chứa đầy đủ các split không chồng lấn cho TOFU, MNLI, Amazon.
1. Đăng nhập [Kaggle](https://www.kaggle.com/) -> **Datasets** -> **New Dataset**.
2. Đặt tên dataset: `c2u-splits`.
3. Kéo thả file `c2u_splits.zip` vào và nhấn **Create**.
4. Mở một **New Notebook** trên Kaggle, bật **GPU T4 x2** (hoặc P100), chọn **Add Data** -> Chọn dataset `c2u-splits`.

---

## 3. Các lệnh thực thi trong Kaggle Notebook

### Cell 1: Kiểm tra môi trường GPU
```python
import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

### Cell 2: Chuẩn bị mã nguồn & dữ liệu
```bash
# Giải nén data splits vào /kaggle/working/processed_splits
!mkdir -p /kaggle/working/processed_splits
!unzip -q /kaggle/input/c2u-splits/c2u_splits.zip -d /kaggle/working/processed_splits
```

### Cell 3: Chạy Thực nghiệm 1 lệnh duy nhất (`run_kaggle_pipeline.py`)

#### A. Chạy TOFU (Unlearn thông tin cá nhân nhạy cảm Family/Location)
```bash
!python run_kaggle_pipeline.py \
    --dataset tofu \
    --model_name sentence-transformers/all-MiniLM-L6-v2 \
    --data_dir /kaggle/working/processed_splits/tofu \
    --work_dir /kaggle/working \
    --epochs 15 \
    --batch_size 128
```

#### B. Mở rộng sang BGE ($d=768, N=1536$)
```bash
!python run_kaggle_pipeline.py \
    --dataset tofu \
    --model_name BAAI/bge-base-en-v1.5 \
    --data_dir /kaggle/working/processed_splits/tofu \
    --work_dir /kaggle/working \
    --epochs 15 \
    --batch_size 64
```

#### C. Mở rộng sang E5 ($d=768, N=1536$)
```bash
!python run_kaggle_pipeline.py \
    --dataset tofu \
    --model_name intfloat/e5-base-v2 \
    --data_dir /kaggle/working/processed_splits/tofu \
    --work_dir /kaggle/working \
    --epochs 15 \
    --batch_size 64
```

#### D. Chạy toàn bộ các dataset (TOFU, MNLI, Amazon)
```bash
!python run_kaggle_pipeline.py \
    --dataset all \
    --model_name sentence-transformers/all-MiniLM-L6-v2 \
    --work_dir /kaggle/working
```

---

## 4. Xem và tải kết quả
File kết quả tổng hợp được tự động lưu vào `/kaggle/working/c2u_final_results.csv`.
Đọc và hiển thị ngay trong Notebook:
```python
import pandas as pd
df = pd.read_csv('/kaggle/working/c2u_final_results.csv')
display(df)
```

Bảng báo cáo 3 chỉ số chính xác theo bài báo:
- **Target Hit@5 Drop ($\uparrow$):** Mức giảm tỷ lệ tài liệu target lọt vào Top-5 của `test_target`.
- **Control Utility Drop / CN ($\le 0$):** Mức suy giảm độ chính xác trên `test_control`.
- **Target Exposure / TE ($\le 0$):** Mức thay đổi tỷ lệ tài liệu target vô tình lọt vào Top-5 của `test_control`.
