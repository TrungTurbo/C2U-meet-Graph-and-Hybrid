# C2U in RAG: Concept-to-Unit Machine Unlearning & Retrieval

Dự án nghiên cứu và hiện thực hóa phương pháp **C2U (Concept-to-Unit)** phục vụ bài toán **Machine Unlearning trong Hệ thống Sinh Tăng cường Truy xuất (RAG)**. 

Phương pháp sử dụng **Từ điển Khái niệm Nguyên tử (ACR - Atomic Concept Representation)** kết hợp với toán tử chiếu trực giao có giới hạn vùng tin cậy (**Trust-Region Update**) nhằm loại bỏ thông tin nhạy cảm/mục tiêu (Target Concept) khỏi biểu diễn truy vấn (Query Embeddings) mà không làm suy giảm độ chính xác trên các truy vấn bình thường (Control Utility).

---

## 🌟 Tính Năng Nổi Bật

- **Từ điển Khái niệm Nguyên tử (ACR)**: Huấn luyện offline ma trận từ điển $A \in \mathbb{R}^{N \times d}$ chuẩn hóa hàng, tối ưu hóa đồng thời 3 hàm mất mát (Cosine Triplet, Cosine Alignment và Entropy Penalty).
- **Toán tử C2U Động**:
  - Hướng triệt tiêu tiền điều kiện (*Preconditioned Suppression Direction*).
  - Cổng kích hoạt hiệu chuẩn (*Calibrated Gate* $s(q)$).
  - Phân rã bình phương tối thiểu (*Least Squares Decomposition*) và bù phần dư (*Residual Compensation*).
  - Cập nhật trong vùng tin cậy (*Trust-Region Update*) khống chế góc xoay truy vấn $\theta \le 0.2\text{ rad}$ ($\langle q', q \rangle \ge 0.98$).
- **Độc lập & Rời rạc Dữ liệu Tuyệt đối (Zero-Leakage Splits)**: 15 bộ dữ liệu đã tiền xử lý độc lập theo `row_id` và `text_hash` (4 tập dữ liệu $\times$ 3 seeds + 3 multi-concept).
- **Hỗ trợ 4 Bộ Mã Hóa Văn Bản (Frozen Encoders)**:
  - `all-MiniLM-L6-v2` (`sentence-transformers/all-MiniLM-L6-v2`)
  - `bge-base-en-v1.5` (`BAAI/bge-base-en-v1.5`)
  - `e5-base-v2` (`intfloat/e5-base-v2`)
  - `Qwen3-Embedding-0.6B` (`Qwen/Qwen3-Embedding-0.6B`)
- **Tương thích Phần cứng Linh hoạt**: Chạy mượt mà trên môi trường cục bộ (CPU/GPU) và tối ưu hóa sẵn cho Kaggle GPU (Tesla T4 / P100).

---

## 📁 Cấu Trúc Thư Mục Dự Án

```text
Project001/
├── src/                          # Mã nguồn module hóa cốt lõi
│   ├── acr_model.py              # Định nghĩa kiến trúc ACRModel và hàm loss ACRLoss
│   ├── acr_dataset.py            # DataLoader và tạo triplet cho huấn luyện ACR
│   ├── train_acr.py              # Script huấn luyện offline từ điển ACR
│   ├── c2u_operator.py           # Toán tử C2U 4 bước (Preconditioning, Gate, Rotation)
│   ├── encode_cache.py           # Mã hóa vector bằng frozen encoder ra cache .pt
│   ├── evaluate.py               # Đánh giá hiệu năng Top-5 Retrieval & Unlearning
│   └── split_units.py            # Quy trình tiền xử lý và chia tách các rổ dữ liệu
├── configs/                      # Cấu hình danh mục và từ khóa mục tiêu (YAML)
│   ├── amazon_keywords.yaml
│   ├── arxiv_keywords.yaml
│   ├── mnli_keywords.yaml
│   └── tofu_keywords.yaml
├── manifests/                    # Checksum SHA256 từng file dữ liệu và encoder revisions
│   ├── encoders.json
│   └── *.json
├── processed_splits/             # 15 bộ split dữ liệu chuẩn hóa sẵn (12 single + 3 multi)
│   ├── amazon_seed{1,2,3}/
│   ├── arxiv_seed{1,2,3}/
│   ├── arxiv_multi_seed{1,2,3}/
│   ├── mnli_seed{1,2,3}/
│   └── tofu_seed{1,2,3}/
├── tests/                        # Bộ kiểm thử toàn vẹn và toán học
│   ├── test_data_integrity.py    # Kiểm tra tính rời rạc và toàn vẹn của 15 split
│   └── test_dry_run.py           # Dry-run mô phỏng ACR loss, C2U rotation và Top-5 retrieval
├── cache_embeddings/             # Cache vector nhúng (.pt) được sinh tự động
├── checkpoints/                  # Trọng số ACR model (.pt) sau khi huấn luyện
├── run_kaggle.py                 # Kịch bản tự động hóa toàn trình (Auto-pipeline A-Z)
├── README_KAGGLE.md              # Sổ tay hướng dẫn chi tiết chạy trên Kaggle GPU
├── c2u_full_bundle.zip           # Gói zip nén sẵn toàn bộ code và splits để upload lên Cloud
├── requirements.txt              # Danh sách thư viện phụ thuộc
└── .gitignore                    # Bỏ qua môi trường ảo, cache và dữ liệu thô lớn
```

---

## ⚙️ Cài Đặt Môi Trường

### 1. Yêu cầu hệ thống
- Hệ điều hành: Linux / macOS / Windows
- Python: phiên bản $\ge 3.10$

### 2. Thiết lập Môi trường ảo (Virtual Environment)
```bash
# Tạo môi trường ảo
python3 -m venv .venv

# Kích hoạt môi trường ảo:
# Trên Linux / macOS:
source .venv/bin/activate
# Trên Windows (Command Prompt hoặc PowerShell):
# .venv\Scripts\activate
```

### 3. Cài đặt các thư viện phụ thuộc
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Lưu ý PyTorch với GPU CUDA**: Nếu bạn sử dụng GPU NVIDIA, hãy cài đặt bản PyTorch hỗ trợ CUDA phù hợp:
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cu121
> ```

---

## 🚀 Hướng Dẫn Sử Dụng Chi Tiết

### Bước 1: Kiểm thử Toàn vẹn & Chạy Thử Toán Tử (Dry-Run)
Trước khi chạy với dữ liệu lớn, chạy kiểm thử mô phỏng để đảm bảo môi trường tính toán hoạt động chính xác:

```bash
# Kiểm thử gradient backward, toán tử C2U và tính toán góc xoay:
python tests/test_dry_run.py

# Kiểm tra tính toàn vẹn và không rò rỉ dữ liệu giữa các rổ:
python tests/test_data_integrity.py
```

---

### Bước 2: Mã Hóa Dữ Liệu & Lưu Cache Embeddings
Mã hóa văn bản thành vector embeddings đóng băng bằng pretrained encoder (ví dụ với `tofu_seed1` và `all-MiniLM-L6-v2`):

```bash
python src/encode_cache.py \
    --split_dir processed_splits/tofu_seed1 \
    --encoder all-MiniLM-L6-v2 \
    --output_dir cache_embeddings \
    --batch_size 128
```
*Kết quả sẽ được lưu vào `cache_embeddings/tofu_seed1_all-MiniLM-L6-v2.pt` (và bản sao tên ngắn `tofu_seed1_minilm.pt`).*

---

### Bước 3: Huấn Luyện Từ Điển Khái Niệm Nguyên Tử (ACR)
Huấn luyện mô hình ACR offline từ cache embeddings vừa tạo:

```bash
python src/train_acr.py \
    --cache_path cache_embeddings/tofu_seed1_minilm.pt \
    --output_dir checkpoints \
    --max_epochs 10 \
    --batch_size 32 \
    --lr 0.001
```
*Mô hình tối ưu nhất sẽ được lưu tại `checkpoints/acr_tofu_seed1_minilm.pt`.*

---

### Bước 4: Đánh Giá Hiệu Năng Top-5 Retrieval & Unlearning
Đánh giá độ suy giảm truy xuất của tài liệu mục tiêu (Unlearning effect) và bảo toàn tri thức thông thường (Control utility):

```bash
python src/evaluate.py \
    --cache_dir cache_embeddings \
    --checkpoint checkpoints/acr_tofu_seed1_minilm.pt \
    --dataset_name tofu \
    --model_name all-MiniLM-L6-v2 \
    --k 5 \
    --output_csv results/eval_results.csv
```

---

### Bước 5: Chạy Tự Động Toàn Trình (Full Pipeline)
Để tự động chạy từ đầu đến cuối cho tất cả các đơn vị thử nghiệm (4 tập dữ liệu $\times$ 3 seeds $\times$ 4 encoders = 48 units):

```bash
# Chạy với thiết lập mặc định (tự động nhận diện GPU nếu có):
python run_kaggle.py

# Hoặc tùy biến chỉ định bộ dữ liệu và encoder mong muốn:
python run_kaggle.py \
    --datasets tofu amazon \
    --encoders all-MiniLM-L6-v2 bge-base-en-v1.5 \
    --seeds 1 2 \
    --max_epochs 10
```

---

## 📊 Các Chỉ Số Đánh Giá Cốt Lõi

Khi thực hiện lệnh đánh giá (`evaluate.py`), hệ thống sẽ tính toán các chỉ số sau:

| Chỉ số | Ký hiệu | Ý nghĩa | Mục tiêu |
| :--- | :---: | :--- | :---: |
| **Target Hit@5 Drop** | $\Delta \text{Hit@5}$ | Tỷ lệ giảm tài liệu Target bị truy xuất lọt vào Top-5 khi người dùng truy vấn về chủ đề nhạy cảm. | **Càng cao càng tốt** ($\uparrow$) |
| **Control Utility Drop** | $\text{CN}$ | Mức độ suy giảm độ chính xác truy xuất trên các truy vấn bình thường (Control queries). | **$\le 0$ hoặc xấp xỉ $0$** ($\downarrow$) |
| **Target Exposure Change** | $\text{TE}$ | Mức độ thay đổi tỷ lệ tài liệu Target vô tình lọt vào Top-5 của các truy vấn bình thường. | **$\le 0$** ($\downarrow$) |

---

## ☁️ Thực Thi Trên Kaggle GPU (T4 / P100)

Dự án đã được tối ưu hóa đặc biệt để chạy không gặp lỗi OOM (Out Of Memory) trên Kaggle:
1. Đã chuẩn bị sẵn file **`c2u_full_bundle.zip`** chứa toàn bộ `src/`, `configs/`, `manifests/`, `processed_splits/` và `run_kaggle.py`.
2. Tạo Kaggle Dataset từ `c2u_full_bundle.zip` và thêm vào Notebook.
3. Tham khảo các lệnh và cell mẫu chi tiết trong tài liệu: [README_KAGGLE.md](file:///home/marisnguyen/Documents/Project001/README_KAGGLE.md).

---

## 📌 Lưu Ý Quan Trọng Về Quản Lý Dữ Liệu & Git

- **Môi trường ảo (`.venv/`)**: Đã được thêm vào `.gitignore` để tránh đẩy hàng trăm MB thư viện thừa lên GitHub.
- **Tập dữ liệu thô lớn (`data/`)**: Thư mục `data/` chứa file `arxiv-metadata-oai-snapshot.json` (~3.8 GB) và `amazon_test_retrieval.jsonl` (~180 MB) vượt quá giới hạn 100 MB của GitHub. Vì vậy `data/` được giữ trong `.gitignore`. Bạn **không cần** các file thô này để tái lập thí nghiệm vì toàn bộ 15 split chuẩn hóa đã được tạo sẵn trong thư mục `processed_splits/`.
- **Checkpoint & Embedding Cache**: Các checkpoint mẫu được lưu trữ trong `checkpoints/` và `cache_embeddings/` đều có kích thước nhỏ gọn (< 15 MB) để tiện theo dõi và kiểm thử nhanh.

---

## 📜 Giấy Phép & Đóng Góp

Dự án phục vụ mục đích nghiên cứu học thuật về Machine Unlearning trong RAG. Mọi đóng góp hoặc câu hỏi vui lòng mở Issue hoặc tạo Pull Request trên GitHub repository.
