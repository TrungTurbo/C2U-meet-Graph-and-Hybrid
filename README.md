# C2U for Hybrid RAG & Graph RAG

> **Kho lưu trữ mã nguồn dự án nghiên cứu áp dụng kỹ thuật C2U (Concept-to-Unit Machine Unlearning) cho các kiến trúc Hybrid RAG và Graph RAG.**

---

## 📖 1. Giới Thiệu Dự Án

Dự án này hiện thực hóa và đánh giá giải pháp **C2U (Concept-to-Unit)** nhằm giải quyết bài toán **Machine Unlearning** (loại bỏ tri thức mục tiêu/nhạy cảm) trong hai kiến trúc RAG hiện đại mà không cần tốn kém chi phí tái huấn luyện mô hình ngôn ngữ lớn (LLM):

1. **Hybrid RAG**: Tích hợp unlearning giữa tầng biểu diễn vector ngữ nghĩa dày đặc (*Dense Semantic Embeddings* qua Sentence-Transformers) và tầng truy xuất từ khóa thưa (*Sparse Lexical Retrieval* qua **BM25 / `rank-bm25`**). Toán tử C2U triệt tiêu độ kích hoạt của khái niệm mục tiêu mà không làm mất độ chính xác khớp từ khóa.
2. **Graph RAG**: Kết hợp không gian nhúng với mạng lưới quan hệ thực thể (*Knowledge Graph* qua **`networkx`**), cho phép triệt tiêu khái niệm cục bộ mà vẫn bảo tồn tính liên kết và độ hữu ích của đồ thị tri thức.

### Các Cơ Chế Cốt Lõi:
* **Từ điển Khái niệm Nguyên tử (ACR)**: Ma trận $A \in \mathbb{R}^{N \times d}$ tối ưu hóa đồng thời 3 hàm mất mát (Cosine Triplet, Cosine Alignment và Entropy Penalty).
* **Toán tử C2U Động**: Cổng kích hoạt hiệu chuẩn $s(q)$ và cập nhật truy vấn trong vùng tin cậy (*Trust-Region Update*), khống chế góc xoay $\theta \le 0.2\text{ rad}$ ($\langle q', q \rangle \ge 0.98$) giúp triệt tiêu khái niệm mục tiêu mà không làm suy giảm tri thức thông thường (*Control Utility*).
* **Độc lập dữ liệu tuyệt đối (Zero-Leakage)**: 15 split chuẩn hóa độc lập theo `row_id` và `text_hash` trên 4 bộ dữ liệu: **TOFU**, **Amazon**, **MNLI**, và **ArXiv**.

---

## 📁 2. Cấu Trúc Thư Mục

```text
Project001/
├── src/                          # Mã nguồn cốt lõi (ACR model, C2U operator, evaluate, dataloader)
├── configs/                      # Cấu hình từ khóa và danh mục mục tiêu (.yaml)
├── manifests/                    # Checksum SHA256 từng file dữ liệu và encoder revisions
├── processed_splits/             # 15 bộ dữ liệu đã chia tách chuẩn hóa sẵn (12 single + 3 multi)
├── tests/                        # Kiểm thử toàn vẹn dữ liệu và dry-run mô phỏng toán tử
├── cache_embeddings/             # Cache vector nhúng (.pt)
├── checkpoints/                  # Checkpoints trọng số mô hình ACR (.pt)
├── run_kaggle.py                 # Kịch bản tự động hóa toàn trình (Auto-pipeline 48 units)
├── README_KAGGLE.md              # Hướng dẫn chi tiết chạy trên Kaggle GPU (Tesla T4 / P100)
├── c2u_full_bundle.zip           # Gói nén chứa toàn bộ code và splits sẵn sàng cho Cloud
├── requirements.txt              # Danh sách thư viện phụ thuộc
└── README.md                     # Tài liệu giới thiệu & hướng dẫn sử dụng này
```

---

## 🛠️ 3. Hướng Dẫn Cài Đặt Môi Trường (`.venv`) Chi Tiết

Khuyến nghị sử dụng **Python $\ge 3.10$** (3.10, 3.11 hoặc 3.12). Thực hiện tuần tự các bước dưới đây:

### Bước 3.1: Tạo môi trường ảo `.venv`
Mở Terminal tại thư mục gốc của dự án:

* **Linux / macOS**:
  ```bash
  python3 -m venv .venv
  ```
* **Windows**:
  ```powershell
  python -m venv .venv
  ```

---

### Bước 3.2: Kích hoạt môi trường ảo

* **Linux / macOS (Bash / Zsh)**:
  ```bash
  source .venv/bin/activate
  ```
* **Windows (Command Prompt - cmd)**:
  ```cmd
  .venv\Scripts\activate.bat
  ```
* **Windows (PowerShell)**:
  ```powershell
  .venv\Scripts\Activate.ps1
  ```
  *(Nếu Windows PowerShell báo lỗi script execution policy, chạy lệnh sau rồi kích hoạt lại)*:
  ```powershell
  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
  .venv\Scripts\Activate.ps1
  ```

> Khi kích hoạt thành công, terminal sẽ có tiền tố `(.venv)` ở đầu dòng lệnh.

---

### Bước 3.3: Nâng cấp pip & Cài đặt PyTorch phù hợp

```bash
# Nâng cấp công cụ cài đặt
python -m pip install --upgrade pip setuptools wheel
```

* **Nếu máy có GPU NVIDIA (CUDA 12.1 - Khuyến nghị)**:
  ```bash
  pip install torch --index-url https://download.pytorch.org/whl/cu121
  ```
* **Nếu chỉ dùng CPU**:
  ```bash
  pip install torch --index-url https://download.pytorch.org/whl/cpu
  ```

---

### Bước 3.4: Cài đặt toàn bộ thư viện phụ thuộc
```bash
pip install -r requirements.txt
```

Các thư viện chính được cài đặt bao gồm:
* `torch`: Huấn luyện mạng nơ-ron và tính toán tensor.
* `sentence-transformers`: Tải và suy luận 4 pretrained encoders (`MiniLM`, `BGE`, `E5`, `Qwen3`).
* `concept-erasure`: Thư viện lõi hỗ trợ các toán tử can thiệp và loại bỏ biểu diễn khái niệm.
* `rank-bm25`: Truy xuất từ khóa thưa phục vụ module **Hybrid RAG**.
* `networkx`: Xử lý cấu trúc mạng lưới tri thức phục vụ module **Graph RAG**.
* `scikit-learn`, `pandas`, `numpy`, `scipy`: Xử lý số liệu và đánh giá thống kê.

---

### Bước 3.5: Kiểm tra xác minh môi trường
Chạy dòng lệnh sau để đảm bảo toàn bộ thư viện đã nhận diện chính xác:

```bash
python -c "import torch, sentence_transformers, concept_erasure, rank_bm25, networkx; print('--> MÔI TRƯỜNG ĐÃ SẴN SÀNG! CUDA Available:', torch.cuda.is_available())"
```

---

## 🚀 4. Hướng Dẫn Vận Hành Từng Bước

### Bước 4.1: Kiểm thử mô phỏng (Dry-run & Data Integrity)
```bash
# Kiểm thử toán tử C2U, gradient và cập nhật vùng tin cậy
python tests/test_dry_run.py

# Kiểm tra tính toàn vẹn và rời rạc dữ liệu của 15 split
python tests/test_data_integrity.py
```

### Bước 4.2: Trích xuất vector đóng băng ra Cache
```bash
python src/encode_cache.py \
    --split_dir processed_splits/tofu_seed1 \
    --encoder all-MiniLM-L6-v2 \
    --output_dir cache_embeddings \
    --batch_size 64
```

### Bước 4.3: Huấn luyện Từ điển Nguyên tử ACR
```bash
python src/train_acr.py \
    --cache_path cache_embeddings/tofu_seed1_minilm.pt \
    --output_dir checkpoints \
    --max_epochs 10 \
    --batch_size 32 \
    --lr 0.001
```

### Bước 4.4: Đánh giá Top-5 Retrieval & Unlearning Effect
```bash
python src/evaluate.py \
    --cache_dir cache_embeddings \
    --checkpoint checkpoints/acr_tofu_seed1_minilm.pt \
    --dataset_name tofu \
    --model_name all-MiniLM-L6-v2 \
    --k 5 \
    --output_csv results/eval_results.csv
```

### Bước 4.5: Chạy tự động toàn trình (Auto-pipeline)
```bash
python run_kaggle.py --datasets tofu amazon mnli arxiv --encoders all-MiniLM-L6-v2 --seeds 1 2 3
```

---

## 📊 5. Các Chỉ Số Đánh Giá Cốt Lõi

| Chỉ số | Ký hiệu | Ý nghĩa | Mục tiêu |
| :--- | :---: | :--- | :---: |
| **Target Hit@5 Drop** | $\Delta \text{Hit@5}$ | Mức giảm tỷ lệ tài liệu nhạy cảm mục tiêu lọt vào Top-5 khi truy vấn liên quan. | **Càng cao càng tốt** ($\uparrow$) |
| **Control Utility Drop** | $\text{CN}$ | Mức độ suy giảm độ chính xác truy xuất trên các truy vấn bình thường. | **$\le 0$** (bảo toàn tri thức gốc) |
| **Target Exposure Change** | $\text{TE}$ | Mức độ rò rỉ tài liệu mục tiêu vào các truy vấn bình thường. | **$\le 0$** ($\downarrow$) |

---

## ☁️ 6. Chạy Trên Kaggle GPU (T4 / P100)

Dự án đã đóng gói sẵn file **`c2u_full_bundle.zip`** và tối ưu hóa bộ nhớ chống tràn VRAM khi chạy các mô hình lớn trên GPU Tesla T4/P100. Vui lòng xem hướng dẫn chi tiết tại [README_KAGGLE.md](file:///home/marisnguyen/Documents/Project001/README_KAGGLE.md).
