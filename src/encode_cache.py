#!/usr/bin/env python3
"""
src/encode_cache.py
Trích xuất text embeddings bằng bộ mã hóa đóng băng (Frozen Pretrained Encoders):
- 4 encoders ghim revision theo manifests/encoders.json:
    * all-MiniLM-L6-v2 (sentence-transformers/all-MiniLM-L6-v2)
    * bge-base-en-v1.5 (BAAI/bge-base-en-v1.5)
    * e5-base-v2 (intfloat/e5-base-v2)
    * Qwen3-Embedding-0.6B (Qwen/Qwen3-Embedding-0.6B)
- Quy ước prompt nhất quán:
    * all-MiniLM-L6-v2: query: (none), doc: (none)
    * bge-base-en-v1.5: query: "Represent this sentence for searching relevant passages: ", doc: (none)
    * e5-base-v2: query: "query: ", doc: "passage: "
    * Qwen3-Embedding-0.6B: query: prompt_name="query" / model card prompt, doc: (none)
- Profile, calibration-doc và index được mã hóa như TÀI LIỆU (documents).
- query_text của calibration, val, confirm, test được mã hóa như TRUY VẤN (queries).
- Chuẩn hóa L2, float32, torch.no_grad(), model.eval().
- Lưu file cache_embeddings/<dataset>_seed<k>_<encoder>.pt
"""

import os
import json
import argparse
from typing import List, Dict, Any, Optional
import torch
import torch.nn.functional as F

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


def load_jsonl(filepath: str) -> List[Dict[str, Any]]:
    """Đọc dữ liệu từ file JSON Lines."""
    items = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line.strip()))
    return items


ENCODER_ALIASES = {
    "minilm": "all-MiniLM-L6-v2",
    "all-minilm-l6-v2": "all-MiniLM-L6-v2",
    "sentence-transformers/all-minilm-l6-v2": "all-MiniLM-L6-v2",
    "bge": "bge-base-en-v1.5",
    "bge-base-en-v1.5": "bge-base-en-v1.5",
    "baai/bge-base-en-v1.5": "bge-base-en-v1.5",
    "e5": "e5-base-v2",
    "e5-base-v2": "e5-base-v2",
    "intfloat/e5-base-v2": "e5-base-v2",
    "qwen3": "Qwen3-Embedding-0.6B",
    "qwen3-embedding-0.6b": "Qwen3-Embedding-0.6B",
    "qwen/qwen3-embedding-0.6b": "Qwen3-Embedding-0.6B"
}


def get_encoder_config(encoder_key: str, manifest_path: str = "manifests/encoders.json") -> Dict[str, Any]:
    """Tải cấu hình revision và prompt của encoder."""
    canonical_key = ENCODER_ALIASES.get(encoder_key.lower(), encoder_key)
    with open(manifest_path, "r", encoding="utf-8") as f:
        encoders = json.load(f)
    if canonical_key not in encoders:
        raise ValueError(f"Encoder '{encoder_key}' không có trong {manifest_path}. Các lựa chọn: {list(encoders.keys())}")
    cfg = dict(encoders[canonical_key])
    cfg["canonical_key"] = canonical_key
    return cfg


def encode_batch(
    model: Any,
    texts: List[str],
    prefix: str = "",
    prompt_name: Optional[str] = None,
    batch_size: int = 128,
    device: str = "cpu"
) -> torch.Tensor:
    """Mã hóa văn bản, chuẩn hóa L2 và kiểm tra tính toàn vẹn."""
    if not texts:
        return torch.empty(0, dtype=torch.float32)

    # Thêm prefix prompt nếu có
    if prefix:
        prepared_texts = [prefix + t for t in texts]
    else:
        prepared_texts = texts

    model.eval()
    with torch.no_grad():
        kwargs = {
            "batch_size": batch_size,
            "show_progress_bar": False,
            "normalize_embeddings": True,
            "convert_to_tensor": True,
            "device": device
        }
        if prompt_name and hasattr(model, "prompts") and prompt_name in model.prompts:
            kwargs["prompt_name"] = prompt_name

        embs = model.encode(prepared_texts, **kwargs)
        if not isinstance(embs, torch.Tensor):
            embs = torch.tensor(embs, dtype=torch.float32)
        else:
            embs = embs.float()

        # Chuẩn hóa L2 bổ sung để đảm bảo sai số float < 1e-6
        embs = F.normalize(embs, p=2, dim=-1).cpu()

    # Kiểm tra nghiệm thu toán học
    assert not torch.isnan(embs).any(), "Phát hiện NaN trong embedding!"
    assert not torch.isinf(embs).any(), "Phát hiện Inf trong embedding!"
    norms = torch.norm(embs, p=2, dim=-1)
    diff = (norms - 1.0).abs().max().item()
    assert diff < 1e-4, f"Độ lệch chuẩn hóa L2 vượt ngưỡng: {diff}"

    return embs


def encode_unit(
    split_dir: str,
    encoder_key: str = "all-MiniLM-L6-v2",
    output_dir: str = "cache_embeddings",
    device: str = "cpu",
    batch_size: Optional[int] = None
) -> str:
    """Mã hóa một unit split hoàn chỉnh ra file .pt bit-identical."""
    os.makedirs(output_dir, exist_ok=True)
    cfg = get_encoder_config(encoder_key)
    model_name = cfg["model_name_or_path"]
    revision = cfg["revision"]
    q_prompt = cfg.get("query_prompt", "")
    d_prompt = cfg.get("doc_prompt", "")
    p_name = cfg.get("prompt_name", None)

    # Tự động điều chỉnh batch_size tối ưu cho GPU để tránh OOM (đặc biệt với Qwen3 và văn bản dài như arxiv)
    split_name = os.path.basename(split_dir.rstrip("/"))
    canonical_key = cfg.get("canonical_key", encoder_key)

    if batch_size is None:
        if "qwen3" in canonical_key.lower():
            batch_size = 16 if "arxiv" in split_name.lower() else 32
        elif "arxiv" in split_name.lower():
            batch_size = 64
        else:
            batch_size = 128

    print(f"\n[Encode] Đang tải encoder '{encoder_key}' ({model_name} @ {revision[:8]})... (batch_size={batch_size})")
    try:
        model = SentenceTransformer(model_name, revision=revision, device=device, local_files_only=True)
    except Exception:
        model = SentenceTransformer(model_name, revision=revision, device=device)
    out_filename = f"{split_name}_{canonical_key.replace('/', '_')}.pt"
    out_path = os.path.join(output_dir, out_filename)

    print(f"[Encode] Đang mã hóa các rổ dữ liệu cho '{split_name}'...")
    
    # 1. target_profile (tài liệu)
    profile_data = load_jsonl(os.path.join(split_dir, "target_profile.jsonl"))
    profile_embs = encode_batch(model, [x["document"] for x in profile_data], prefix=d_prompt, device=device, batch_size=batch_size)

    # 2. calibration_control (cả doc và query)
    calib_ctrl_data = load_jsonl(os.path.join(split_dir, "calibration_control.jsonl"))
    calib_doc_embs = encode_batch(model, [x["document"] for x in calib_ctrl_data], prefix=d_prompt, device=device, batch_size=batch_size)
    calib_query_embs = encode_batch(model, [x["query"] for x in calib_ctrl_data], prefix=q_prompt, prompt_name=p_name, device=device, batch_size=batch_size)

    # 3. calibration_labeled (cả query và doc cho các baseline phân loại)
    calib_lab_data = load_jsonl(os.path.join(split_dir, "calibration_labeled.jsonl"))
    calib_labeled_embs = encode_batch(model, [x["query"] for x in calib_lab_data], prefix=q_prompt, prompt_name=p_name, device=device, batch_size=batch_size)
    calib_labeled_doc_embs = encode_batch(model, [x["document"] for x in calib_lab_data], prefix=d_prompt, device=device, batch_size=batch_size)

    # 4. index_corpus (tài liệu)
    index_data = load_jsonl(os.path.join(split_dir, "index_corpus.jsonl"))
    index_embs = encode_batch(model, [x["document"] for x in index_data], prefix=d_prompt, device=device, batch_size=batch_size)

    # 5. val_queries (truy vấn)
    val_data = load_jsonl(os.path.join(split_dir, "val_queries.jsonl"))
    val_q_embs = encode_batch(model, [x["query"] for x in val_data], prefix=q_prompt, prompt_name=p_name, device=device, batch_size=batch_size)

    # 6. confirm_queries (truy vấn)
    conf_data = load_jsonl(os.path.join(split_dir, "confirm_queries.jsonl"))
    confirm_q_embs = encode_batch(model, [x["query"] for x in conf_data], prefix=q_prompt, prompt_name=p_name, device=device, batch_size=batch_size)

    # 7. test_queries (truy vấn)
    test_data = load_jsonl(os.path.join(split_dir, "test_queries.jsonl"))
    test_q_embs = encode_batch(model, [x["query"] for x in test_data], prefix=q_prompt, prompt_name=p_name, device=device, batch_size=batch_size)

    # 8. train_acr (query và doc cho ACR contrastive learning)
    train_data = load_jsonl(os.path.join(split_dir, "train_acr.jsonl"))
    train_q_embs = encode_batch(model, [x["query"] for x in train_data], prefix=q_prompt, prompt_name=p_name, device=device, batch_size=batch_size)
    train_d_embs = encode_batch(model, [x["document"] for x in train_data], prefix=d_prompt, device=device, batch_size=batch_size)

    # Đóng gói package tensor & metadata
    package = {
        # Embeddings
        "train_q_embs": train_q_embs,
        "train_d_embs": train_d_embs,
        "profile_embs": profile_embs,
        "calib_doc_embs": calib_doc_embs,
        "calib_query_embs": calib_query_embs,
        "calib_labeled_embs": calib_labeled_embs,
        "calib_labeled_doc_embs": calib_labeled_doc_embs,
        "index_embs": index_embs,
        "val_q_embs": val_q_embs,
        "confirm_q_embs": confirm_q_embs,
        "test_q_embs": test_q_embs,
        # Metadata / Labels / IDs
        "train_ids": [x["row_id"] for x in train_data],
        "train_labels": [x.get("label", "") for x in train_data],
        "train_is_target": torch.tensor([x.get("is_target", 0) for x in train_data], dtype=torch.long),

        "profile_ids": [x["row_id"] for x in profile_data],
        "profile_labels": [x.get("label", "") for x in profile_data],
        "profile_is_target": torch.tensor([x.get("is_target", 1) for x in profile_data], dtype=torch.long),

        "calib_doc_ids": [x["row_id"] for x in calib_ctrl_data],
        "calib_doc_labels": [x.get("label", "") for x in calib_ctrl_data],
        "calib_query_ids": [x["row_id"] for x in calib_ctrl_data],
        "calib_query_labels": [x.get("label", "") for x in calib_ctrl_data],

        "calib_labeled_ids": [x["row_id"] for x in calib_lab_data],
        "calib_labeled_labels": [x.get("label", "") for x in calib_lab_data],
        "calib_labeled_is_target": torch.tensor([x.get("is_target", 0) for x in calib_lab_data], dtype=torch.long),

        "index_ids": [x["row_id"] for x in index_data],
        "index_labels": [x.get("label", "") for x in index_data],
        "index_is_target": torch.tensor([x.get("is_target", 0) for x in index_data], dtype=torch.long),

        "val_q_ids": [x["row_id"] for x in val_data],
        "val_q_labels": [x.get("label", "") for x in val_data],
        "val_q_is_target": torch.tensor([x.get("is_target", 0) for x in val_data], dtype=torch.long),

        "confirm_q_ids": [x["row_id"] for x in conf_data],
        "confirm_q_labels": [x.get("label", "") for x in conf_data],
        "confirm_q_is_target": torch.tensor([x.get("is_target", 0) for x in conf_data], dtype=torch.long),

        "test_q_ids": [x["row_id"] for x in test_data],
        "test_q_labels": [x.get("label", "") for x in test_data],
        "test_q_is_target": torch.tensor([x.get("is_target", 0) for x in test_data], dtype=torch.long),

        "encoder_info": {
            "encoder_key": encoder_key,
            "model_name_or_path": model_name,
            "revision": revision,
            "dimension": int(index_embs.shape[1]),
            "query_prompt": q_prompt,
            "doc_prompt": d_prompt
        }
    }

    torch.save(package, out_path)
    print(f"  -> Đã lưu embedding cache thành công: {out_path} (d={package['encoder_info']['dimension']})")

    # Tạo thêm file với tên ngắn tương thích (ví dụ tofu_seed1_minilm.pt)
    short_names = {
        "all-MiniLM-L6-v2": "minilm",
        "bge-base-en-v1.5": "bge",
        "e5-base-v2": "e5",
        "Qwen3-Embedding-0.6B": "qwen3"
    }
    if canonical_key in short_names:
        short_fn = f"{split_name}_{short_names[canonical_key]}.pt"
        short_path = os.path.join(output_dir, short_fn)
        if short_path != out_path:
            import shutil
            shutil.copyfile(out_path, short_path)
            print(f"  -> Đã tạo bản sao tên ngắn: {short_path}")

    # Giải phóng VRAM GPU triệt để sau khi encode
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc
    gc.collect()

    return out_path


def main():
    parser = argparse.ArgumentParser(description="Mã hóa văn bản đóng băng và lưu cache embedding.")
    parser.add_argument("--split_dir", type=str, required=True, help="Đường dẫn thư mục split (ví dụ processed_splits/tofu_seed1)")
    parser.add_argument("--encoder", type=str, default="all-MiniLM-L6-v2", choices=["all-MiniLM-L6-v2", "bge-base-en-v1.5", "e5-base-v2", "Qwen3-Embedding-0.6B"])
    parser.add_argument("--output_dir", type=str, default="cache_embeddings")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()

    encode_unit(
        split_dir=args.split_dir,
        encoder_key=args.encoder,
        output_dir=args.output_dir,
        device=args.device,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()
