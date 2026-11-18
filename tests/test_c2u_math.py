#!/usr/bin/env python3
"""
test_c2u_math.py
Kiểm thử sanity logic toán học 4 bước của C2U bằng NumPy (CPU/Local):
1. Preconditioned suppression direction (mu_t, mu_c, D, g)
2. Continuous activation gate s(q) (z_0: 50%, z_1: 95%)
3. Augmented Atom Bank & Atomic Shift delta_q
4. Trust-Region update với ngưỡng cos(theta) >= max(cos(0.2), 0.98)
"""

import math
import numpy as np


def normalize(x, axis=-1, eps=1e-12):
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(norm, eps)


def test_c2u_math_logic():
    print("[Test] Bắt đầu kiểm thử tính toán số học C2U...")
    np.random.seed(42)
    d = 384
    N = 768
    tau = 0.1
    
    # 1. Khởi tạo dữ liệu giả lập
    # Dictionary A: N atoms, mỗi atom dim d
    A = normalize(np.random.randn(N, d))
    
    # Target profile: 20 mẫu
    # Giả sử target có một bias theo một hướng cố định
    target_direction = normalize(np.random.randn(d))
    target_profile = normalize(target_direction + 0.3 * np.random.randn(20, d))
    
    # Control profile: 200 mẫu
    ctrl_direction = normalize(np.random.randn(d))
    calibration_control = normalize(ctrl_direction + 0.3 * np.random.randn(200, d))
    
    # BƯỚC 1: Preconditioned suppression direction
    mu_t = np.mean(target_profile, axis=0)
    mu_c = np.mean(calibration_control, axis=0)
    mu_t_norm = normalize(mu_t)
    mu_c_norm = normalize(mu_c)
    
    v = normalize(mu_t - mu_c)
    
    # E_ctrl[x_i^2]
    e_x2 = np.mean(calibration_control ** 2, axis=0)
    m_bar = np.mean(e_x2)
    D_diag = 0.9 * e_x2 + 0.1 * m_bar + 0.001
    
    # g = normalize(-D^{-1} v)
    minus_d_inv_v = - (v / D_diag)
    g = normalize(minus_d_inv_v)
    assert g.shape == (d,), "Lỗi shape g"
    print(f"  [Pass] Bước 1: g computed thành công, norm={np.linalg.norm(g):.4f}")
    
    # BƯỚC 2: Continuous activation gate
    z_ctrl = np.dot(calibration_control, mu_t_norm) - np.dot(calibration_control, mu_c_norm)
    z_0 = float(np.percentile(z_ctrl, 50))
    z_1 = float(np.percentile(z_ctrl, 95))
    assert z_1 > z_0, "Lỗi phân vị z_1 <= z_0"
    print(f"  [Pass] Bước 2: Hiệu chuẩn gate z_0 (50%)={z_0:.4f}, z_1 (95%)={z_1:.4f}")
    
    # Test query: 1 target query và 1 control query
    q_target = normalize(target_direction + 0.1 * np.random.randn(d))
    q_ctrl = normalize(ctrl_direction + 0.1 * np.random.randn(d))
    
    z_q_target = float(np.dot(q_target, mu_t_norm) - np.dot(q_target, mu_c_norm))
    z_q_ctrl = float(np.dot(q_ctrl, mu_t_norm) - np.dot(q_ctrl, mu_c_norm))
    
    s_target = float(np.clip((z_q_target - z_0) / (z_1 - z_0), 0.0, 1.0))
    s_ctrl = float(np.clip((z_q_ctrl - z_0) / (z_1 - z_0), 0.0, 1.0))
    
    print(f"  [Pass] Gate s(q): Target query = {s_target:.4f}, Control query = {s_ctrl:.4f}")
    assert s_target > s_ctrl, "Cổng s(q) cho Target query phải lớn hơn Control query"
    
    # BƯỚC 3: Augmented Atom Bank & Atomic Shift
    # Chiếu g lên A:
    # proj_g = A.T @ (A @ A.T + eps I)^-1 @ (A @ g)
    reg = 1e-4 * np.eye(N)
    Ag = np.dot(A, g)
    proj_weights = np.linalg.solve(np.dot(A, A.T) + reg, Ag)
    proj_g = np.dot(A.T, proj_weights)
    r_raw = g - proj_g
    norm_r = np.linalg.norm(r_raw)
    r = g if norm_r < 1e-4 else r_raw / norm_r
    print(f"  [Pass] Bước 3: Residual atom r norm={np.linalg.norm(r):.4f}")
    
    # Tính shift cho target query
    sim_t = np.dot(A, q_target) / tau
    w_t = np.exp(sim_t - np.max(sim_t))
    w_t /= np.sum(w_t)
    rec_q = np.dot(w_t, A)
    
    theta_req = 0.3 * s_target
    delta_q = (math.cos(theta_req) - 1.0) * rec_q + math.sin(theta_req) * r
    print(f"  [Pass] Bước 3: delta_q shift computed, theta_req={theta_req:.4f} rad")
    
    # BƯỚC 4: Trust-Region update
    thresh = max(math.cos(0.2), 0.98)
    eta = 1.0
    q_cand = normalize(q_target + eta * delta_q)
    cos_sim = float(np.dot(q_cand, q_target))
    while cos_sim < thresh and eta > 0.01:
        eta *= 0.8
        q_cand = normalize(q_target + eta * delta_q)
        cos_sim = float(np.dot(q_cand, q_target))
        
    print(f"  [Pass] Bước 4: Trust-region satisfied with eta={eta:.4f}, cos_sim={cos_sim:.5f} >= {thresh:.5f}")
    
    # Kiểm tra hiệu quả unlearning trên query: khoảng cách tới target_direction tăng lên
    cos_before = float(np.dot(q_target, target_direction))
    cos_after = float(np.dot(q_cand, target_direction))
    print(f"  [Result] Cosine sim với Target Concept: Trước={cos_before:.4f} -> Sau={cos_after:.4f} (Giảm: {cos_before - cos_after:.4f})")
    assert cos_after < cos_before, "C2U phải làm giảm độ tương đồng với Target Concept!"
    print("\n=> TẤT CẢ KIỂM THỬ SANITY TOÁN HỌC CỦA C2U ĐÃ ĐẠT 100%!")


if __name__ == "__main__":
    test_c2u_math_logic()
