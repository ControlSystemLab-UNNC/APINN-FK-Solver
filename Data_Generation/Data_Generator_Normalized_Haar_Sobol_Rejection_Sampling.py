#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stewart Platform Data Generator with Sobol Sequence, Rejection Sampling & Condition Number Filtering
================================================================================================
Updated Strategy:
1. Low-Discrepancy Sequence: 使用 Sobol 序列代替 LHS，保证高维空间均匀性。
2. Rejection Sampling: 
   - 检查1: 腿长约束 (Leg Length Constraints)
   - 检查2: 雅各比矩阵条件数 (Jacobian Condition Number <= 150)
3. Haar Measure SO(3) Sampling: 严格遵守旋转群体积元 dμ = cos(θ)dφdθdψ。
   - Roll/Yaw: 线性均匀采样
   - Pitch: 正弦逆变换采样 arcsin(uniform(sin(min), sin(max)))
"""

import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from typing import Dict, Tuple, List
from tqdm import tqdm
import json
from scipy.stats import qmc  # Sobol

# -----------------------------------------------------------------------------
# 几何与运动学定义
# -----------------------------------------------------------------------------

def create_platform_geometry(spread_angle_deg: float = 45.53,
                             platform_radius: float = 0.160078,
                             base_radius: float = 0.230489):
    """创建Stewart平台几何参数"""
    # 平台铰点角度分布
    platform_angles = [
        -spread_angle_deg, +spread_angle_deg,
        120 -spread_angle_deg, 120 + spread_angle_deg,
        240 - spread_angle_deg, 240 + spread_angle_deg
    ]
    
    # 基座铰点角度分布
    base_spread = 12.53
    base_angles = [
        -base_spread, +base_spread,
        120 - base_spread, 120 + base_spread,
        240 - base_spread, 240 + base_spread
    ]
    
    # 计算局部坐标系下的坐标
    def get_points(radius, angles):
        points = []
        for angle in angles:
            rad = np.radians(angle)
            x = radius * np.cos(rad)
            y = radius * np.sin(rad)
            z = 0.0
            points.append([x, y, z])
        return np.array(points, dtype=np.float64)

    platform_points = get_points(platform_radius, platform_angles)
    base_points = get_points(base_radius, base_angles)

    
    # 物理限制
    min_leg_length = 0.310
    max_leg_length = 0.435
    
    l_char = platform_radius
    
    return {
        'platform_radius': platform_radius,
        'base_radius': base_radius,
        'platform_points': platform_points,
        'base_points': base_points,
        'min_leg_length': min_leg_length,
        'max_leg_length': max_leg_length,
        'spread_angle_deg': spread_angle_deg,
        'L_char': l_char
    }

def compute_leg_lengths(positions: np.ndarray, 
                        eulers: np.ndarray,
                        base_points: np.ndarray, 
                        platform_points: np.ndarray) -> np.ndarray:
    """
    批量计算腿长
    Returns: [N, 6] leg_lengths
    """
    n_samples = positions.shape[0]
    
    # 提取欧拉角
    roll, pitch, yaw = eulers[:, 0], eulers[:, 1], eulers[:, 2]
    
    # 预计算三角函数
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    
    # 构建旋转矩阵 R = Rz * Ry * Rx
    R = np.zeros((n_samples, 3, 3), dtype=np.float64)
    
    # Row 1
    R[:, 0, 0] = cy * cp
    R[:, 0, 1] = cy * sp * sr - sy * cr
    R[:, 0, 2] = cy * sp * cr + sy * sr
    
    # Row 2
    R[:, 1, 0] = sy * cp
    R[:, 1, 1] = sy * sp * sr + cy * cr
    R[:, 1, 2] = sy * sp * cr - cy * sr
    
    # Row 3
    R[:, 2, 0] = -sp
    R[:, 2, 1] = cp * sr
    R[:, 2, 2] = cp * cr
    
    # 计算平台点在世界坐标系的位置: P_world = R * P_local + T
    rotated_points = np.matmul(R, platform_points.T).transpose(0, 2, 1) # [N, 6, 3]
    platform_world = rotated_points + positions[:, np.newaxis, :] # [N, 6, 3]
    
    # 计算腿矢量: L = P_world - Base
    leg_vectors = platform_world - base_points[np.newaxis, :, :] # [N, 6, 3]
    
    # 计算模长
    leg_lengths = np.linalg.norm(leg_vectors, axis=2) # [N, 6]
    
    return leg_lengths

def compute_jacobian_cond(positions: np.ndarray,
                          eulers: np.ndarray,
                          base_points: np.ndarray,
                          platform_points: np.ndarray,
                          L_char: float) -> np.ndarray:
    n_samples = positions.shape[0]
    roll, pitch, yaw = eulers[:, 0], eulers[:, 1], eulers[:, 2]
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    
    R = np.zeros((n_samples, 3, 3), dtype=np.float64)
    R[:, 0, 0] = cy * cp
    R[:, 0, 1] = cy * sp * sr - sy * cr
    R[:, 0, 2] = cy * sp * cr + sy * sr
    R[:, 1, 0] = sy * cp
    R[:, 1, 1] = sy * sp * sr + cy * cr
    R[:, 1, 2] = sy * sp * cr - cy * sr
    R[:, 2, 0] = -sp
    R[:, 2, 1] = cp * sr
    R[:, 2, 2] = cp * cr
    
    r_vectors = np.matmul(R, platform_points.T).transpose(0, 2, 1)
    platform_world = r_vectors + positions[:, np.newaxis, :]
    leg_vectors = platform_world - base_points[np.newaxis, :, :]
    
    norms = np.linalg.norm(leg_vectors, axis=2, keepdims=True)
    norms = np.maximum(norms, 1e-6)
    n_vectors = leg_vectors / norms
    
    cross_products = np.cross(r_vectors, n_vectors)
    
    # [核心修改]: 使用特征长度归一化旋转部分 (Gosselin & Angeles 方法)
    cross_products_norm = cross_products / L_char
    
    # 拼接得到无量纲化雅可比矩阵
    J_norm = np.concatenate([n_vectors, cross_products_norm], axis=2)
    
    # 计算归一化后的条件数
    cond_numbers = np.linalg.cond(J_norm)
    return cond_numbers

# -----------------------------------------------------------------------------
# 采样核心逻辑
# -----------------------------------------------------------------------------

def generate_valid_samples_sobol(target_samples: int,
                                 geometry: dict,
                                 pos_range: float = 0.20,
                                 angle_range_deg: float = 35,
                                 z_min: float = 0.280,
                                 z_max: float = 0.420,
                                 cond_max: float = 150.0,
                                 seed: int = 42,
                                 batch_size: int = 262144) -> Tuple[np.ndarray, np.ndarray]:
    """
    使用Sobol序列生成样本，并进行筛选。严格遵循SO(3) Haar测度。
    """
    print(f"  Starting Sobol sampling (Target: {target_samples}, Cond Max: {cond_max})...")
    
    # 初始化Sobol采样器 (d=6)
    sampler = qmc.Sobol(d=6, scramble=True, seed=seed)
    
    valid_poses_list = []
    valid_legs_list = []
    total_collected = 0
    total_generated = 0
    
    # 物理参数
    min_len = geometry['min_leg_length']
    max_len = geometry['max_leg_length']
    base_points = geometry['base_points']
    plat_points = geometry['platform_points']
    l_char = geometry['L_char']
    
    # 角度范围的正弦值 (用于 Pitch 的 Sine Correction)
    angle_rad_max = np.radians(angle_range_deg)
    sin_angle_max = np.sin(angle_rad_max)
    sin_angle_min = np.sin(-angle_rad_max)
    
    pbar = tqdm(total=target_samples, desc="  Collecting Valid Samples")
    
    while total_collected < target_samples:
        # 1. 生成原始 [0, 1] 样本 (Sobol)
        raw_samples = sampler.random(n=batch_size)
        total_generated += batch_size
        
        # 2. 映射到物理空间
        poses = np.zeros_like(raw_samples)
        
        # --- 位置: 线性映射 ---
        poses[:, 0] = (raw_samples[:, 0] - 0.5) * 2 * pos_range
        poses[:, 1] = (raw_samples[:, 1] - 0.5) * 2 * pos_range
        poses[:, 2] = z_min + raw_samples[:, 2] * (z_max - z_min)
        
        # --- 姿态: 严格的 SO(3) Haar 测度采样 ---
        # Roll (横滚) 和 Yaw (偏航) 使用严格的线性均匀分布
        poses[:, 3] = -angle_rad_max + raw_samples[:, 3] * 2 * angle_rad_max # Roll
        poses[:, 5] = -angle_rad_max + raw_samples[:, 5] * 2 * angle_rad_max # Yaw
        
        # Pitch (俯仰) 必须使用正弦逆变换以补偿余弦体积元畸变
        u_pitch = sin_angle_min + raw_samples[:, 4] * (sin_angle_max - sin_angle_min)
        poses[:, 4] = np.arcsin(u_pitch)
        
        # 3. 第一轮筛选: 腿长约束 (计算快)
        leg_lengths = compute_leg_lengths(poses[:, :3], poses[:, 3:], base_points, plat_points)
        valid_len_mask = np.all((leg_lengths >= min_len) & (leg_lengths <= max_len), axis=1)
        
        # 如果没有点通过第一轮，直接跳过
        if np.sum(valid_len_mask) == 0:
            continue
            
        # 提取通过第一轮的点
        potential_poses = poses[valid_len_mask]
        potential_legs = leg_lengths[valid_len_mask]
        
        # 4. 第二轮筛选: 雅各比条件数 (计算慢，只算通过第一轮的点)
        cond_numbers = compute_jacobian_cond(potential_poses[:, :3], potential_poses[:, 3:], base_points, plat_points, l_char)
        valid_cond_mask = cond_numbers <= cond_max
        
        # 最终有效点
        final_valid_poses = potential_poses[valid_cond_mask]
        final_valid_legs = potential_legs[valid_cond_mask]
        
        if len(final_valid_poses) > 0:
            valid_poses_list.append(final_valid_poses)
            valid_legs_list.append(final_valid_legs)
            
            new_count = len(final_valid_poses)
            total_collected += new_count
            pbar.update(new_count)
            
    pbar.close()
    
    # 拼接结果
    all_poses = np.vstack(valid_poses_list)
    all_legs = np.vstack(valid_legs_list)
    
    # 截取精确数量
    final_poses = all_poses[:target_samples]
    final_legs = all_legs[:target_samples]
    
    rejection_rate = 1.0 - (total_collected / total_generated)
    print(f"  Done. Rejection Rate: {rejection_rate*100:.2f}% (Generated {total_generated} to get {total_collected})")
    
    return final_poses, final_legs

# -----------------------------------------------------------------------------
# 数据集封装与保存
# -----------------------------------------------------------------------------

def save_dataset(poses: np.ndarray, legs: np.ndarray, split_name: str, output_dir: Path):
    """保存为PyTorch格式"""
    dataset = {
        'poses': torch.tensor(poses, dtype=torch.float32),        # [N, 6]
        'leg_lengths': torch.tensor(legs, dtype=torch.float32),   # [N, 6]
        'split': split_name,
        'n_samples': poses.shape[0]
    }
    
    save_path = output_dir / f'{split_name}.pt'
    torch.save(dataset, save_path)
    print(f"  Saved {split_name} dataset to {save_path}")

def visualize_distribution(poses: np.ndarray, output_dir: Path, title_suffix: str = ""):
    """简单的分布可视化"""
    fig = plt.figure(figsize=(12, 5))
    
    # 1. 3D Position
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    idx = np.random.choice(len(poses), min(5000, len(poses)), replace=False)
    sub = poses[idx]
    
    sc = ax1.scatter(sub[:, 0], sub[:, 1], sub[:, 2], c=sub[:, 2], cmap='viridis', s=1)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title(f'Position Distribution {title_suffix}')
    plt.colorbar(sc, ax=ax1, shrink=0.5)
    
    # 2. Angle Histograms
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.hist(np.rad2deg(poses[:, 3]), bins=50, alpha=0.5, label='Roll')
    ax2.hist(np.rad2deg(poses[:, 4]), bins=50, alpha=0.5, label='Pitch')
    ax2.hist(np.rad2deg(poses[:, 5]), bins=50, alpha=0.5, label='Yaw')
    ax2.set_xlabel('Angle (deg)')
    ax2.set_title('Orientation Histogram (Haar Measure)')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / f'distribution_{title_suffix.lower()}.png')
    plt.close()

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Stewart Platform Data Generator (Sobol + Rejection + HaarMeasure + CondCheck)")
    
    # 数据集大小
    parser.add_argument('--n_train', type=int, default=300000)
    parser.add_argument('--n_val', type=int, default=100000)
    parser.add_argument('--n_test', type=int, default=500000)
    
    # 几何参数
    parser.add_argument('--pos_range', type=float, default=0.20)
    parser.add_argument('--z_min', type=float, default=0.280)
    parser.add_argument('--z_max', type=float, default=0.420) 
    parser.add_argument('--angle_range', type=float, default=35.0) # degrees
    parser.add_argument('--cond_max', type=float, default=150.0)  # 最大条件数
    
    parser.add_argument('--output_dir', type=str, default='./stewart_sobol_dataset')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print(" STEWART DATA GENERATOR (Sobol + Rejection + HaarMeasure + CondCheck)")
    print("="*60)
    
    # 1. 创建几何模型
    geometry = create_platform_geometry()
    
    # 2. 生成各部分数据
    
    # Train
    print(f"\n[Train Set] Generating {args.n_train} samples...")
    train_poses, train_legs = generate_valid_samples_sobol(
        args.n_train, geometry, 
        args.pos_range, args.angle_range, args.z_min, args.z_max, args.cond_max,
        seed=42
    )
    save_dataset(train_poses, train_legs, 'train', output_dir)
    
    # Val
    print(f"\n[Val Set] Generating {args.n_val} samples...")
    val_poses, val_legs = generate_valid_samples_sobol(
        args.n_val, geometry, 
        args.pos_range, args.angle_range, args.z_min, args.z_max, args.cond_max,
        seed=100
    )
    save_dataset(val_poses, val_legs, 'val', output_dir)
    
    # Test
    print(f"\n[Test Set] Generating {args.n_test} samples...")
    test_poses, test_legs = generate_valid_samples_sobol(
        args.n_test, geometry, 
        args.pos_range, args.angle_range, args.z_min, args.z_max, args.cond_max,
        seed=200
    )
    save_dataset(test_poses, test_legs, 'test', output_dir)
    
    # 3. 可视化检查
    print("\nVisualizing Train distribution...")
    visualize_distribution(train_poses, output_dir, "Train")
    
    # 4. 保存Metadata
    metadata = {
        'geometry': {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in geometry.items()},
        'settings': vars(args),
        'sampling_method': 'Sobol + Rejection(Leg+Cond) + HaarMeasure'
    }
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
        
    print(f"\n✅ All done! Data saved to {output_dir}")

if __name__ == '__main__':
    main()
