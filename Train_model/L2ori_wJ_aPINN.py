#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adaptive PINN (Diagonal + Loop Closure)
基于 Pure MLP，集成 Adaptive 权重机制。

物理损失完全复现论文公式 (Eq 17):
L_phys = L_fk + beta * L_diag

1. L_fk (Loop-Closure): || F_IK(pred_pose) - input_legs ||^2
2. L_diag: (1/6) * Σ[(J_inv @ J_network)_ii - 1]^2
3. Adaptive Weighting: α(κ) 作用于整个物理损失项

Updates:
- 修正了 J_network 的计算，引入了变换矩阵 E (Eq 9, 10)，将欧拉角梯度映射为角速度雅可比。
==================================================
"""

import sys
import os
from datetime import datetime
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import argparse
import random
from tqdm import tqdm

# ==================== 1. 基础配置 ====================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

# ==================== 2. 几何与数学工具 ====================
def euler_to_rotation_matrix_batch(euler_rad):
    """批量欧拉角(弧度)转旋转矩阵 [B,3] -> [B,3,3] (ZYX顺序)"""
    batch_size = euler_rad.shape[0]
    device = euler_rad.device
    
    roll, pitch, yaw = euler_rad[:, 0], euler_rad[:, 1], euler_rad[:, 2]
    
    cos_r, sin_r = torch.cos(roll), torch.sin(roll)
    cos_p, sin_p = torch.cos(pitch), torch.sin(pitch)
    cos_y, sin_y = torch.cos(yaw), torch.sin(yaw)
    
    R = torch.zeros(batch_size, 3, 3, device=device)
    
    R[:, 0, 0] = cos_y * cos_p
    R[:, 0, 1] = cos_y * sin_p * sin_r - sin_y * cos_r
    R[:, 0, 2] = cos_y * sin_p * cos_r + sin_y * sin_r
    
    R[:, 1, 0] = sin_y * cos_p
    R[:, 1, 1] = sin_y * sin_p * sin_r + cos_y * cos_r
    R[:, 1, 2] = sin_y * sin_p * cos_r - cos_y * sin_r
    
    R[:, 2, 0] = -sin_p
    R[:, 2, 1] = cos_p * sin_r
    R[:, 2, 2] = cos_p * cos_r
    
    return R

def get_platform_geometry():
    """Stewart平台几何参数定义"""
    R = 160.078 / 1000  # 动平台半径 (m)
    r = 230.489 / 1000  # 静平台半径 (m)
    
    up = [-45.53, 45.53, 120-45.53, 120+45.53, 240-45.53, 240+45.53]
    down = [-12.53, 12.53, 120-12.53, 120+12.53, 240-12.53, 240+12.53]
    
    platform_points = torch.tensor([
        [R*np.cos(np.deg2rad(a)), R*np.sin(np.deg2rad(a)), 0.0] for a in up
    ], dtype=torch.float32)
    
    base_points = torch.tensor([
        [r*np.cos(np.deg2rad(a)), r*np.sin(np.deg2rad(a)), 0.0] for a in down
    ], dtype=torch.float32)
    
    return base_points, platform_points

def compute_J_inv_analytical(pose, base_points, platform_points):
    """计算解析逆运动学雅可比 J_inv"""
    batch_size = pose.shape[0]
    device = pose.device
    
    T = pose[:, :3]      # 平移
    euler = pose[:, 3:]  # 欧拉角
    R_mat = euler_to_rotation_matrix_batch(euler)
    
    # r_i = R * p_i
    r_i = torch.matmul(R_mat, platform_points.to(device).T).transpose(-1, -2) # [B, 6, 3]
    d_i = T.unsqueeze(1) - base_points.to(device).unsqueeze(0) # [B, 6, 3]
    l_i = r_i + d_i 
    L_i = torch.norm(l_i, dim=-1, keepdim=True) # [B, 6, 1]
    c_i = torch.cross(r_i, d_i, dim=-1)
    
    J_inv = torch.zeros(batch_size, 6, 6, device=device)
    J_inv[:, :, :3] = l_i / (L_i + 1e-8)
    J_inv[:, :, 3:] = c_i / (L_i + 1e-8)
    
    return J_inv

def compute_J_network(model, legs_norm, legs_std, create_graph=True):
    """
    通过 Autograd 计算网络雅可比 J_fk = ∂pose/∂L
    [Corrected] 引入矩阵 E 将欧拉角梯度映射为角速度雅可比 (Eq 9, 10)
    J_NN = diag(I3, E) * d(x_hat)/d(L)
    """
    device = legs_norm.device
    legs_input = legs_norm.clone().detach().requires_grad_(True)
    
    pose = model(legs_input)
    
    # 1. 计算原始 Autograd 雅可比 (Raw Gradients)
    # J_raw shape: [B, 6, 6] -> 前3行是位置梯度，后3行是欧拉角梯度
    J_rows = []
    for i in range(6):
        grad_outputs = torch.zeros_like(pose)
        grad_outputs[:, i] = 1.0
        grads = torch.autograd.grad(
            outputs=pose, inputs=legs_input, grad_outputs=grad_outputs,
            create_graph=create_graph, retain_graph=True
        )[0]
        J_rows.append(grads)
        
    J_raw = torch.stack(J_rows, dim=1) # [B, 6, 6]
    
    # 2. 构建变换矩阵 E (Eq 10)
    # pose 结构: [x, y, z, roll, pitch, yaw]
    # 对应: roll(phi), pitch(theta), yaw(psi)
    theta = pose[:, 4] # Pitch
    psi = pose[:, 5]   # Yaw
    
    c_theta = torch.cos(theta)
    s_theta = torch.sin(theta)
    c_psi = torch.cos(psi)
    s_psi = torch.sin(psi)
    zeros = torch.zeros_like(theta)
    ones = torch.ones_like(theta)
    
    # 构建 3x3 矩阵 E 的列向量
    # Col 1: [c_theta * c_psi, c_theta * s_psi, -s_theta]^T
    # Col 2: [-s_psi,          c_psi,            0      ]^T
    # Col 3: [0,               0,                1      ]^T
    # 注意：图片中矩阵是 [row1; row2; row3]，这里按行构建
    
    # Row 1: [c_theta * c_psi, -s_psi, 0]
    row1 = torch.stack([c_theta * c_psi, -s_psi, zeros], dim=1)
    # Row 2: [c_theta * s_psi, c_psi, 0]
    row2 = torch.stack([c_theta * s_psi, c_psi, zeros], dim=1)
    # Row 3: [-s_theta, 0, 1]
    row3 = torch.stack([-s_theta, zeros, ones], dim=1)
    
    E_mat = torch.stack([row1, row2, row3], dim=1) # [B, 3, 3]
    
    # 3. 应用变换: J_angular = E * J_euler_raw
    J_pos_raw = J_raw[:, :3, :]   # [B, 3, 6]
    J_euler_raw = J_raw[:, 3:, :] # [B, 3, 6]
    
    # 批量矩阵乘法 [B,3,3] x [B,3,6] -> [B,3,6]
    J_angular_transformed = torch.bmm(E_mat, J_euler_raw)
    
    # 重新拼接
    J_network_combined = torch.cat([J_pos_raw, J_angular_transformed], dim=1) # [B, 6, 6]
    
    # 4. 归一化处理 (除以输入的 std)
    J_network = J_network_combined / (legs_std.to(device).unsqueeze(0).unsqueeze(0) + 1e-8)
    
    return J_network, pose

# ==================== 3. 模型定义 ====================
class SimpleMLP(nn.Module):
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, 6) 
        )
    def forward(self, x): return self.net(x)

# ==================== 4. 数据集 ====================
class StewartDataset(Dataset):
    def __init__(self, pt_path, legs_mean, legs_std, noise_std=0.0):
        data = torch.load(pt_path)
        if isinstance(data, dict):
            self.legs = data['leg_lengths'].numpy()
            self.poses = data['poses'].numpy()
        else:
            all_legs = []
            all_poses = []
            for s in data:
                all_legs.append(s['input']['A']['edge_features'][:6, 0].numpy())
                p_pos = s['target']['pose_position'].numpy()
                p_euler = s['target']['pose_euler'].numpy()
                all_poses.append(np.concatenate([p_pos, p_euler]))
            self.legs = np.array(all_legs)
            self.poses = np.array(all_poses)

        self.legs_mean = legs_mean
        self.legs_std = legs_std
        # 归一化
        self.legs = (self.legs - legs_mean.numpy()) / (legs_std.numpy() + 1e-8)
        self.noise_std = noise_std
        
        # 初始化 Kappa 数组
        self.kappas = np.zeros(len(self.legs), dtype=np.float32)
 
    def compute_kappas(self):
        print("Pre-computing Normalized Condition Numbers (Kappas) for Adaptive Loss...")
        b, p = get_platform_geometry()
        batch_size = 2000
        kappas_list = []
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        b, p = b.to(device), p.to(device)
        
        # 定义特征长度 L_char (动平台半径，单位：米)
        L_char = 160.078 / 1000.0
        
        with torch.no_grad():
            for i in tqdm(range(0, len(self.poses), batch_size)):
                batch_pose = torch.from_numpy(self.poses[i:i+batch_size]).float().to(device)
                
                # 1. 计算原始的雅可比矩阵
                J_inv = compute_J_inv_analytical(batch_pose, b, p)
                
                # 2. 核心修改：拷贝一份用于归一化，不影响后续 Loss 中的求导和矩阵乘法
                J_norm = J_inv.clone()
                
                # 3. 将旋转部分（后3列）除以特征长度 L_char，统一量纲 (Gosselin & Angeles 法)
                J_norm[:, :, 3:] = J_norm[:, :, 3:] / L_char
                
                # 4. 计算归一化后的条件数
                k = torch.linalg.cond(J_norm).cpu().numpy()
                kappas_list.append(k)
        
        self.kappas = np.concatenate(kappas_list)
        print(f"Normalized Kappas computed. Min: {self.kappas.min():.2f}, Max: {self.kappas.max():.2f}")
    def __len__(self): return len(self.legs)
    
    def __getitem__(self, idx):
        legs = self.legs[idx].copy()
        if self.noise_std > 0:
            legs += np.random.randn(6).astype(np.float32) * self.noise_std
            
        return {
            'legs': torch.from_numpy(legs),
            'pose': torch.from_numpy(self.poses[idx]),
            'kappa': torch.tensor(self.kappas[idx], dtype=torch.float32)
        }

def compute_leg_stats_pt(pt_path):
    data = torch.load(pt_path)
    if isinstance(data, dict):
        all_legs = data['leg_lengths'].numpy()
    else:
        all_legs = np.array([s['input']['A']['edge_features'][:6, 0].numpy() for s in data])
    return all_legs.mean(axis=0), all_legs.std(axis=0)

# ==================== 5. 损失函数 (Updated) ====================
class AdaptivePhysicsLoss(nn.Module):
    def __init__(self, base_points, platform_points, kappa_threshold=50.0, 
                 kappa_scale=10.0, max_alpha=5.0, adaptive_offset=2.0, beta=0.1, angular_weight=50.0):
        super().__init__()
        self.register_buffer('base_points', base_points)
        self.register_buffer('platform_points', platform_points)
        
        self.kappa_threshold = kappa_threshold
        self.kappa_scale = kappa_scale
        self.max_alpha = max_alpha
        self.adaptive_offset = adaptive_offset
        self.beta = beta  # 论文中的平衡系数 beta

        # =========== 雅可比姿态分块权重矩阵 ===========
        # 这是一个 6x6 矩阵
        # 前3行 (0-2) 对应线速度 v -> 权重 1.0
        # 后3行 (3-5) 对应角速度 w -> 权重 angular_weight (比如 50.0)
        weights = torch.ones(6, 6)
        weights[3:, :] = angular_weight 
        self.register_buffer('matrix_weights', weights)
        # ===============================================
        
        self.mse = nn.MSELoss()
        self.w_pos = 600.0
        self.w_ori = 300.0

    def compute_adaptive_alpha(self, kappas):
        """
        当 kappa 小 (良态) -> alpha 接近 1.0 
        当 kappa 大 (病态) -> alpha 接近 max_alpha
        """
        sigmoid_input = (kappas - self.kappa_threshold) / self.kappa_scale
        score = torch.sigmoid(sigmoid_input)
        alpha = self.adaptive_offset + (self.max_alpha - self.adaptive_offset) * score
        return alpha
            
    def forward(self, model, legs_norm, gt_pose, kappas, legs_mean, legs_std):
        device = legs_norm.device
        
        # 1. 网络前向与雅可比计算 (已修正)
        J_network, pred_pose = compute_J_network(model, legs_norm, legs_std, create_graph=True)
        
        # 2. 数据损失 (Supervised)
        loss_pos = self.mse(pred_pose[:, :3] * self.w_pos, gt_pose[:, :3] * self.w_pos)
        loss_ori = self.mse(pred_pose[:,3:] * self.w_ori, gt_pose[:,3:] * self.w_ori)   
        L_data = loss_pos + loss_ori
        
        # ============ 物理约束部分 (Physics Informed) ============
        
        # A. 真实逆运动学计算 (Loop-Closure / F_IK)
        T = pred_pose[:, :3]
        euler = pred_pose[:, 3:]
        R_mat = euler_to_rotation_matrix_batch(euler)
        
        platform_world = torch.matmul(R_mat, self.platform_points.T).transpose(-1, -2) + T.unsqueeze(1) 
        leg_vectors = platform_world - self.base_points.unsqueeze(0) 
        computed_legs_m = torch.norm(leg_vectors, dim=-1) 
        
        # B. 将重建的腿长重新归一化
        computed_legs_norm = (computed_legs_m - legs_mean) / (legs_std + 1e-8)
        
        # C. 物理损失项1: Loop-Closure Consistency
        L_fk_per_sample = torch.mean((computed_legs_norm - legs_norm) ** 2, dim=1) 
        
        # D. 物理损失项2: Full Matrix Identity Constraint       
        # D1. 计算解析雅可比 J_gt(x_hat) -> 使用 gt_pose (或 pred_pose，视策略而定，通常使用 gt 更稳定)
        J_inv = compute_J_inv_analytical(gt_pose, self.base_points, self.platform_points)
        
        # D2. 计算矩阵乘积 J_net * J_inv
        product = torch.bmm(J_network, J_inv) # [B, 6, 6]
        
        # D3. 创建目标单位矩阵 I
        batch_size = product.shape[0]
        target_eye = torch.eye(6, device=device).unsqueeze(0).expand(batch_size, -1, -1)
        
        # D4. 计算全矩阵均方误差
        diff_sq = (product - target_eye) ** 2 
        weighted_diff = diff_sq * self.matrix_weights
        L_matrix_per_sample = torch.mean(weighted_diff, dim=(1,2))
      
        # E. 组合物理损失
        L_phys_per_sample = L_fk_per_sample + self.beta * L_matrix_per_sample
        
        # F. 自适应加权
        alphas = self.compute_adaptive_alpha(kappas) 
        weighted_physics = (alphas * L_phys_per_sample).mean()
        
        # 3. 正则化
        L_reg = 0.0
        for p in model.parameters():
            L_reg += torch.sum(p ** 2)
        L_reg *= 1e-5
        
        # 总损失
        total = L_data + weighted_physics + L_reg
        
        return total, {
            'pos': loss_pos.item(),
            'ori': loss_ori.item(),
            'phy_total': weighted_physics.item(),
            'L_fk_raw': L_fk_per_sample.mean().item(),
            'L_matrix_raw': L_matrix_per_sample.mean().item(),
            'alpha_mean': alphas.mean().item()
        }

# ==================== 6. 训练逻辑 ====================
def train(args):
    set_seed(args.seed)
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.cuda.set_device(0) 
    else:
        device = torch.device('cpu')
        
    print(f"Training Adaptive PINN (LoopClosure + Diagonal) on {device}...")

    # ================= 创建结果目录 =================
    base_timestamp = datetime.now().strftime("%y%m%d_%H%M")
    base_dir = os.path.join("./result", base_timestamp)
    
    if not os.path.exists(base_dir):
        result_dir = base_dir
    else:
        counter = 1
        while True:
            new_dir = f"{base_dir}_{counter}"
            if not os.path.exists(new_dir):
                result_dir = new_dir
                break
            counter += 1
            
    os.makedirs(result_dir, exist_ok=True)
    
    model_filename = os.path.basename(args.save_path)
    final_save_path = os.path.join(result_dir, model_filename)
    
    print(f"Results will be saved to: {result_dir}")
    
    b, p = get_platform_geometry()
    
    # 统计量
    lm, ls = compute_leg_stats_pt(args.train_data)
    lm, ls = torch.from_numpy(lm.astype(np.float32)), torch.from_numpy(ls.astype(np.float32))
    lm_dev, ls_dev = lm.to(device), ls.to(device)
    
    print("Loading Training Data...")
    train_ds = StewartDataset(args.train_data, lm, ls, noise_std=args.noise_std)
    train_ds.compute_kappas()
    
    val_ds = StewartDataset(args.val_data, lm, ls)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    
    model = SimpleMLP(args.hidden_dim).to(device)
    
    loss_fn = AdaptivePhysicsLoss(
        b, p, 
        kappa_threshold=args.kappa_threshold,
        kappa_scale=args.kappa_scale,
        max_alpha=args.max_alpha,
        adaptive_offset=args.adaptive_offset,
        beta=args.beta,
        angular_weight=args.angular_weight
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)
    
    best_val_pos = float('inf')
    best_val_ori = float('inf')
    best_epoch = 0
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        ep_loss = 0.0
        ep_phy = 0.0
        ep_fk = 0.0
        ep_diag = 0.0
        ep_alpha = 0.0
        
        for batch in train_loader:
            legs = batch['legs'].to(device)
            gt = batch['pose'].to(device)
            kappas = batch['kappa'].to(device)
            
            optimizer.zero_grad()
            
            loss, comps = loss_fn(model, legs, gt, kappas, lm_dev, ls_dev)
            
            loss.backward()
            optimizer.step()
            
            ep_loss += loss.item()
            ep_phy += comps['phy_total']
            ep_fk += comps['L_fk_raw']
            ep_diag += comps['L_matrix_raw']
            ep_alpha += comps['alpha_mean']
            
        # --- 验证 ---
        if epoch % 10 == 0:
            model.eval()
            val_pos_accum = 0.0
            val_ori_accum = 0.0
            
            with torch.no_grad():
                for batch in val_loader:
                    legs = batch['legs'].to(device)
                    gt = batch['pose'].to(device)
                    pred = model(legs)
                    
                    batch_pos_err = torch.norm((pred[:,:3]-gt[:,:3])*1000.0, dim=1).mean().item()
                    batch_ori_err = torch.mean(torch.abs(pred[:,3:]-gt[:,3:]) * 180.0 / np.pi).item()
                    
                    val_pos_accum += batch_pos_err
                    val_ori_accum += batch_ori_err
            
            curr_val_pos = val_pos_accum / len(val_loader)
            curr_val_ori = val_ori_accum / len(val_loader)

            scheduler.step(curr_val_pos)

            n_batches = len(train_loader)
            avg_loss = ep_loss / n_batches
            avg_phy = ep_phy / n_batches
            avg_fk = ep_fk / n_batches
            avg_diag = ep_diag / n_batches
            avg_alpha = ep_alpha / n_batches
            
            if curr_val_pos < best_val_pos:
                best_val_pos = curr_val_pos
                best_val_ori = curr_val_ori
                best_epoch = epoch
                
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'legs_mean': lm,
                    'legs_std': ls,
                    'epoch': epoch,
                    'val_pos_error': best_val_pos,
                    'val_ori_error': best_val_ori
                }, final_save_path)
            
            print(f"Epoch {epoch:03d}: "
                  f"Train Loss: {avg_loss:.5f} "
                  f"(Phy: {avg_phy:.5f} [FK:{avg_fk:.5f}, Mat:{avg_diag:.5f}], α:{avg_alpha:.2f}) | "
                  f"Val: {curr_val_pos:.4f}mm / {curr_val_ori:.4f}° | "
                  f"Best: {best_val_pos:.4f}mm (@Ep {best_epoch})")

    print(f"\n✅ Finished. Best Model @ Ep {best_epoch}: {best_val_pos:.4f}mm / {best_val_ori:.4f}°")
    
    info_path = os.path.join(result_dir, "result_info.txt")
    with open(info_path, "w", encoding='utf-8') as f:
        f.write("="*20 + " Training Results " + "="*20 + "\n")
        f.write(f"Best Epoch: {best_epoch}\n")
        f.write(f"Best Val Position Error: {best_val_pos:.6f} mm\n")
        f.write(f"Best Val Orientation Error: {best_val_ori:.6f} deg\n\n")
        f.write("="*20 + " Configuration " + "="*20 + "\n")
        args_dict = vars(args)
        for key, value in args_dict.items():
            f.write(f"{key}: {value}\n")
            
    print(f"Log saved: {info_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', default="/home/txy/APINN_Train/UPS_stewart/Data_generation/stewart_sobol_dataset/train.pt", help="Path to training data")
    parser.add_argument('--val_data', default="/home/txy/APINN_Train/UPS_stewart/Data_generation/stewart_sobol_dataset/val.pt", help="Path to validation data")
    parser.add_argument('--test_data', default="/home/txy/APINN_Train/UPS_stewart/Data_generation/stewart_sobol_dataset/test.pt", help="Path to test data")
    parser.add_argument('--save_path', default='L2ori_wJ_adaptive_PINN.pt')
    
    parser.add_argument('--kappa_threshold', type=float, default=7.5)
    parser.add_argument('--kappa_scale', type=float, default=1.0)
    parser.add_argument('--max_alpha', type=float, default=1000.0)
    parser.add_argument('--adaptive_offset', type=float, default=200.0)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--angular_weight', type=float, default=1.0)
    parser.add_argument('--epochs', type=int, default=6000)
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--noise_std', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    train(args)
