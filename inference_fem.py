"""使用神经网络进行推理与可视化"""

import numpy as np
import torch
import taichi as ti
from torch_geometric.data import Data, Batch
from collections import deque

# 导入模型定义
from model.network import FEMSurrogate
from utils.normalization import Normalizer

# 导入FEM求解器（已经初始化了Taichi）
from tiFEM import (
    get_vertices, init, get_indices,
    pos, pos0, substep_implicit, vel,
    n_verts, dt,
    compute_potential_energy, get_vol_err,
    edge_indices, gravity_constant
)


class FEMInference:
    """使用训练好的神经网络进行FEM推理"""
    
    def __init__(self, model_path='best_fem_model.pth', device='cuda'):
        self.device = device
        # 使用训练时的配置 (必须与训练时的参数一致)
        self.model = FEMSurrogate(
            n_nodes=4, 
            message_passing_num=3, 
            hidden_dim=128,
            node_feat_dim=9,      # 3步×3维速度
            edge_feat_dim=6       # current_len(3) + rest_len(3)
        )
        
        # 加载 checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()

        # 预先缓存边索引
        edges_np = edge_indices.to_numpy()
        self.edge_index = torch.tensor(edges_np, dtype=torch.long).t().contiguous()
        
        # 缓存最近两步速度和当前速度
        self.vel_buffer = deque(maxlen=3)  # 保存 [vel_t_minus_2, vel_t_minus_1, vel_t]
        
        # 初始化速度归一化器 (必须与训练时一致)
        self.vel_normalizer = Normalizer(size=9, device=device)
        self.next_vel_normalizer = Normalizer(size=3, device=device)
        
        # 从 checkpoint 中加载 normalizer 的累积统计信息
        self.vel_normalizer._acc_sum = checkpoint['vel_history_normalizer_acc_sum'].to(device)
        self.vel_normalizer._acc_sum_squared = checkpoint['vel_history_normalizer_acc_sum_squared'].to(device)
        self.vel_normalizer._acc_count = checkpoint['vel_history_normalizer_acc_count'].to(device)
        self.next_vel_normalizer._acc_sum = checkpoint['next_vel_normalizer_acc_sum'].to(device)
        self.next_vel_normalizer._acc_sum_squared = checkpoint['next_vel_normalizer_acc_sum_squared'].to(device)
        self.next_vel_normalizer._acc_count = checkpoint['next_vel_normalizer_acc_count'].to(device)
        
        self.step_count = 0
        
        print(f"Model loaded from {model_path} on {device}")
        print(f"Normalizer statistics loaded from checkpoint")
    
    def predict_next_vel(self):
        """使用神经网络预测下一步速度"""
        # 获取当前状态
        pos_np = pos.to_numpy()        # [N,3]
        pos0_np = pos0.to_numpy()      # [N,3]
        vel_np = vel.to_numpy()        # [N,3]
        
        # 获取历史速度
        vel_history_list = list(self.vel_buffer)  # 应该有3个元素
        if len(vel_history_list) != 3:
            raise ValueError(f"Buffer should have 3 velocities, but has {len(vel_history_list)}")
        
        # 转为torch
        pos_torch = torch.from_numpy(pos_np).float()
        pos0_torch = torch.from_numpy(pos0_np).float()
        
        # 拼接历史速度
        vel_t_minus_2_torch = torch.from_numpy(vel_history_list[0]).float()
        vel_t_minus_1_torch = torch.from_numpy(vel_history_list[1]).float()
        vel_t_torch = torch.from_numpy(vel_history_list[2]).float()
        vel_history_torch = torch.cat([vel_t_minus_2_torch, vel_t_minus_1_torch, vel_t_torch], dim=-1)  # [N, 9]
        
        # 构建图特征
        src = self.edge_index[0]   # [E]
        dst = self.edge_index[1]   # [E]

        current_len = pos_torch[dst] - pos_torch[src]     # [E,3]
        rest_len    = pos0_torch[dst] - pos0_torch[src]   # [E,3]
        edge_attr = torch.cat([current_len, rest_len], dim=-1)  # [E,6]
        
        # 创建图
        graph = Data(
            x=vel_history_torch,
            edge_index=self.edge_index,
            edge_attr=edge_attr.float(),
        )
        batch = Batch.from_data_list([graph]).to(self.device)
        
        # Normalize输入
        batch.x = self.vel_normalizer(batch.x, accumulate=False)
        
        # 推理
        with torch.no_grad():
            output = self.model(batch)  # (num_nodes, 3) - normalized velocity
        
        # Denormalize输出
        output = self.next_vel_normalizer.inverse(output)
        
        output_np = output.cpu().numpy()
        next_vel = output_np  # [N,3]
        
        return next_vel
    
    def run_nn_step(self):
        from tiFEM import floor_bound, integrate_vel_explicit_euler
        """运行一步模拟"""
        self.step_count += 1
        
        if self.step_count <= 3:
            # 前3步使用FEM真实模拟
            substep_implicit()
            # 记录当前速度到缓冲区
            self.vel_buffer.append(vel.to_numpy().copy())
        else:
            # 之后使用神经网络预测
            next_vel = self.predict_next_vel()
            
            # 更新速度
            vel.from_numpy(next_vel)
            
            # 更新位置, 使用显式欧拉
            integrate_vel_explicit_euler()

            # 强制地面碰撞
            floor_bound()
            
            # 记录新速度到缓冲区(自动移除最老的)
            self.vel_buffer.append(vel.to_numpy().copy())


def main():
    get_vertices()
    get_indices()
    init()
    
    # 加载模型
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    inference = FEMInference('best_fem_model.pth', device=device)
    
    from tiFEM import render
    render(inference.run_nn_step, init)


if __name__ == "__main__":
    main()