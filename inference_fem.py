"""使用神经网络进行推理与可视化"""

import numpy as np
import torch
import taichi as ti
from torch_geometric.data import Data, Batch

# 导入模型定义
from model.network import FEMSurrogate

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
            node_feat_dim=3,      # 速度维度
            edge_feat_dim=6       # current_len(3) + rest_len(3)
        )
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device)
        self.model.eval()

        # 预先缓存边索引
        edges_np = edge_indices.to_numpy()
        self.edge_index = torch.tensor(edges_np, dtype=torch.long).t().contiguous()
        
        self.step_count = 0  # 用于前3步的FEM模拟
        
        print(f"Model loaded from {model_path} on {device}")
    
    def predict_next_state(self):
        """使用神经网络预测下一步速度"""
        # 获取当前状态
        pos_np = pos.to_numpy()        # [N,3]
        pos0_np = pos0.to_numpy()      # [N,3]
        vel_np = vel.to_numpy()        # [N,3]
        
        # 转为torch
        pos_torch = torch.from_numpy(pos_np).float()
        pos0_torch = torch.from_numpy(pos0_np).float()
        vel_torch = torch.from_numpy(vel_np).float()
        
        # 构建图特征
        src = self.edge_index[0]   # [E]
        dst = self.edge_index[1]   # [E]

        current_len = pos_torch[dst] - pos_torch[src]     # [E,3]
        rest_len    = pos0_torch[dst] - pos0_torch[src]   # [E,3]
        edge_attr = torch.cat([current_len, rest_len], dim=-1)  # [E,6]
        
        # 创建图
        graph = Data(
            x=vel_torch,
            edge_index=self.edge_index,
            edge_attr=edge_attr.float(),
        )
        batch = Batch.from_data_list([graph]).to(self.device)
        
        # 推理
        with torch.no_grad():
            output = self.model(batch)  # (num_nodes, 3) - 预测速度变化

        output_np = output.cpu().numpy()
        next_vel = output_np  # [N,3]
        
        return next_vel
    
    def run_nn_step(self):
        """运行一步模拟"""
        self.step_count += 1
        
        if self.step_count <= 3:
            # 前3步使用FEM真实模拟
            substep_implicit()
        else:
            # 之后使用神经网络预测
            next_vel = self.predict_next_state()
            
            # 更新速度
            for i in range(n_verts):
                vel[i] = next_vel[i]
            
            # 更新位置
            for i in range(n_verts):
                new_pos = pos[i] + vel[i] * dt
                pos[i] = new_pos


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