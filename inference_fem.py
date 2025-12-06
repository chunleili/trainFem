"""使用神经网络进行推理与可视化"""

import numpy as np
import torch
import taichi as ti
from collections import deque

# 导入训练脚本中的模型定义
import sys
sys.path.append('.')
from train_fem import FEMSurrogate

# 导入FEM求解器（已经初始化了Taichi）
from tiFEM import (
    get_vertices, init, cg,
    pos, pos0, substep_implicit, vel, force, mass, tet_indices, tri_indices, edge_indices,
    n_verts, dt,
    compute_potential_energy, get_vol_err,
    get_indices, floor_bound, gravity_constant
)


class FEMInference:
    """使用训练好的神经网络进行FEM推理"""
    
    def __init__(self, model_path='best_fem_model.pth', device='cuda'):
        self.device = device
        # 使用训练时的维度: 输入75维, 输出24维
        self.model = FEMSurrogate(input_dim=75, hidden_dim=256, output_dim=24)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device)
        self.model.eval()
        
        # 维护3步历史状态缓冲区
        self.state_buffer = deque(maxlen=3)  # 存储(state, time)
        self.current_time = 0.0
        
        print(f"Model loaded from {model_path} on {device}")
    
    def add_current_state(self):
        """将当前状态添加到历史缓冲区"""
        # 状态格式: pos(12) + vel(12) = 24维
        state = np.concatenate([
            pos.to_numpy().flatten(),
            vel.to_numpy().flatten(),
        ]).astype(np.float32)
        
        self.state_buffer.append((state, self.current_time))
        self.current_time += dt
    
    def predict_next_state(self):
        """
        使用3步历史预测下一步状态
        
        Returns:
            next_pos: (n_verts, 3) 预测的下一步位置
            next_vel: (n_verts, 3) 预测的下一步速度
        """
        if len(self.state_buffer) < 3:
            raise ValueError(f"Need 3 history states, but only have {len(self.state_buffer)}")
        
        # 构建输入: (state1, time1, state2, time2, state3, time3)
        states = list(self.state_buffer)
        s1, t1 = states[0]
        s2, t2 = states[1]
        s3, t3 = states[2]
        
        input_window = np.concatenate([s1, [t1], s2, [t2], s3, [t3]]).astype(np.float32)  # (75,)
        
        # 转为张量
        input_tensor = torch.from_numpy(input_window).unsqueeze(0).to(self.device)  # (1, 75)
        
        # 推理
        with torch.no_grad():
            output = self.model(input_tensor)  # (1, 24)
        
        output_np = output.cpu().numpy()[0]  # (24,)
        
        # 解析输出: pos(12) + vel(12)
        next_pos = output_np[:12].reshape(n_verts, 3)
        next_vel = output_np[12:].reshape(n_verts, 3)

        pos.from_numpy(next_pos)
        vel.from_numpy(next_vel)
        
        return next_pos, next_vel
    
    def run_nn_step(self):
        """运行一步模拟（使用神经网络预测,除了前三步）"""
        if len(self.state_buffer) < 3:
            substep_implicit()
            self.add_current_state()
            return
            
        # 预测下一步状态
        self.predict_next_state()
        
        # 添加新状态到缓冲区(自动移除最老的)
        self.add_current_state()

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