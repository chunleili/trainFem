import numpy as np
import torch
import taichi as ti

# 导入模型定义
from train_fem import FEMSurrogate

# 导入FEM求解器（已经初始化了Taichi）
from tiFEM import (
    get_vertices, init, cg,
    pos, pos0, vel, force, mass, tet_indices, tri_indices, edge_indices,
    n_verts,
    compute_potential_energy, compute_volume_error,
    get_indices, floor_bound
)



class FEMInference:
    """使用训练好的神经网络进行FEM推理"""
    
    def __init__(self, model_path='best_fem_model.pth', device='cpu'):
        self.device = device
        self.model = FEMSurrogate(input_dim=74, hidden_dim=256, output_dim=12)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device)
        self.model.eval()
        
        print(f"Model loaded from {model_path} on {device}")
    
    def predict(self, current_pos, initial_pos, current_vel, prev_pos, dt_value=0.01):
        """
        预测一步位移
        
        Args:
            current_pos: (n_verts, 3) 当前位置
            initial_pos: (n_verts, 3) 初始位置
            current_vel: (n_verts, 3) 当前速度
            prev_pos: (n_verts, 3) 前一步位置
            dt_value: float 时间步长
            
        Returns:
            displacement: (n_verts, 3) 预测的位移
        """
        # 获取质量
        cur_mass = mass.to_numpy().copy()  # (4,)
        
        # 获取rest_matrix
        from tiFEM import rest_matrix
        cur_rest_matrix = rest_matrix.to_numpy()[0].copy()  # (3, 3)
        
        # 计算变形梯度F
        verts = tet_indices.to_numpy()[0]  # (4,)
        # Ds矩阵：每一列是边向量 (verts[i] - verts[3])
        edge0 = current_pos[verts[0]] - current_pos[verts[3]]
        edge1 = current_pos[verts[1]] - current_pos[verts[3]]
        edge2 = current_pos[verts[2]] - current_pos[verts[3]]
        Ds_matrix = np.column_stack([edge0, edge1, edge2])  # (3, 3)
        F_matrix = Ds_matrix @ cur_rest_matrix  # (3, 3)
        
        # 重力向量
        g = np.array([0.0, -9.8, 0.0], dtype=np.float32)
        
        # 准备输入：pos(12) + g(3) + prev_pos(12) + pos0(12) + dt(1) + vel(12) + mass(4) + F(9) + RestMatrix(9) = 74
        input_features = np.concatenate([
            current_pos.flatten(),       # 12
            g,                           # 3
            prev_pos.flatten(),          # 12
            initial_pos.flatten(),       # 12
            np.array([dt_value]),        # 1
            current_vel.flatten(),       # 12
            cur_mass,                    # 4
            F_matrix.flatten(),          # 9
            cur_rest_matrix.flatten(),   # 9
        ]).astype(np.float32)
        
        # 添加时间维度：(1, 1, 74) 表示 (batch, T, 74)
        input_tensor = torch.from_numpy(input_features).unsqueeze(0).unsqueeze(0).to(self.device)
        
        # 推理
        with torch.no_grad():
            output = self.model(input_tensor)
        
        output_np = output.cpu().numpy()[0, 0]  # 取第一个batch和第一个时间步
        
        # 解析输出为位移
        displacement = output_np.reshape(n_verts, 3)
        
        return displacement
    
    def run_simulation_step(self, prev_pos_record, dt_value=0.01):
        """运行一步模拟（使用神经网络）"""
        # 获取当前状态
        current_pos = pos.to_numpy().copy()
        initial_pos = pos0.to_numpy().copy()
        current_vel = vel.to_numpy().copy()
        
        # 预测位移
        displacement = self.predict(current_pos, initial_pos, current_vel, prev_pos_record, dt_value)
        
        # 更新位置
        new_pos = current_pos + displacement
        
        # 写回Taichi field
        for i in range(n_verts):
            pos[i] = new_pos[i]


def main():
    """使用神经网络进行推理与可视化"""
    get_vertices()
    get_indices()
    init()
    
    # 加载模型
    inference = FEMInference('best_fem_model.pth')
    
    # 追踪前一步位置
    prev_pos_record = pos.to_numpy().copy()
    dt_value = 0.01
    
    # 投影函数（与tiFEM相同）
    def T(a):
        phi, theta = np.radians(28), np.radians(32)
        a = a - 0.2
        x, y, z = a[:, 0], a[:, 1], a[:, 2]
        c, s = np.cos(phi), np.sin(phi)
        C, S = np.cos(theta), np.sin(theta)
        x, z = x * c + z * s, z * c - x * s
        u, v = x, y * C + z * S
        return np.array([u, v]).swapaxes(0, 1) + 0.5
    
    # 使用GUI可视化
    gui = ti.GUI("Neural FEM Inference", res=(800, 600))
    
    frame = 0
    neural_energy = 0.0
    fem_energy = 0.0
    vol_err = 0.0
    
    while gui.running:
        # 保存当前状态用于对比
        state_pos = pos.to_numpy().copy()
        state_vel = vel.to_numpy().copy()
        
        # 使用神经网络求解一步
        inference.run_simulation_step(prev_pos_record, dt_value)
        neural_pos = pos.to_numpy().copy()
        neural_energy = compute_potential_energy()
        vol_err = compute_volume_error()
        
        # 更新prev_pos
        prev_pos_record = state_pos.copy()
        
        
        # 地板碰撞
        for i in range(n_verts):
            if pos[i].y < 0:
                pos[i] = [pos[i].x, 0.0, pos[i].z]
                if vel[i].y < 0:
                    vel[i] = [vel[i].x, 0.0, vel[i].z]
        
        if gui.get_event(ti.GUI.PRESS):
            if gui.event.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
                break
        if gui.is_pressed("r"):
            init()
            frame = 0
            prev_pos_record = pos.to_numpy().copy()
            print("Simulation reset")
        
        # 渲染（与tiFEM相同的风格）
        gui.clear(0xFFFFFF)
        
        # 绘制地面网格
        grid_size = 10
        grid_step = 0.2
        for i in range(grid_size + 1):
            x = i * grid_step - 1
            # x方向的线
            p1 = T(np.array([[x, 0, -1]]) / 3)[0]
            p2 = T(np.array([[x, 0, 1]]) / 3)[0]
            gui.line(p1, p2, radius=1, color=0xCCCCCC)
            
            z = i * grid_step - 1
            # z方向的线
            p1 = T(np.array([[-1, 0, z]]) / 3)[0]
            p2 = T(np.array([[1, 0, z]]) / 3)[0]
            gui.line(p1, p2, radius=1, color=0xCCCCCC)
        
        # 投影顶点
        vertices_2d = T(pos.to_numpy() / 3)
        
        # 绘制三角形面（半透明）
        for i in range(4):
            face = tri_indices.to_numpy()[i]
            triangle = vertices_2d[face]
            gui.triangle(triangle[0], triangle[1], triangle[2], color=0xEECCAA)
        
        # 绘制线框边
        for i in range(6):
            edge = edge_indices.to_numpy()[i]
            gui.line(vertices_2d[edge[0]], vertices_2d[edge[1]], radius=2, color=0x0000FF)
        
        # 绘制顶点
        gui.circles(vertices_2d, radius=5, color=0xBA543A)
        
        # 显示信息
        gui.text(f"Neural Energy: {neural_energy:.4f}", pos=(0.02, 0.95), color=0x000000)
        gui.text(f"FEM Energy: {fem_energy:.4f}", pos=(0.02, 0.90), color=0x000000)
        gui.text(f"Volume Error: {vol_err:.2e}", pos=(0.02, 0.85), color=0x000000)
        gui.text(f"Frame: {frame}", pos=(0.02, 0.80), color=0x000000)
        
        gui.show()
        frame += 1


if __name__ == "__main__":
    main()
