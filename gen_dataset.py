# 导入FEM求解器（已经初始化了Taichi）
from tiFEM import (
    get_vertices, get_indices, init, 
    pos,  substep_implicit, vel,
    n_verts,  dt, gravity_constant, pos0
)

import numpy as np
from args import args
import pickle


# ============== 数据生成 ==============
class FEMDataGenerator:
    '''使用FEM求解器生成训练数据
    ::
        if args.gen:
            generator = FEMDataGenerator()
            data = generator.generate_dataset()
            generator.save_dataset(data, filename='fem_dataset.pkl')
        else:
            data = FEMDataGenerator.load_dataset('fem_dataset.pkl')'''
    
    def __init__(self,):
        # 只调用一次get_vertices和get_indices来初始化
        get_vertices()
        get_indices()
        init()
        
    def generate_sequence(self):
        """通过模拟生成一条轨迹"""
        init()
        # 初始随机扰动
        x_array = pos.to_numpy()
        v_array = vel.to_numpy()
        x_array += np.random.randn(n_verts, 3) * 0.1
        v_array = np.random.randn(n_verts, 3) * 0.1
        pos.from_numpy(x_array)
        vel.from_numpy(v_array)

        states = []  # 记录所有时间步的状态 (速度)
        state_global = {
            "pos0": pos0.to_numpy(),
        }
        # states0 保存的是global信息
        states.append(state_global) 

        # 生成完整序列
        for t in range(args.seq_len):
            # 每个时刻系统状态
            state ={
                "vel": vel.to_numpy(),
                "pos": pos.to_numpy(),
            }
            states.append(state)
            # 模拟前进一步
            substep_implicit()
        return states
    
    def generate_dataset(self):
        """生成完整数据集
        """
        dataset = []
        print(f"Generating {args.num_traj} trajectories (each {args.seq_len*dt:.1f}s)...")
        for traj_idx in range(args.num_traj):
            print(f"Progress: {traj_idx}/{args.num_traj}", end='\r')
            trajectory = self.generate_sequence()
            dataset.append(trajectory)
        
        print(f"\nGenerated {args.num_traj} trajectories")
        return dataset

    def save_dataset(self, dataset, filename='fem_dataset.pkl'):
        """保存数据集到文件"""
        with open(filename, 'wb') as f:
            pickle.dump(dataset, f)
        print(f"Dataset saved to {filename}")
    
    @staticmethod
    def load_dataset(filename='fem_dataset.pkl'):
        """从文件加载数据集"""
        with open(filename, 'rb') as f:
            dataset = pickle.load(f)
        print(f"Dataset loaded from {filename}, size: {len(dataset)}")
        return dataset



