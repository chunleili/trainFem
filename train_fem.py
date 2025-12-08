"""使用仿真生成数据并训练FEM神经网络代理模型"""


import numpy as np
import os
import pickle  # 添加pickle导入
# 解决OpenMP库冲突
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.profiler import profile, record_function, ProfilerActivity
import taichi as ti
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GeometricDataLoader

from model.network import FEMSurrogate
from utils import normalization
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
import numpy as np
import time

class FEMDataset(Dataset):
    def __init__(self, trajectories):
        self.samples = []

        # 1. 预计算 edge_index（固定的，不要每次 __getitem__ 重复做）
        from tiFEM import edge_indices
        edges_np = edge_indices.to_numpy()
        self.edge_index = torch.tensor(edges_np, dtype=torch.long).t()  # [2, E]

        # 2. 展平轨迹（保存前两步和当前步的速度）
        for traj in trajectories:
            pos0 = traj[0]['pos0']
            for t in range(3, len(traj) - 1):  # 从t=3开始,这样能获取t-2, t-1, t的速度 (traj[0]只有pos0，traj[1:]才有vel)
                self.samples.append({
                    'pos0': pos0,
                    'pos': traj[t]['pos'],
                    'vel_t_minus_2': traj[t - 2]['vel'],
                    'vel_t_minus_1': traj[t - 1]['vel'],
                    'vel_t': traj[t]['vel'],
                    'next_vel': traj[t + 1]['vel'],
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        # ---- 转 torch（一次即可，不要 numpy index）----
        pos = torch.from_numpy(s['pos']).float()        # [N,3]
        pos0 = torch.from_numpy(s['pos0']).float()      # [N,3]
        vel_t_minus_2 = torch.from_numpy(s['vel_t_minus_2']).float()  # [N,3]
        vel_t_minus_1 = torch.from_numpy(s['vel_t_minus_1']).float()  # [N,3]
        vel_t = torch.from_numpy(s['vel_t']).float()    # [N,3]
        next_vel = torch.from_numpy(s['next_vel']).float()  # [N,3]

        # ---- 直接 torch indexing，避免 .numpy() ----
        src = self.edge_index[0]   # [E]
        dst = self.edge_index[1]

        current_len = pos[dst] - pos[src]     # [E,3]
        rest_len    = pos0[dst] - pos0[src]   # [E,3]
        edge_attr = torch.cat([current_len, rest_len], dim=-1)  # [E,6]

        # 拼接三步速度作为节点特征 [N, 9]
        vel_history = torch.cat([vel_t_minus_2, vel_t_minus_1, vel_t], dim=-1)

        graph = Data(
            x=vel_history,
            edge_index=self.edge_index,
            edge_attr=edge_attr,
            y=next_vel,
        )
        return graph

class FEMTrainer:
    def __init__(self, model, device='cpu', log_dir='runs/fem_training'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=1e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        # 初始化归一化器 (参考simulator.py的设计)
        self._vel_history_normalizer = normalization.Normalizer(
            size=9, name='vel_history_normalizer', device=device
        )
        self._next_vel_normalizer = normalization.Normalizer(
            size=3, name='next_vel_normalizer', device=device
        )

        # 初始化TensorBoard
        self.writer = SummaryWriter(log_dir)
    def train_epoch(self, train_loader, noise_std=0.01):
        self.model.train()
        total_loss = 0
        num_batches = 0
        criterion = nn.MSELoss()
        
        for batch_idx, batch in enumerate(train_loader):
            batch = batch.to(self.device)
            
            # 1. Normalize输入速度历史
            node_attr_normalized = self._vel_history_normalizer(batch.x, accumulate=True)
            
            # 2. 添加velocity噪声 (参考simulator.py)
            noise = torch.randn_like(node_attr_normalized) * noise_std
            noised_node_attr = node_attr_normalized + noise
            batch.x = noised_node_attr
            
            # 3. 前向传播
            predictions = self.model(batch)  # (batch*n_verts, 3)
            
            # 4. Normalize目标速度
            targets = batch.y  # (batch*n_verts, 3)
            targets_normalized = self._next_vel_normalizer(targets, accumulate=True)
            
            # 5. MSE损失
            loss = criterion(predictions, targets_normalized)
            
            # 6. 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return {
            'loss': total_loss / num_batches if num_batches > 0 else 0.0,
        }

    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        num_batches = 0
        criterion = nn.MSELoss()
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(self.device)
                
                # Normalize输入速度历史 (不累积统计, 参考simulator.py的self.training参数)
                node_attr_normalized = self._vel_history_normalizer(batch.x, accumulate=False)
                batch.x = node_attr_normalized
                
                # 前向传播
                predictions = self.model(batch)
                
                # Normalize目标速度 (不累积统计)
                targets = batch.y
                targets_normalized = self._next_vel_normalizer(targets, accumulate=False)
                
                # MSE损失
                loss = criterion(predictions, targets_normalized)
                
                total_loss += loss.item()
                num_batches += 1
        
        return {
            'loss': total_loss / num_batches if num_batches > 0 else 0.0,
        }       
    
    def train(self, train_loader, val_loader, epochs, noise_std=0.01, enable_profiler=False):
        best_val_loss = float('inf')
        
        # 性能分析（只分析第一个 epoch）
        if enable_profiler:
            print("\n" + "="*60)
            print("Running Profiler on first epoch...")
            print("="*60)
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
                profile_memory=True,
            ) as profiler:
                self.train_epoch(train_loader, noise_std=noise_std)
            
            # 打印性能报告
            print("\n" + "="*60)
            print("PROFILER RESULTS (sorted by CPU time)")
            print("="*60)
            print(profiler.key_averages().table(sort_by="cpu_time_total", row_limit=20))
            
            print("\n" + "="*60)
            print("PROFILER RESULTS (sorted by CUDA time)")
            print("="*60)
            print(profiler.key_averages().table(sort_by="cuda_time_total", row_limit=20))
            
            print("\n" + "="*60)
            print("PROFILER RESULTS (Memory)")
            print("="*60)
            print(profiler.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))
            print("="*60 + "\n")
        
        for epoch in range(epochs):
            start_time = time.time()
            train_metrics = self.train_epoch(train_loader, noise_std=noise_std)
            val_metrics = self.validate(val_loader)
            
            # 记录到TensorBoard
            self.writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
            self.writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
            self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)
            
            self.scheduler.step(val_metrics['loss'])
            
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                # 保存模型和 normalizer 状态
                checkpoint = {
                    'model_state_dict': self.model.state_dict(),
                    'vel_history_normalizer_acc_sum': self._vel_history_normalizer._acc_sum.clone(),
                    'vel_history_normalizer_acc_sum_squared': self._vel_history_normalizer._acc_sum_squared.clone(),
                    'vel_history_normalizer_acc_count': self._vel_history_normalizer._acc_count.clone(),
                    'next_vel_normalizer_acc_sum': self._next_vel_normalizer._acc_sum.clone(),
                    'next_vel_normalizer_acc_sum_squared': self._next_vel_normalizer._acc_sum_squared.clone(),
                    'next_vel_normalizer_acc_count': self._next_vel_normalizer._acc_count.clone(),
                }
                torch.save(checkpoint, 'best_fem_model.pth')
                self.writer.add_scalar('Loss/best_val', best_val_loss, epoch)
            
            if epoch % 5 == 0:
                print(f'Epoch {epoch}/{epochs}:')
                print(f'  Train Loss: {train_metrics["loss"]:.6e}')
                print(f'  Val   Loss: {val_metrics["loss"]:.6e}')
                print(f'  LR: {self.optimizer.param_groups[0]["lr"]:.6e}')
                print(f'  Time: {time.time() - start_time:.1f}s\n')
        
        # 关闭TensorBoard writer
        self.writer.close()


# ============== 主程序 ==============
def main():
    from gen_dataset import FEMDataGenerator
    from args import args
    from tiFEM import n_verts

    # 1. 生成数据
    print("=" * 50)
    print("Step 1: Generating training data...")
    
    if args.gen:
        generator = FEMDataGenerator()
        data = generator.generate_dataset()
        generator.save_dataset(data, filename='fem_dataset.pkl')
    # 2. 划分训练集和验证集
    split_idx = int(len(data) * (1 - args.val_split))
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    
    train_dataset = FEMDataset(train_data)
    val_dataset = FEMDataset(val_data)
    
    train_loader = GeometricDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = GeometricDataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # 3. 创建模型
    print("=" * 50)
    print("Step 2: Creating neural network model...")
    model = FEMSurrogate(n_nodes=n_verts, message_passing_num=args.message_passing, hidden_dim=args.hidden_dim, node_feat_dim=9, edge_feat_dim=6)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(model)
    
    # 4. 训练
    print("=" * 50)
    print("Step 3: Training...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    trainer = FEMTrainer(model, device=device, log_dir='runs/fem_training')
    trainer.train(train_loader, val_loader, args.epochs, enable_profiler=args.enable_profiler)  # 设为 True 开启性能分析
    
    print("=" * 50)
    print("Training completed! Model saved to 'best_fem_model.pth'")
    print("TensorBoard logs saved to 'runs/fem_training'")
    print("Run 'tensorboard --logdir=runs' to view training progress")


if __name__ == "__main__":
    main()
