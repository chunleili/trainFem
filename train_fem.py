"""使用仿真生成数据并训练FEM神经网络代理模型"""


# global parameters
g_num_traj = 100  # 生成的轨迹数
g_seq_len = 500     # 每条轨迹的长度(时间步数)
g_samples_per_traj = 100  # 每条轨迹随机采样的窗口数
g_epochs = 1000       # 训练轮数
g_input_dim = 75    # 输入维度: 3*(pos12+vel12) + 3*gravity3 = 72+3=75
g_output_dim = 24   # 输出维度 (不再预测时间)
g_hidden_dim = 256 # 隐藏层维度
g_batch_size = 32  # 批大小
g_val_split = 0.2   # 验证集比例
g_load_data = False  # 是否加载已存在的数据集
g_save_data = True   # 是否保存生成的数据集


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
import taichi as ti

# 导入FEM求解器（已经初始化了Taichi）
from tiFEM import (
    get_vertices, get_indices, init, cg, 
    pos, pos0, substep_implicit, vel, force, mass, rest_matrix, rest_volume, tet_indices,
    compute_potential_energy, get_vol_err,
    n_verts, floor_bound, dt, deformation_gradient, gravity_constant, prev_pos
)

# ============== 数据生成 ==============
class FEMDataGenerator:
    """使用FEM求解器生成训练数据"""
    
    def __init__(self,):
        # 只调用一次get_vertices和get_indices来初始化
        get_vertices()
        get_indices()
        init()
        
    def generate_sequence(self):
        """生成轨迹并随机采样窗口
        """
        init()
        # 初始随机扰动
        x_array = pos.to_numpy()
        v_array = vel.to_numpy()
        x_array += np.random.randn(n_verts, 3) * 0.1
        v_array = np.random.randn(n_verts, 3) * 0.1
        for i in range(n_verts):
            pos[i] = x_array[i]
            vel[i] = v_array[i]

        states = []  # 记录所有时间步的状态
        
        # 重力向量 (0, gravity_constant, 0) 在y方向
        gravity_vec = np.array([0.0, gravity_constant, 0.0], dtype=np.float32)

        # 生成完整序列
        for t in range(g_seq_len):
            # 当前状态：pos(12) + vel(12) = 24维
            state = np.concatenate([
                pos.to_numpy().flatten(),
                vel.to_numpy().flatten(),
            ]).astype(np.float32)
            states.append(state)
            
            # 前进一步
            substep_implicit()
        
        # 随机采样窗口
        window_size = 3
        inputs = []
        outputs = []
        
        # 可采样的起始位置: 0 到 seq_len-window_size-1
        valid_starts = list(range(len(states) - window_size))
        
        # 随机采样(如果样本数大于可用数量,则全部使用)
        num_samples = min(g_samples_per_traj, len(valid_starts))
        sampled_starts = np.random.choice(valid_starts, size=num_samples, replace=False)
        
        for i in sampled_starts:
            # 输入: (state1, state2, state3, gravity*3)
            s1, s2, s3 = states[i], states[i+1], states[i+2]
            # 拼接3个状态和重力向量: 24+24+24+3 = 75
            input_window = np.concatenate([s1, s2, s3, gravity_vec])  # (75,)
            
            # 输出: 第4步状态(24) 不再预测时间
            output_state = states[i+window_size]  # (24,)
            
            inputs.append(input_window)
            outputs.append(output_state)
        
        return {
            'inputs': np.stack(inputs),   # (num_samples, 75)
            'outputs': np.stack(outputs),  # (num_samples, 24)
        }
    
    def generate_dataset(self ):
        """生成完整数据集
        """
        dataset = []
        print(f"Generating {g_num_traj} trajectories (each {g_seq_len*dt:.1f}s)...")
        print(f"Randomly sampling {g_samples_per_traj} windows per trajectory...")
        
        for traj_idx in range(g_num_traj):
            print(f"Progress: {traj_idx}/{g_num_traj}", end='\r')
            
            # 生成一条10秒轨迹并随机采样
            trajectory = self.generate_sequence()
            
            # 添加所有采样的样本
            for i in range(len(trajectory['inputs'])):
                sample = {
                    'inputs': trajectory['inputs'][i],
                    'outputs': trajectory['outputs'][i]
                }
                dataset.append(sample)
        
        print(f"\nGenerated {len(dataset)} training samples from {g_num_traj} trajectories")
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


# ============== PyTorch Dataset ==============
class FEMDataset(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        inputs = torch.from_numpy(sample['inputs'])    # (75,)
        outputs = torch.from_numpy(sample['outputs'])  # (24,)
        return inputs, outputs


# ============== 训练函数 ==============
class FEMTrainer:
    def __init__(self, model, device='cpu', log_dir='runs/fem_training'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=1e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        # 初始化TensorBoard
        self.writer = SummaryWriter(log_dir)
        
    def compute_energy_loss(self, predictions, inputs):
        """计算能量loss（自监督）"""
        g_batch_size, T, _ = predictions.shape
        energy_losses = []
        
        for i in range(g_batch_size):
            for t in range(T):
                pred_disp = predictions[i, t].detach().cpu().numpy()  # (12,)
                displacement = pred_disp.reshape(n_verts, 3)
                
                inp = inputs[i, t].cpu().numpy()  # (74,)
                # 输入格式：pos(12) + g(3) + prev_pos(12) + pos0(12) + dt(1) + vel(12) + mass(4) + F(9) + RestMatrix(9)
                cur_pos = inp[:12].reshape(n_verts, 3)
                
                # 预测后的位置
                pred_pos = cur_pos + displacement
                
                # 写入Taichi field
                for j in range(n_verts):
                    pos[j] = pred_pos[j]
                
                # 计算预测的能量
                pred_energy = compute_potential_energy()
                
                # 防止NaN：检查能量是否有效
                if not np.isnan(pred_energy) and not np.isinf(pred_energy):
                    energy_losses.append(pred_energy)  # 直接最小化能量
        
        if len(energy_losses) == 0:
            return 0.0
        
        energy_loss = np.mean(energy_losses)
        # 防止NaN和inf
        if np.isnan(energy_loss) or np.isinf(energy_loss):
            energy_loss = 0.0
        
        return energy_loss
        
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        num_batches = 0
        criterion = nn.MSELoss()
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(self.device).float()
            targets = targets.to(self.device).float()
            
            # 前向传播
            predictions = self.model(inputs)  # (batch, 24)
            
            # MSE损失
            loss = criterion(predictions, targets)
            
            # 反向传播
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
            for inputs, targets in val_loader:
                inputs = inputs.to(self.device).float()
                targets = targets.to(self.device).float()
                
                predictions = self.model(inputs)
                loss = criterion(predictions, targets)
                
                total_loss += loss.item()
                num_batches += 1
        
        return {
            'loss': total_loss / num_batches if num_batches > 0 else 0.0,
        }
    
    def train(self, train_loader, val_loader):
        best_val_loss = float('inf')
        
        for epoch in range(g_epochs):
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.validate(val_loader)
            
            # 记录到TensorBoard
            self.writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
            self.writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
            self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)
            
            self.scheduler.step(val_metrics['loss'])
            
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                torch.save(self.model.state_dict(), 'best_fem_model.pth')
                self.writer.add_scalar('Loss/best_val', best_val_loss, epoch)
            
            if epoch % 10 == 0:
                print(f'Epoch {epoch}/{g_epochs}:')
                print(f'  Train Loss: {train_metrics["loss"]:.6f}')
                print(f'  Val   Loss: {val_metrics["loss"]:.6f}')
                print(f'  LR: {self.optimizer.param_groups[0]["lr"]:.6f}')
        
        # 关闭TensorBoard writer
        self.writer.close()




class FEMSurrogate(nn.Module):
    """神经网络:用前3步预测下一步(包含重力信息)"""
    
    def __init__(self, input_dim=75, hidden_dim=256, output_dim=24):
        super(FEMSurrogate, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # 简单的全连接网络
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        # x: (batch, 75) = 3*24 + 3
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)  # (batch, 24)
        return x



# ============== 主程序 ==============
def main():
    # 1. 生成数据
    print("=" * 50)
    print("Step 1: Generating training data...")
    print(f"Number of trajectories: {g_num_traj}")
    print(f"Trajectory duration: {g_seq_len * dt} seconds ({g_seq_len} timesteps)")
    print(f"Randomly sampling {g_samples_per_traj} windows per trajectory...")
    print("Window: 3 steps + gravity -> predict 4th step")
    generator = FEMDataGenerator()
    
    # 可选：保存数据集到文件
    assert not (g_load_data and g_save_data), "Cannot both load and save data."
    if g_load_data and os.path.exists('fem_dataset.pkl'):
        print("Loading existing dataset from 'fem_dataset.pkl'...")
        data = FEMDataGenerator.load_dataset('fem_dataset.pkl')
    else:
        data = generator.generate_dataset()

    # 保存数据集到文件
    if g_save_data:
        print("Saving dataset to 'fem_dataset.pkl'...")
        generator.save_dataset(data, 'fem_dataset.pkl')
    
    # 2. 划分训练集和验证集
    split_idx = int(len(data) * (1 - g_val_split))
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    
    train_dataset = FEMDataset(train_data)
    val_dataset = FEMDataset(val_data)
    
    train_loader = DataLoader(train_dataset, batch_size=g_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=g_batch_size, shuffle=False)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # 3. 创建模型
    print("=" * 50)
    print("Step 2: Creating neural network model...")
    print(f"Input: 3 steps × 24 features + gravity(3) = {g_input_dim} dim")
    print(f"Output: next step state = {g_output_dim} dim (no time)")
    model = FEMSurrogate(input_dim=g_input_dim, hidden_dim=g_hidden_dim, output_dim=g_output_dim)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 4. 训练
    print("=" * 50)
    print("Step 3: Training...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    trainer = FEMTrainer(model, device=device, log_dir='runs/fem_training')
    trainer.train(train_loader, val_loader)
    
    print("=" * 50)
    print("Training completed! Model saved to 'best_fem_model.pth'")
    print("TensorBoard logs saved to 'runs/fem_training'")
    print("Run 'tensorboard --logdir=runs' to view training progress")


if __name__ == "__main__":
    main()
