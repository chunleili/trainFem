import numpy as np
import os
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
    pos, pos0, vel, force, mass, rest_matrix, rest_volume, tet_indices,
    compute_potential_energy, compute_volume_error,
    n_verts, floor_bound, dt
)

# ============== 数据生成 ==============
class FEMDataGenerator:
    """使用FEM求解器生成训练数据"""
    
    def __init__(self, num_samples=1000):
        self.num_samples = num_samples
        # 只调用一次get_vertices和get_indices来初始化
        get_vertices()
        get_indices()
        init()
        
    def generate_sequence(self, seq_len=20):
        """生成一个时间序列样本：记录所有物理量作为输入"""
        init()
        # 初始随机扰动（减小扰动幅度）
        x_array = pos.to_numpy()
        v_array = vel.to_numpy()
        x_array += np.random.randn(n_verts, 3) * 0.01  # 从0.1减小到0.01
        v_array = np.random.randn(n_verts, 3) * 0.1    # 从0.5减小到0.1
        for i in range(n_verts):
            pos[i] = x_array[i]
            vel[i] = v_array[i]

        inputs = []
        prev_pos_record = pos.to_numpy().copy()  # 记录前一步位置

        for t in range(seq_len):
            # 获取当前物理量
            cur_pos = pos.to_numpy().copy()        # (4, 3)
            cur_pos0 = pos0.to_numpy().copy()      # (4, 3)
            cur_vel = vel.to_numpy().copy()        # (4, 3)
            cur_mass = mass.to_numpy().copy()      # (4,)
            
            # 获取rest_matrix (只有一个cell)
            cur_rest_matrix = rest_matrix.to_numpy()[0].copy()  # (3, 3)
            
            # 计算变形梯度F = Ds @ rest_matrix
            verts = tet_indices.to_numpy()[0]  # (4,)
            # Ds矩阵：每一列是边向量 (verts[i] - verts[3])
            edge0 = cur_pos[verts[0]] - cur_pos[verts[3]]
            edge1 = cur_pos[verts[1]] - cur_pos[verts[3]]
            edge2 = cur_pos[verts[2]] - cur_pos[verts[3]]
            Ds_matrix = np.column_stack([edge0, edge1, edge2])  # (3, 3)
            
            # 检查Ds_matrix是否有效
            if np.isnan(Ds_matrix).any() or np.isinf(Ds_matrix).any():
                print(f"Warning: NaN/Inf in Ds_matrix at step {t}")
                break
            
            F_matrix = Ds_matrix @ cur_rest_matrix  # (3, 3)
            
            # 重力向量
            g = np.array([0.0, -9.8, 0.0], dtype=np.float32)  # (3,)
            
            # dt标量
            dt_scalar = float(dt)
            
            # 检查数据有效性
            if np.isnan(cur_pos).any() or np.isinf(cur_pos).any():
                print(f"Warning: NaN/Inf in pos at step {t}, reinitializing")
                init()
                break
            
            # 组合输入：pos(12) + g(3) + prev_pos(12) + pos0(12) + dt(1) + vel(12) + mass(4) + F(9) + RestMatrix(9) = 74维
            input_features = np.concatenate([
                cur_pos.flatten(),              # 12
                g,                              # 3
                prev_pos_record.flatten(),      # 12
                cur_pos0.flatten(),             # 12
                np.array([dt_scalar]),          # 1
                cur_vel.flatten(),              # 12
                cur_mass,                       # 4
                F_matrix.flatten(),             # 9
                cur_rest_matrix.flatten(),      # 9
            ]).astype(np.float32)
            
            inputs.append(input_features)
            
            # 保存当前位置作为下一步的prev_pos
            prev_pos_record = cur_pos.copy()

            # 前进一步（隐式CG）
            cg()
            floor_bound()

            # 检查新位置有效性
            new_pos = pos.to_numpy().copy()
            if np.isnan(new_pos).any() or np.isinf(new_pos).any():
                print(f"Warning: NaN/Inf in new_pos at step {t}, skipping rest")
                break
        
        # 如果序列太短，返回None
        if len(inputs) < seq_len // 2:
            return None

        return {
            'inputs': np.stack(inputs),  # (T, 74)
        }
    
    def generate_dataset(self, seq_len=20):
        """生成完整数据集"""
        dataset = []
        print(f"Generating {self.num_samples} training samples...")
        attempts = 0
        max_attempts = self.num_samples * 3  # 最多尝试3倍样本数
        
        while len(dataset) < self.num_samples and attempts < max_attempts:
            if len(dataset) % 10 == 0:
                print(f"Progress: {len(dataset)}/{self.num_samples}")
            sample = self.generate_sequence(seq_len)
            if sample is not None:  # 只添加有效样本
                dataset.append(sample)
            attempts += 1
        
        if len(dataset) < self.num_samples:
            print(f"Warning: Only generated {len(dataset)} valid samples out of {self.num_samples} requested")
        
        return dataset


# ============== PyTorch Dataset ==============
class FEMDataset(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        inputs = torch.from_numpy(sample['inputs'])              # (T, 74)
        return inputs


# ============== 神经网络模型 ==============
class FEMSurrogate(nn.Module):
    """神经网络预测delta_x"""
    
    def __init__(self, input_dim=74, hidden_dim=256, output_dim=12, num_layers=2):
        super(FEMSurrogate, self).__init__()
        # 输入：pos(12) + g(3) + prev_pos(12) + pos0(12) + dt(1) + vel(12) + mass(4) + F(9) + RestMatrix(9) = 74维
        # 输出：delta_x位移12维
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1,
            bidirectional=False,
        )
        # 每个时间步输出12维（位移）
        self.head = nn.Linear(hidden_dim, self.output_dim)
        
    def forward(self, x):
        # x: (batch, T, 74)
        y, _ = self.lstm(x)
        out_seq = self.head(y)  # (batch, T, 12)
        return out_seq


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
        
        # 初始化FEM用于计算能量
        get_vertices()
        
    def compute_energy_loss(self, predictions, inputs):
        """计算能量loss（自监督）"""
        batch_size, T, _ = predictions.shape
        energy_losses = []
        
        for i in range(batch_size):
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
        total_energy_loss = 0
        num_batches = 0
        
        for batch_idx, inputs in enumerate(train_loader):
            inputs = inputs.to(self.device)
            
            # 检查输入数据
            if torch.isnan(inputs).any() or torch.isinf(inputs).any():
                print(f"Warning: NaN/Inf in inputs at batch {batch_idx}, skipping")
                continue
            
            # 前向传播
            outputs = self.model(inputs)  # (batch, T, 12)
            
            # 检查模型输出
            if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                print(f"Warning: NaN/Inf in model outputs at batch {batch_idx}")
                print(f"  Outputs stats: min={outputs.min().item()}, max={outputs.max().item()}, mean={outputs.mean().item()}")
                continue
            
            # 能量loss（自监督）
            energy_loss = self.compute_energy_loss(outputs, inputs)
            
            # 检查loss
            if np.isnan(energy_loss) or np.isinf(energy_loss) or energy_loss == 0.0:
                print(f"Warning: energy_loss is {energy_loss} at batch {batch_idx}, skipping")
                continue
            
            # 反向传播
            self.optimizer.zero_grad()
            
            # 将numpy loss转为torch tensor
            loss_tensor = torch.tensor(energy_loss, dtype=torch.float32, requires_grad=True, device=self.device)
            
            loss_tensor.backward()
            
            # 梯度裁剪防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_energy_loss += energy_loss
            num_batches += 1
        
        if num_batches == 0:
            print("Warning: All batches were skipped due to NaN/Inf!")
            return {
                'energy': 0.0,
            }
        
        return {
            'energy': total_energy_loss / num_batches,
        }
    
    def validate(self, val_loader):
        self.model.eval()
        total_energy_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for inputs in val_loader:
                inputs = inputs.to(self.device)
                
                # 检查输入数据
                if torch.isnan(inputs).any() or torch.isinf(inputs).any():
                    continue
                
                outputs = self.model(inputs)  # (batch, T, 12)
                
                # 检查模型输出
                if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                    continue
                
                energy_loss = self.compute_energy_loss(outputs, inputs)
                
                if np.isnan(energy_loss) or np.isinf(energy_loss) or energy_loss == 0.0:
                    continue
                
                total_energy_loss += energy_loss
                num_batches += 1
        
        if num_batches == 0:
            return {
                'energy': 0.0,
            }
        
        return {
            'energy': total_energy_loss / num_batches,
        }
    
    def train(self, train_loader, val_loader, epochs=100):
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.validate(val_loader)
            
            # 记录到TensorBoard
            self.writer.add_scalar('Loss/train_energy', train_metrics['energy'], epoch)
            self.writer.add_scalar('Loss/val_energy', val_metrics['energy'], epoch)
            self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)
            
            self.scheduler.step(val_metrics['energy'])
            
            if val_metrics['energy'] < best_val_loss and val_metrics['energy'] > 0:
                best_val_loss = val_metrics['energy']
                torch.save(self.model.state_dict(), 'best_fem_model.pth')
                self.writer.add_scalar('Loss/best_val_energy', best_val_loss, epoch)
            
            if epoch % 10 == 0:
                print(f'Epoch {epoch}/{epochs}:')
                print(f'  Train Energy: {train_metrics["energy"]:.6f}')
                print(f'  Val   Energy: {val_metrics["energy"]:.6f}')
                print(f'  LR: {self.optimizer.param_groups[0]["lr"]:.6f}')
        
        # 关闭TensorBoard writer
        self.writer.close()


# ============== 主程序 ==============
def main():
    # 1. 生成数据
    print("=" * 50)
    print("Step 1: Generating training data...")
    generator = FEMDataGenerator(num_samples=100)
    data = generator.generate_dataset(seq_len=10)
    
    # 2. 划分训练集和验证集
    split_idx = int(len(data) * 0.8)
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    
    train_dataset = FEMDataset(train_data)
    val_dataset = FEMDataset(val_data)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # 3. 创建模型
    print("=" * 50)
    print("Step 2: Creating neural network model...")
    model = FEMSurrogate(input_dim=74, hidden_dim=256, output_dim=12)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 4. 训练
    print("=" * 50)
    print("Step 3: Training...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    trainer = FEMTrainer(model, device=device, log_dir='runs/fem_training')
    trainer.train(train_loader, val_loader, epochs=50)
    
    print("=" * 50)
    print("Training completed! Model saved to 'best_fem_model.pth'")
    print("TensorBoard logs saved to 'runs/fem_training'")
    print("Run 'tensorboard --logdir=runs' to view training progress")


if __name__ == "__main__":
    main()
