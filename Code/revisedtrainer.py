import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os


class TwoStageTrainer:
    """两阶段训练器"""

    def __init__(self, model, train_loader, val_loader, test_loader,
                 device='cuda', lr=1e-3, weight_decay=1e-5,scaler=None):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.scaler=scaler

        # 优化器
        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )

        # 损失函数
        self.criterion = nn.MSELoss()

        # 训练历史
        self.train_history = {'stage_a': [], 'stage_b': []}
        self.val_history = {'stage_a': [], 'stage_b': []}

    def freeze_interaction_modules(self):
        """冻结交互相关模块"""
        for name, param in self.model.named_parameters():
            if 'interaction' in name or 'edge_scoring' in name:
                param.requires_grad = False
                print(f"Frozen: {name}")

    def unfreeze_all_modules(self):
        """解冻所有模块"""
        for name, param in self.model.named_parameters():
            param.requires_grad = True
            print(f"Unfrozen: {name}")

    def train_epoch(self, stage='A'):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f'Training Stage {stage}')
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            y_hist = data[:, :, -1]  # (B, L_in)
            aux_target = y_hist[:, 1:].squeeze(-1) # (B, L_in-1)   shift by 1

            #print(torch.isnan(data).any(), torch.isnan(target).any())
            #print(torch.isinf(data).any(), torch.isinf(target).any())

            self.optimizer.zero_grad()

            # 前向传播
            outputs = self.model(data)
            predictions = outputs['prediction']

            # 计算损失
            if stage == 'A':
                # Stage A: 只训练主效应
                loss = self.criterion(predictions, target)
            else:
                # Stage B: 训练完整模型
                loss = self.criterion(predictions, target)

                # 添加正则化损失
                interactions = outputs['interactions']
                sparsity_loss = torch.mean(torch.abs(interactions))

                # 时间平滑损失 (只在有多个时间步时计算)
                if interactions.size(1) > 1:
                    smoothness_loss = torch.mean(torch.abs(
                        interactions[:, 1:, :, :] - interactions[:, :-1, :, :]
                    ))
                else:
                    smoothness_loss = torch.tensor(0.0, device=interactions.device)

                # contrib_main, contrib_int: (B, H)
                # 正交 loss: dot product squared, 越小越正交
                #orth_loss = torch.mean((torch.sum(contrib_main * contrib_int, dim=-1)) ** 2)

                # 对每个预测步正交
                orth_loss_per_step = (outputs['contrib_main'] * outputs['contrib_int_']) ** 2  # (B, H)
                orth_loss = torch.mean(orth_loss_per_step)  # scalar

                loss_aux = self.criterion(outputs['aux_preds'], aux_target)

                loss = loss + 1e-3 * sparsity_loss + 1e-3 * smoothness_loss*1e-3*orth_loss+1e-2*loss_aux

            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})

        return total_loss / num_batches

    def validate(self, stage='A'):
        """验证"""
        self.model.eval()
        total_loss = 0.0
        predictions_list = []
        targets_list = []

        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)

                outputs = self.model(data)
                predictions = outputs['prediction']

                loss = self.criterion(predictions, target)
                total_loss += loss.item()

                predictions_list.append(predictions.cpu().numpy())
                targets_list.append(target.cpu().numpy())

        # 计算指标
        predictions_all = np.concatenate(predictions_list, axis=0)
        targets_all = np.concatenate(targets_list, axis=0)

        rmse = np.sqrt(mean_squared_error(targets_all.flatten(), predictions_all.flatten()))
        mae = mean_absolute_error(targets_all.flatten(), predictions_all.flatten())

        return total_loss / len(self.val_loader), rmse, mae

    def train_stage_a(self, epochs=20, patience=10):
        """训练阶段A: 主效应"""
        print("=" * 50)
        print("Stage A: Training Main Effects Only")
        print("=" * 50)

        # 冻结交互模块
        self.freeze_interaction_modules()

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")

            # 训练
            train_loss = self.train_epoch(stage='A')

            # 验证
            val_loss, rmse, mae = self.validate(stage='A')

            # 学习率调度
            self.scheduler.step(val_loss)

            # 记录历史
            self.train_history['stage_a'].append(train_loss)
            self.val_history['stage_a'].append(val_loss)

            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")

            # 早停
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # 保存最佳模型
                torch.save(self.model.state_dict(), 'best_stage_a.pth')
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        # 加载最佳模型
        self.model.load_state_dict(torch.load('best_stage_a.pth'))
        print("Stage A completed!")

    def train_stage_b(self, epochs=100, patience=20):
        """训练阶段B: 完整模型"""
        print("=" * 50)
        print("Stage B: Training Full Model")
        print("=" * 50)

        # 解冻所有模块
        self.unfreeze_all_modules()

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")

            # 训练
            train_loss = self.train_epoch(stage='B')

            # 验证
            val_loss, rmse, mae = self.validate(stage='B')

            # 学习率调度
            self.scheduler.step(val_loss)

            # 记录历史
            self.train_history['stage_b'].append(train_loss)
            self.val_history['stage_b'].append(val_loss)

            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")

            # 早停
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # 保存最佳模型
                torch.save(self.model.state_dict(), 'best_stage_b.pth')
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        # 加载最佳模型
        self.model.load_state_dict(torch.load('best_stage_b.pth'))
        print("Stage B completed!")

    def train(self, stage_a_epochs=20, stage_b_epochs=100):
        """完整的两阶段训练"""
        # Stage A
        self.train_stage_a(epochs=stage_a_epochs)

        # Stage B
        self.train_stage_b(epochs=stage_b_epochs)

        print("Training completed!")

    def evaluate(self):
        """评估模型"""
        self.model.eval()
        predictions_list = []
        targets_list = []
        main_effects_list = []  #batch, time, N
        interactions_list = [] #batch time N N
        contrib_main_list = []  # (B, H)
        contrib_int_list = [] # (B, H)

        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)

                outputs = self.model(data)
                predictions = outputs['prediction']
                main_effects = outputs['main_effects']
                interactions = outputs['interactions']
                contrib_main = outputs['contrib_main']
                contrib_int_ = outputs['contrib_int_']

                predictions_list.append(predictions.cpu().numpy())
                targets_list.append(target.cpu().numpy())
                main_effects_list.append(main_effects.cpu().numpy())
                interactions_list.append(interactions.cpu().numpy())
                contrib_main_list.append(contrib_main.cpu().numpy())
                contrib_int_list.append(contrib_int_.cpu().numpy())

        # 计算指标
        predictions_all = np.concatenate(predictions_list, axis=0)  #shape Number, H
        targets_all = np.concatenate(targets_list, axis=0)
        main_effects_all = np.concatenate(main_effects_list, axis=0) # Number ,T, feature
        interactions_all = np.concatenate(interactions_list, axis=0)
        contrib_main_all = np.concatenate(contrib_main_list, axis=0) #number, horizon
        contrib_int_all = np.concatenate(contrib_int_list, axis=0)

        '''
        #反归一化
        if self.scaler is not None:
            # 假设 scaler 有 inverse_transform 方法
            if predictions_all.ndim == 2:  # (B,H)
                predictions_all = self.scaler.inverse_transform(predictions_all)
                targets_all = self.scaler.inverse_transform(targets_all)
        '''
        rmse = np.sqrt(mean_squared_error(targets_all.flatten(), predictions_all.flatten()))
        mae = mean_absolute_error(targets_all.flatten(), predictions_all.flatten())
        mse= mean_squared_error(targets_all.flatten(), predictions_all.flatten())

        print(f"\nTest Results:")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"MSE: {mse:.4f}")

        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'predictions': predictions_all,
            'targets': targets_all,
            'main_effects': main_effects_all,
            'interactions': interactions_all,
            'contrib_main': contrib_main_all,
            'contrib_int_': contrib_int_all
        }

    def plot_training_history(self,save_path, model_name):
        """绘制训练历史"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Stage A
        if self.train_history['stage_a']:
            ax1.plot(self.train_history['stage_a'], label='Train Loss', color='blue')
            ax1.plot(self.val_history['stage_a'], label='Val Loss', color='red')
            ax1.set_title('Stage A: Main Effects Training')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()
            ax1.grid(True)

        # Stage B
        if self.train_history['stage_b']:
            ax2.plot(self.train_history['stage_b'], label='Train Loss', color='blue')
            ax2.plot(self.val_history['stage_b'], label='Val Loss', color='red')
            ax2.set_title('Stage B: Full Model Training')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Loss')
            ax2.legend()
            ax2.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(save_path, model_name+'_training_history.png'), dpi=300, bbox_inches='tight')

        plt.show()

    def plot_interactions(self, interactions, save_path='interactions.png'):
        """绘制交互强度矩阵"""
        # 计算平均交互强度
        avg_interactions = np.mean(np.abs(interactions), axis=(0, 1))  # (N, N)

        plt.figure(figsize=(10, 8))
        sns.heatmap(avg_interactions, annot=True, fmt='.3f', cmap='RdBu_r',
                    center=0, square=True)
        plt.title('Average Interaction Strengths')
        plt.xlabel('Variable Index')
        plt.ylabel('Variable Index')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

        return avg_interactions

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import os

class SequenceTrainer:
    """
    通用序列预测训练器，适用于 LSTM / GRU / Transformer
    不涉及 main/interaction
    """
    def __init__(self, model,model_name, train_loader, val_loader, test_loader,save_dir,
                 device='cuda', lr=1e-3, weight_decay=1e-5, scaler=None):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.scaler = scaler
        self.model_name = model_name
        self.save_dir=save_dir

        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )

        self.criterion = nn.MSELoss()
        self.train_history = []
        self.val_history = []

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc='Training')
        for data, target in pbar:
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)  # output: (B, H)
            loss = self.criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})

        return total_loss / num_batches

    def validate(self):
        self.model.eval()
        predictions_list = []
        targets_list = []
        total_loss = 0.0

        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                total_loss += loss.item()

                predictions_list.append(output.cpu().numpy())
                targets_list.append(target.cpu().numpy())

        predictions_all = np.concatenate(predictions_list, axis=0)
        targets_all = np.concatenate(targets_list, axis=0)

        rmse = np.sqrt(mean_squared_error(targets_all.flatten(), predictions_all.flatten()))
        mae = mean_absolute_error(targets_all.flatten(), predictions_all.flatten())

        return total_loss / len(self.val_loader), rmse, mae

    def train(self, epochs=100, early_stop_patience=10, save_path=None):
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            train_loss = self.train_epoch()
            val_loss, rmse, mae = self.validate()
            self.scheduler.step(val_loss)

            self.train_history.append(train_loss)
            self.val_history.append(val_loss)

            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                if save_path:
                    torch.save(self.model.state_dict(), save_path)
            else:
                patience_counter += 1

            if patience_counter >= early_stop_patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        if save_path:
            self.model.load_state_dict(torch.load(save_path))

    def evaluate(self,save_results=True):
        self.model.eval()
        predictions_list = []
        targets_list = []
        inputs_list = []  # 保存输入

        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                predictions_list.append(output.cpu().numpy())
                targets_list.append(target.cpu().numpy())
                inputs_list.append(data.cpu().numpy())

        predictions_all = np.concatenate(predictions_list, axis=0)
        targets_all = np.concatenate(targets_list, axis=0)
        inputs_all = np.concatenate(inputs_list, axis=0)


        rmse = np.sqrt(mean_squared_error(targets_all.flatten(), predictions_all.flatten()))
        mae = mean_absolute_error(targets_all.flatten(), predictions_all.flatten())
        mse = mean_squared_error(targets_all.flatten(), predictions_all.flatten())

        if save_results and "transformer" in self.model_name.lower():
            np.save(os.path.join(self.save_dir, f"X_test.npy"), inputs_all)
            np.save(os.path.join(self.save_dir, f"Y_test.npy"), targets_all)
            np.save(os.path.join(self.save_dir, f"preds.npy"), predictions_all)
            torch.save(self.model, os.path.join(self.save_dir, f"{self.model_name}_model.pt"))
            print(f"Saved test data and model to {self.save_dir}")

        print(f"\nTest Results:")
        print(f"MSE: {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")

        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'predictions': predictions_all,
            'targets': targets_all
        }

    def plot_training_history(self,save_path, model_name):
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_history, label='Train Loss')
        plt.plot(self.val_history, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training History')
        plt.legend()
        plt.grid(True)
        if save_path:
            plt.savefig(os.path.join(save_path, model_name + '_training_history.png'), dpi=300, bbox_inches='tight')

        plt.show()
