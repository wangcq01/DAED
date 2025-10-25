import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import os

class SingleStageTrainer:
    """
    单阶段训练器：适用于主效应 + 交互 + 时间聚合模型
    """
    def __init__(self, model, train_loader, val_loader, test_loader,
                 device='cuda', lr=1e-3, weight_decay=1e-5, scaler=None):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.scaler = scaler

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

            outputs = self.model(data)  # dict，包含 'prediction', 'main_effects', 'interactions', 'aux_preds' 等
            predictions = outputs['prediction']
            pred_orig=outputs['preds_orig']# (B, H)

            # 可选正则项
            interactions = outputs['interactions']  # (B, T, N, N)
            sparsity_loss=outputs['sparsity_loss']
            #sparsity_loss = torch.mean(torch.abs(interactions))
            if interactions.size(1) > 1:
                smoothness_loss = torch.mean(torch.abs(interactions[:, 1:] - interactions[:, :-1]))
            else:
                smoothness_loss = torch.tensor(0.0, device=interactions.device)

            # 贡献正交损失
            contrib_main = outputs['contrib_main_htn']  # (B, H)
            contrib_int_ = outputs['contrib_int_htp']  # (B, H)
            #orth_loss = torch.mean((contrib_main * contrib_int_) ** 2)

            lambda_main = 0.01  # 主效应正则化系数
            lambda_int = 0.01  # 交互效应正则化系数
            l1_main = torch.mean(torch.abs(contrib_main))
            l1_int = torch.mean(torch.abs(contrib_int_))

            # 总损失
            #loss = self.criterion(predictions, target) + 1e-2 * sparsity_loss
            # 总损失
            #loss = self.criterion(predictions, target)+ 1e-2 * sparsity_loss + 1e-2 * smoothness_loss
            #loss = loss + 1e-3 * sparsity_loss + 1e-3 * smoothness_loss
            L_fidelity = self.criterion(predictions, pred_orig)  # keep masked close to original

            loss = self.criterion(predictions, target)+1e-2*L_fidelity+ 1e-2 * sparsity_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})

        return total_loss / num_batches

    def validate(self):
        self.model.eval()
        total_loss = 0.0
        predictions_list, targets_list = [], []

        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                outputs = self.model(data)
                predictions = outputs['prediction']

                loss = self.criterion(predictions, target)
                total_loss += loss.item()

                predictions_list.append(predictions.cpu().numpy())
                targets_list.append(target.cpu().numpy())

        predictions_all = np.concatenate(predictions_list, axis=0)
        targets_all = np.concatenate(targets_list, axis=0)

        rmse = np.sqrt(mean_squared_error(targets_all.flatten(), predictions_all.flatten()))
        mae = mean_absolute_error(targets_all.flatten(), predictions_all.flatten())

        return total_loss / len(self.val_loader), rmse, mae

    def train(self, epochs=50, early_stop_patience=10, save_path=None):
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

    def evaluate(self):
        self.model.eval()
        predictions_list, targets_list = [], []
        main_effects_list, interactions_list = [], []
        contrib_main_list, contrib_int_list = [], []
        time_weight_list=[]

        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                outputs = self.model(data)
                predictions_list.append(outputs['prediction'].cpu().numpy())
                targets_list.append(target.cpu().numpy())
                main_effects_list.append(outputs['main_effects'].cpu().numpy())
                interactions_list.append(outputs['interactions'].cpu().numpy())
                contrib_main_list.append(outputs['contrib_main_htn'].cpu().numpy())
                contrib_int_list.append(outputs['contrib_int_htp'].cpu().numpy())
                time_weight_list.append(outputs['time_weights'].cpu().numpy())

            predictions_all = np.concatenate(predictions_list, axis=0)
            targets_all = np.concatenate(targets_list, axis=0)
            main_effects_all = np.concatenate(main_effects_list, axis=0)
            interactions_all = np.concatenate(interactions_list, axis=0)
            contrib_main_all = np.concatenate(contrib_main_list, axis=0)
            contrib_int_all = np.concatenate(contrib_int_list, axis=0)
            time_weight_all=np.concatenate(time_weight_list, axis=0)

            rmse = np.sqrt(mean_squared_error(targets_all.flatten(), predictions_all.flatten()))
            mae = mean_absolute_error(targets_all.flatten(), predictions_all.flatten())
            mse = mean_squared_error(targets_all.flatten(), predictions_all.flatten())

            print(f"\nTest Results:")
            print(f"MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")

            return {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'predictions': predictions_all,
                'targets': targets_all,
                'main_effects': main_effects_all,
                'interactions': interactions_all,
                'contrib_main': contrib_main_all,
                'contrib_int_': contrib_int_all,
                'time_weight': time_weight_all
            }


    def plot_training_history(self, save_path=None, model_name='model'):
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
