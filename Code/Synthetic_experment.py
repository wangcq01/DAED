import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import TensorDataset, DataLoader
import config
import os
import argparse
from finalmodel import DynamicInteractionModelCoupled,LSTM, GRU,Transformer
from baselinemodel import  Autoformer
from model_training import elec_training

parser = argparse.ArgumentParser(description="synthetic prediction")
parser.add_argument('--seed', type=int, default=555, help='The random seed')
parser.add_argument('--num_layers', type=int, default=2, help='The layers of transformer')
parser.add_argument('--nhead', type=int, default=4, help='The heads of transformer')
parser.add_argument('--r_factor', type=int, default=16, help='project dimension for interaction')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
parser.add_argument('--use_bilinear', type=bool, default=True, help='dropout rate')
parser.add_argument('--prediction_horizon', type=int, default=12, help='prediction length')
parser.add_argument('--T', type=int, default=24, help='past length')

parser.add_argument('--batch_size', type=int, default=256, help='The batch size when training NN')
parser.add_argument('--lr', type=float, default=1e-3, help='The learning rate when training NN')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay when training NN')
parser.add_argument('--feature_dimensionality', type=int, default=10, help='the feature dimensionality')
parser.add_argument('--input_dim', type=int, default=1, help='The dimension of feature')
parser.add_argument('-hidden_dim', type=int, default=16, help='The hidden size for embedding')
parser.add_argument('--dataset', type=str, default='synthetic')
parser.add_argument('--save_dirs', type=str, default='experiments_results', help='The dirs for saving results')
parser.add_argument('--num_exp', type=int, default=5, help='The number of experiment')
parser.add_argument('--log', type=bool, default=True, help='Whether log the information of training process')
parser.add_argument('--save_models', type=bool, default=False, help='Whether save the training models')
parser.add_argument('-stage_a_epochs', type=int, default=200, help='The epochs of stage A')
parser.add_argument('-stage_b_epochs', type=int, default=200, help='The epochs of stage b')
parser.add_argument('-lambda_sparsity', type=float, default=1e-3, help='The lambda_sparsity')
parser.add_argument('-lambda_smooth', type=float, default=1e-3, help='The lambda_smooth')

args = parser.parse_args()


class SynthDataset(Dataset):
    """
    输入：X (B, T_in, N, F)，Y (B, H)
    """
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def create_dataloaders_synth(X, Y,gt_main, gt_int, L_in=24, H=12, batch_size=64, train_ratio=0.7, val_ratio=0.15):
    """
    输入是 synth_dataset 生成的 X, Y
    X: (B, L_in, N, F)
    Y: (B, H)
    """

    total_len = len(X)
    n_train = int(total_len * train_ratio)
    n_val = int(total_len * val_ratio)
    n_test = total_len - n_train - n_val

    # --- 划分数据 ---
    train_X, train_Y = X[:n_train], Y[:n_train]
    val_X, val_Y     = X[n_train:n_train+n_val], Y[n_train:n_train+n_val]
    test_X, test_Y   = X[n_train+n_val:], Y[n_train+n_val:]
    test_gt_main = gt_main[n_train + n_val:]
    test_gt_int = gt_int[n_train + n_val:]
    np.save("test_gt_main.npy", test_gt_main)
    np.save("test_gt_int.npy", test_gt_int)



    # --- 标准化 (只在 train 上 fit，逐样本 flatten 后缩放再 reshape) ---
    scaler = StandardScaler()
    B, T, N, F = train_X.shape
    train_X_reshaped = train_X.reshape(B, -1)
    scaler.fit(train_X_reshaped)

    def scale_split(X_split):
        B, T, N, F = X_split.shape
        X_scaled = scaler.transform(X_split.reshape(B, -1)).reshape(B, T, N, F)
        return X_scaled

    train_X = scale_split(train_X)
    val_X   = scale_split(val_X)
    test_X  = scale_split(test_X)

    # --- 封装 Dataset & DataLoader ---
    train_set = SynthDataset(train_X, train_Y)
    val_set   = SynthDataset(val_X, val_Y)
    test_set  = SynthDataset(test_X, test_Y)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    X_list, Y_list = [], []
    for x, y in test_loader:
        X_list.append(x.numpy())  # (B, L_in, N, F)
        Y_list.append(y.numpy())  # (B, H)
    X_test = np.concatenate(X_list, axis=0)
    Y_test = np.concatenate(Y_list, axis=0)
    np.save("synthetic_test_X.npy", X_test)
    np.save("synthetic_test_Y.npy", Y_test)
    print(f"Saved test set: X {X_test.shape}, Y {Y_test.shape}")

    return train_loader, val_loader, test_loader, scaler

#主程序
if __name__ == "__main__":
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    folderpath= "D:\interaction\DATA\synthetic"

    X=np.load(os.path.join(folderpath, 'synth_X.npy'))
    Y=np.load(os.path.join(folderpath, 'synth_Y.npy'))
    gt_main_path = os.path.join(folderpath, "synth_gt_main.npy")
    gt_inter_path = os.path.join(folderpath, "synth_gt_int.npy")
    gt_main = np.load(gt_main_path)
    gt_int = np.load(gt_inter_path)
    #missing_counts = df.isnull().sum()
    #print("各列缺失值数量:\n", missing_counts[missing_counts > 0])
    L_in, H = 48, 24 #用48h预测24h
    # 2) 创建 dataloader
    train_loader, val_loader, test_loader, scaler = create_dataloaders_synth(
        X, Y,gt_main,gt_int, L_in=24, H=12, batch_size=args.batch_size, train_ratio=0.7, val_ratio=0.15
    )

    #features= df.drop(columns=["No",'year','month','day','hour']).values.astype(np.float32)

    #train_loader, val_loader, test_loader,scaler = create_dataloaders(features, L_in=L_in, H=H)

    #开始训练
    model_list = ['Dynamic', 'LSTM', 'GRU', 'Transformer','Autoformer']

    for exp_id in range(args.num_exp):
        for model_name in model_list:
            if model_name == 'Dynamic':
                model = DynamicInteractionModelCoupled(config.config(model_name, args)).to(device)

            elif model_name == 'LSTM':
                model =LSTM(config.config(model_name, args)).to(device)

            elif model_name == 'GRU':
                model = GRU(config.config(model_name, args)).to(device)

            elif model_name == 'Transformer':
                model = Transformer(config.config(model_name, args)).to(device)
            elif model_name=='Autoformer':
                model = Autoformer(config.config(model_name, args)).to(device)

            else:
                ModuleNotFoundError(f'Module {model_name} not found')

            print(f'Training model {model_name}')
            print(f'Num of trainable params:', sum(p.numel() for p in model.parameters() if p.requires_grad))
            elec_training(model, model_name, train_loader, val_loader, test_loader,scaler, args, device, exp_id)






















