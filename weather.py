import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import TensorDataset, DataLoader
import config
import os
import argparse

# 导入自定义模块
#from model import DynamicInteractionModel,LSTM, GRU,Transformer
from finalmodel import DynamicInteractionModelCoupled,LSTM, GRU,Transformer
from baselinemodel import Informer, Autoformer

from model_training import elec_training

parser = argparse.ArgumentParser(description="Electricity consumption prediction")
parser.add_argument('--seed', type=int, default=555, help='The random seed')
parser.add_argument('--num_layers', type=int, default=2, help='The layers of transformer')
parser.add_argument('--nhead', type=int, default=4, help='The heads of transformer')
parser.add_argument('--r_factor', type=int, default=16, help='project dimension for interaction')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
parser.add_argument('--use_bilinear', type=bool, default=True, help='dropout rate')
parser.add_argument('--prediction_horizon', type=int, default=24, help='prediction length')
parser.add_argument('--batch_size', type=int, default=256, help='The batch size when training NN')
parser.add_argument('--lr', type=float, default=1e-3, help='The learning rate when training NN')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay when training NN')
parser.add_argument('--feature_dimensionality', type=int, default=21, help='the feature dimensionality')
parser.add_argument('--input_dim', type=int, default=1, help='The dimension of feature')
parser.add_argument('-hidden_dim', type=int, default=32, help='The hidden size for embedding')
parser.add_argument('--dataset', type=str, default='weather')
parser.add_argument('--save_dirs', type=str, default='experiments_results', help='The dirs for saving results')
parser.add_argument('--num_exp', type=int, default=5, help='The number of experiment')
parser.add_argument('--log', type=bool, default=True, help='Whether log the information of training process')
parser.add_argument('--save_models', type=bool, default=False, help='Whether save the training models')
parser.add_argument('-stage_a_epochs', type=int, default=100, help='The epochs of stage A')
parser.add_argument('-stage_b_epochs', type=int, default=100, help='The epochs of stage b')
parser.add_argument('-lambda_sparsity', type=float, default=1e-3, help='The lambda_sparsity')
parser.add_argument('-lambda_smooth', type=float, default=1e-3, help='The lambda_smooth')
parser.add_argument('--T', type=int, default=144, help='past length')

args = parser.parse_args()

#数据加载和划分
class ElectricityDataset(Dataset):
    def __init__(self, data, target_index=1, L_in=48, H=24):
        self.data = data
        self.L_in = L_in
        self.H = H
        self.N = data.shape[1]
        self.target_index = target_index

    def __len__(self):
        return len(self.data) - self.L_in - self.H + 1

    def __getitem__(self, idx):
        x_win = self.data[idx : idx + self.L_in]   # (L_in, N)
        y_win = self.data[idx + self.L_in : idx + self.L_in + self.H, self.target_index]  # (H,)
        x_win = torch.tensor(x_win).unsqueeze(-1)  # (L_in, N, 1)
        y_win = torch.tensor(y_win).float()
        return x_win, y_win


def create_dataloaders(data, L_in=48, H=24, batch_size=64, train_ratio=0.7, val_ratio=0.15):
    total_len = len(data)
    n_train = int(total_len * train_ratio)
    n_val = int(total_len * val_ratio)
    n_test = total_len - n_train - n_val

    # 为了保证输入窗口完整，验证和测试集需要从前L_in开始
    train_data = data[:n_train]
    val_data = data[n_train - L_in: n_train + n_val]
    test_data = data[n_train + n_val - L_in:]

    #在训练集上scaler
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_data)
    val_scaled = scaler.transform(val_data)
    test_scaled = scaler.transform(test_data)

    train_set = ElectricityDataset(train_scaled, L_in=L_in, H=H)
    val_set = ElectricityDataset(val_scaled, L_in=L_in, H=H)
    test_set = ElectricityDataset(test_scaled, L_in=L_in, H=H)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    X_list, Y_list = [], []
    for x, y in test_loader:
        X_list.append(x.numpy())  # (B, L_in, N, F)
        Y_list.append(y.numpy())  # (B, H)
    X_test = np.concatenate(X_list, axis=0)
    Y_test = np.concatenate(Y_list, axis=0)
    np.save("weather_test_X.npy", X_test)
    np.save("weather_test_Y.npy", Y_test)
    print(f"Saved test set: X {X_test.shape}, Y {Y_test.shape}")

    return train_loader, val_loader, test_loader,scaler

#主程序
if __name__ == "__main__":
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    folderpath= "D:\interaction\DATA\weather"
    filepath = "weather.csv"  #
    file=os.path.join(folderpath, filepath)
    df=pd.read_csv(file,encoding='gbk')
    L_in, H = 144, 24#用48h预测24h

    features= df.drop(columns=["date"]).values.astype(np.float32)

    train_loader, val_loader, test_loader,scaler = create_dataloaders(features, L_in=L_in, H=H)

    #开始训练
    model_list = ['Dynamic', 'LSTM', 'GRU', 'Transformer','Informer','Autoformer']
    #model_list = [ 'LSTM', 'GRU', 'Transformer']
    #model_list=['Autoformer','Informer']
    #model_list=['Dynamic']


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
            elif model_name=='Informer':
                model = Informer(config.config(model_name, args)).to(device)
            elif model_name=='Autoformer':
                model = Autoformer(config.config(model_name, args)).to(device)

            else:
                ModuleNotFoundError(f'Module {model_name} not found')

            print(f'Training model {model_name}')
            print(f'Num of trainable params:', sum(p.numel() for p in model.parameters() if p.requires_grad))
            elec_training(model, model_name, train_loader, val_loader, test_loader,scaler, args, device, exp_id)






















