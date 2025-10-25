import os
import numpy as np
import torch
from lime.lime_tabular import LimeTabularExplainer
from torch.autograd import grad
import pandas as pd
from itertools import combinations
import shap
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import warnings
warnings.filterwarnings("ignore")

# -------------------- 工具函数 --------------------
def save_array(array, path):
    np.save(path, array)
    print(f"[INFO] Saved to {path}, shape={array.shape}")


def get_topq_indices(attributions, q):
    num_features = len(attributions)
    k = max(1, int(np.ceil(num_features * q)))
    return np.argsort(attributions)[-k:][::-1]


# -------------------- Sufficiency & Comprehensiveness --------------------
def compute_sufficiency_comprehensiveness_sample(model, x_sample, attributions, q_values):

    model.eval()
    x_tensor = torch.tensor(x_sample[None, ...], dtype=torch.float32).to(device)  # shape (1,T,N,F)

    with torch.no_grad():
        y_full = model(x_tensor).cpu().numpy()  # shape (1,H)

    var_importance = np.abs(attributions).mean(axis=(0, 1))  # (N,)

    suff_list, comp_list = [], []
    for q in q_values:
        k = max(1, int(len(var_importance) * q))
        top_indices = np.argsort(var_importance)[::-1][:k]

        x_suff = np.zeros_like(x_sample)
        x_suff[:, top_indices, :] = x_sample[:, top_indices, :]
        x_suff_tensor = torch.tensor(x_suff[None, ...], dtype=torch.float32).to(device)
        with torch.no_grad():
            y_suff = model(x_suff_tensor).cpu().numpy()

        x_comp = x_sample.copy()
        x_comp[:, top_indices, :] = 0
        x_comp_tensor = torch.tensor(x_comp[None, ...], dtype=torch.float32).to(device)
        with torch.no_grad():
            y_comp = model(x_comp_tensor).cpu().numpy()

        suff_list.append(np.mean(y_full - y_suff))
        comp_list.append(np.mean(y_full - y_comp))

    return suff_list, comp_list


def compute_suff_comp_dataset(model, X_test, attributions_all, q_values=[0.01, 0.05, 0.1, 0.2, 0.5]):

    B = X_test.shape[0]
    suff_matrix = np.zeros((B, len(q_values)))
    comp_matrix = np.zeros((B, len(q_values)))

    for i in range(B):
        x_sample = X_test[i]
        attr_sample = attributions_all[i]
        suff_list, comp_list = compute_sufficiency_comprehensiveness_sample(model, x_sample, attr_sample, q_values)
        suff_matrix[i] = suff_list
        comp_matrix[i] = comp_list

    suff_avg = np.mean(suff_matrix, axis=1).mean()
    comp_avg = np.mean(comp_matrix, axis=1).mean()

    return suff_avg, comp_avg, suff_matrix, comp_matrix


# -------------------- IG计算 --------------------
def compute_ig_per_horizon(model, X_test, h_index, steps=50, baseline=None):
    model.eval()
    B, T_in, N, F = X_test.shape
    ig_vals = []

    for i in range(B):
        x = torch.tensor(X_test[i:i + 1], dtype=torch.float32, requires_grad=True).to(device)

        if baseline is None:
            baseline_x = torch.zeros_like(x)
        else:
            baseline_x = torch.tensor(baseline[i:i + 1], dtype=torch.float32).to(device)

        scaled_inputs = [
            baseline_x + (float(k) / steps) * (x - baseline_x)
            for k in range(1, steps + 1)
        ]

        total_grad = torch.zeros_like(x)
        for scaled_x in scaled_inputs:
            scaled_x.requires_grad_(True)
            y = model(scaled_x)
            g = torch.autograd.grad(y[0, h_index], scaled_x, retain_graph=True)[0]
            total_grad += g

        avg_grad = total_grad / steps
        ig = (x - baseline_x) * avg_grad
        ig = ig.detach().cpu().numpy().sum(-1)[0]  # sum over F
        ig_vals.append(ig)

    return np.stack(ig_vals, axis=0)  # (B, T_in, N)


def compute_integrated_gradients_multihorizon(model, X_test, H, steps=50, baseline=None):
    ig_all = []
    for h in range(H):
        print(f"计算步长 {h + 1}/{H} 的 IG 值")
        ig_h = compute_ig_per_horizon(model, X_test, h, steps, baseline)
        ig_all.append(ig_h)
    return np.stack(ig_all, axis=1)  # (B, H, T_in, N)


#------------------------LIme
def compute_lime_per_horizon(model, X_test, h_index, nsamples=50):

    B, T_in, N, F = X_test.shape
    X_flat = X_test.reshape(B, -1)

    def f_single_output(x_numpy):
        x_torch = torch.tensor(x_numpy, dtype=torch.float32).reshape(-1, T_in, N, F).to(device)
        with torch.no_grad():
            y_pred = model(x_torch)[:, h_index]  # 取指定 horizon
        return y_pred.cpu().numpy()

    explainer = LimeTabularExplainer(X_flat, mode="regression")

    lime_vals_h = np.zeros((B, T_in, N))
    for i in range(B):
        exp = explainer.explain_instance(
            X_flat[i],
            f_single_output,
            num_features=T_in * N
        )

        for feat_idx, weight in exp.as_map()[0]:
            t = feat_idx // N
            n = feat_idx % N
            lime_vals_h[i, t, n] = weight

    return lime_vals_h

def compute_lime_multihorizon(model, X_test, H, nsamples=50):

    B, T_in, N, F = X_test.shape
    lime_all = []
    for h in range(H):
        print(f"计算步长 {h + 1}/{H} 的 LIME 值")
        lime_h = compute_lime_per_horizon(model, X_test, h, nsamples)
        lime_all.append(lime_h)
    lime_vals = np.stack(lime_all, axis=1)
    return lime_vals

def compute_shap_per_horizon(model, X_test, h_index, nsamples=50, background_size=100):

    B, T_in, N, F = X_test.shape
    X_flat = X_test.reshape(B, -1)  # (B, T_in*N*F)

    def f_single_output(x_numpy):
        x_torch = torch.tensor(x_numpy, dtype=torch.float32).reshape(-1, T_in, N, F).to(device)
        with torch.no_grad():
            y_pred = model(x_torch)[:, h_index]  # 取指定 horizon
        return y_pred.cpu().numpy()

    background = X_flat[np.random.choice(B, size=min(background_size, B), replace=False)]

    explainer = shap.KernelExplainer(f_single_output, background)

    shap_vals_h = np.zeros((B, T_in, N))

    for i in range(B):
        shap_values = explainer.shap_values(X_flat[i], nsamples=nsamples)
        shap_values = np.array(shap_values).reshape(T_in, N, F).mean(axis=-1)  # 如果 F>1，取平均
        shap_vals_h[i] = shap_values

    return shap_vals_h
def compute_shap_multihorizon(model, X_test, H, bg_size=50):
    shap_results = []
    for h in range(H):
        print(f"计算步长 {h + 1}/{H} 的SHAP值")
        shap_h = compute_shap_per_horizon(model, X_test, h, bg_size)
        shap_results.append(shap_h)
    shap_vals= np.stack(shap_results, axis=1)
    # 堆叠为 (B, H, T_in, N, F)
    return shap_vals

# -------------------- 主流程 --------------------
if __name__ == "__main__":
    folderpath = r"D:\interaction\experiments_results\electricity\exp_3"

    # 加载数据和模型
    X_test = np.load(os.path.join(folderpath, "electricity_test_X.npy"))
    Y_test = np.load(os.path.join(folderpath, "electricity_test_Y.npy"))

    model = torch.load(os.path.join(folderpath, "Transformer_model.pt"), map_location=device)
    model = model.to(device)

    model.eval()
    B, T_in, N, F = X_test.shape
    H = Y_test.shape[1]

    # -------------------- IG --------------------
    ig_vals = compute_integrated_gradients_multihorizon(model, X_test, H)
    save_array(ig_vals, os.path.join(folderpath, "ig_attributions.npy"))
    q_values = [0.01, 0.05, 0.1, 0.2, 0.5]
    suff_avg, comp_avg, _, _ = compute_suff_comp_dataset(model, X_test, ig_vals, q_values)
    print(f"ig -> 平均Sufficiency: {suff_avg:.4f}, 平均Comprehensiveness: {comp_avg:.4f}")


    # -------------------- Lime 计算与保存 --------------------
    lime_vals = compute_lime_multihorizon(model, X_test, H)
    save_array(lime_vals, os.path.join(folderpath, "lime_attributions.npy"))
    q_values = [0.01, 0.05, 0.1, 0.2, 0.5]
    suff_avg, comp_avg, _, _ = compute_suff_comp_dataset(model, X_test, lime_vals, q_values)

    print(f"LIMe -> 平均Sufficiency: {suff_avg:.4f}, 平均Comprehensiveness: {comp_avg:.4f}")

    gt_main = np.load(os.path.join(folderpath, "contrib_main.npy"))
    #-------------------- shap计算与保存 - -------------------

    shap_vals = compute_shap_multihorizon(model, X_test, H)
    save_array(shap_vals, os.path.join(folderpath, "shap_attributions.npy"))
    # ig_vals=np.load(os.path.join(folderpath, "ig_attributions.npy"))
    q_values = [0.01, 0.05, 0.1, 0.2, 0.5]
    suff_avg, comp_avg, _, _ = compute_suff_comp_dataset(model, X_test, shap_vals, q_values)
    print(f"shap-> 平均Sufficiency: {suff_avg:.4f}, 平均Comprehensiveness: {comp_avg:.4f}")

    # -------------------- 示例: Sufficiency & Comprehensiveness --------------------
    q_values = [0.01, 0.05, 0.1, 0.2, 0.5]
    suff_avg, comp_avg, _, _ = compute_suff_comp_dataset(model, X_test, gt_main, q_values)

    print(f"Dynamic -> 平均Sufficiency: {suff_avg:.4f}, 平均Comprehensiveness: {comp_avg:.4f}")

