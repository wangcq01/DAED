import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

folderpath = r"D:\interaction\oneainal_experiments_results\electricity\exp_4"
gt_main = np.load(os.path.join(folderpath, "contrib_main.npy"))  # (B,T,H,N)
gt_inter = np.load(os.path.join(folderpath, "contrib_int.npy"))  # (B,T,H,P)
B, T, H, N = gt_main.shape

# --- 简化变量名称 ---
full_names = ["Homestead_maxtempC","Homestead_mintempC","Homestead_DewPointC",
              "Homestead_FeelsLikeC","Homestead_HeatIndexC","Homestead_WindChillC",
              "Homestead_WindGustKmph","Homestead_cloudcover","Homestead_humidity",
              "Homestead_precipMM","Homestead_pressure","Homestead_tempC",
              "Homestead_visibility","Homestead_winddirDegree","Homestead_windspeedKmph",
              "Consumption"]

var_names = [name.replace("Homestead_", "") for name in full_names]

# --- Step0: 生成上三角交互对索引 ---
def upper_pair_idx(N):
    idx = np.triu_indices(N, k=1)
    return list(zip(idx[0], idx[1]))

pair_list = upper_pair_idx(N)
P = len(pair_list)

# --- Step1: 平均主效应重要性 ---
main_mean = np.abs(gt_main).mean(axis=(0,1,2))  # (N,)
top_main_idx = np.argsort(-main_mean)[:8]  # 前8个变量

# --- Step2: 可视化单变量贡献 (Top-8) ---
plt.figure(figsize=(10,5))
sns.barplot(x=[var_names[i] for i in top_main_idx], y=main_mean[top_main_idx], palette="Blues_r")
plt.title("Top-8 Main Effect Importance (Electricity)")
plt.ylabel("Mean |Contribution|")
plt.xlabel("Variables")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("main_effect_barplot_top8.pdf")
plt.close()

# --- Step3: 平均交互强度矩阵 (热力图) ---
inter_mean = np.abs(gt_inter).mean(axis=(0,1,2))  # (P,)
inter_matrix = np.zeros((N,N))
for idx, (i,j) in enumerate(pair_list):
    inter_matrix[i,j] = inter_mean[idx]
    inter_matrix[j,i] = inter_mean[idx]  # 对称矩阵

plt.figure(figsize=(10,8))
sns.heatmap(inter_matrix, annot=True, fmt=".2f", cmap="Purples",
            xticklabels=var_names, yticklabels=var_names)
plt.title("Average Pairwise Interaction Strength")
plt.xlabel("Variables")
plt.ylabel("Variables")
plt.tight_layout()
plt.savefig("interaction_heatmap.pdf")
plt.close()

# --- Step4: 交互随时间演化 (Top-5最重要交互对) ---
# 按交互强度选择前5个
top_inter_idx = np.argsort(-inter_mean)[:5]

time_series = np.abs(gt_inter)[:,:, :, top_inter_idx].mean(axis=(0,2))  # (T, 5)

plt.figure(figsize=(12,6))
for k, idx in enumerate(top_inter_idx):
    i,j = pair_list[idx]
    plt.plot(range(T), time_series[:,k], label=f"{var_names[i]}-{var_names[j]}")
plt.xlabel("Time")
plt.ylabel("Mean |Interaction Contribution|")
plt.title("Top-5 Interaction Strength Evolution over Time")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
plt.tight_layout()
plt.savefig("interaction_time_series_top5.pdf")
plt.close()

print("Visualization done: main_effect_barplot_top8.pdf, interaction_heatmap.pdf, interaction_time_series_top5.pdf")
