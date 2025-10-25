import os
import numpy as np
import pandas as pd
# true interpretation results
import numpy as np
from scipy.stats import pearsonr, spearmanr
from scipy.stats import kendalltau


def evaluate_importance_alignment(contrib_main, contrib_int,
                                  gt_main, gt_int,
                                  topk=5, aggregate="mean"):
    """
    对比模型解释与 ground-truth 解释的一致性

    参数:
      contrib_main: (B,H,T,N) 模型主效应贡献
      contrib_int:  (B,H,T,P) 模型交互效应贡献
      gt_main:      (B,H,T,N) 真值主效应贡献
      gt_int:       (B,H,T,P) 真值交互贡献
      topk:         评价 Top-K 变量恢复率
      aggregate:    "mean" 或 "sum"，对 (B,H,T) 聚合的方式

    返回:
      results: dict {
        'main_corr': 主效应相关性,
        'int_corr':  交互效应相关性,
        'main_topk_acc': Top-K 主效应恢复率,
        'int_topk_acc':  Top-K 交互恢复率
      }
    """

    # --- Step1: 聚合 (忽略 batch/horizon/time) ---
    if aggregate == "mean":
        imp_main = np.abs(contrib_main).mean(axis=(0, 1, 2))  # (N,)
        imp_int = np.abs(contrib_int).mean(axis=(0, 1, 2))  # (P,)
        gt_imp_main = np.abs(gt_main).mean(axis=(0, 1, 2))
        gt_imp_int = np.abs(gt_int).mean(axis=(0, 1, 2))
    elif aggregate == "sum":
        imp_main = np.abs(contrib_main).sum(axis=(0, 1, 2))
        imp_int = np.abs(contrib_int).sum(axis=(0, 1, 2))
        gt_imp_main = np.abs(gt_main).sum(axis=(0, 1, 2))
        gt_imp_int = np.abs(gt_int).sum(axis=(0, 1, 2))
    else:
        raise ValueError("aggregate must be 'mean' or 'sum'")

    # --- Step2: 相关性 (Spearman/Pearson) ---

    main_corr = np.corrcoef(imp_main, gt_imp_main)[0, 1]
    int_corr = np.corrcoef(imp_int, gt_imp_int)[0, 1]

    # --- Step3: Top-K 恢复率 ---
    topk_main_idx = np.argsort(-imp_main)[:topk]
    topk_main_gt = np.argsort(-gt_imp_main)[:topk]
    main_topk_acc = len(set(topk_main_idx) & set(topk_main_gt)) / topk

    topk_int_idx = np.argsort(-imp_int)[:topk]
    topk_int_gt = np.argsort(-gt_imp_int)[:topk]
    int_topk_acc = len(set(topk_int_idx) & set(topk_int_gt)) / topk

    return {
        "main_corr": main_corr,
        "int_corr": int_corr,
        "main_topk_acc": main_topk_acc,
        "int_topk_acc": int_topk_acc,
    }

folderpath= "D:\interaction\experiments_results\synthetic\exp_0"
contrib_main_path =os.path.join(folderpath,"contrib_main.npy")  #
contrib_inter_path=os.path.join(folderpath,"contrib_int.npy")
contrib_main=np.load(contrib_main_path)
contrib_inter=np.load(contrib_inter_path)

main_mag = np.abs(contrib_main).sum()   # overall magnitude
int_mag  = np.abs(contrib_inter).sum()
print("main_mag:", main_mag, "int_mag:", int_mag, "int_ratio:", int_mag/(main_mag+int_mag))
gt_main_path =os.path.join(folderpath,"test_gt_main.npy")
gt_inter_path=os.path.join(folderpath,"test_gt_int.npy")
gt_main=np.load(gt_main_path)
gt_int=np.load(gt_inter_path)
def upper_pair_idx(N):
    idx = np.triu_indices(N, k=1)
    pairs = list(zip(idx[0], idx[1]))
    return pairs  # p -> (i,j)

N = contrib_main.shape[-1]

pair_list = [(i, j) for i in range(N) for j in range(i+1, N)]

print(f"gt_main 均值: {gt_main.mean():.4f}, 最大值: {gt_main.max():.4f}")
print(f"gt_int 均值: {gt_int.mean():.4f}, 最大值: {gt_int.max():.4f}")
print(f"com_main 均值: {contrib_main.mean():.4f}, 最大值: {contrib_main.max():.4f}")
print(f"con_int 均值: {contrib_inter.mean():.4f}, 最大值: {contrib_inter.max():.4f}")
results = evaluate_importance_alignment(contrib_main, contrib_inter, gt_main, gt_int, topk=5)
print(results)

shap_vals = np.load(os.path.join(folderpath, "shap_attributions.npy"))
lime_vals = np.load(os.path.join(folderpath, "lime_attributions.npy"))
ig_vals = np.load(os.path.join(folderpath, "ig_attributions.npy"))

def evaluate_importance_alignment(contrib_main,
                                  gt_main,
                                  topk=5, aggregate="mean"):



    # --- Step1: 聚合 (忽略 batch/horizon/time) ---
    if aggregate == "mean":
        imp_main = np.abs(contrib_main).mean(axis=(0, 1, 2))  # (N,)
        gt_imp_main = np.abs(gt_main).mean(axis=(0, 1, 2))
    elif aggregate == "sum":
        imp_main = np.abs(contrib_main).sum(axis=(0, 1, 2))
        gt_imp_main = np.abs(gt_main).sum(axis=(0, 1, 2))
    else:
        raise ValueError("aggregate must be 'mean' or 'sum'")

    # --- Step2: 相关性 (Spearman/Pearson) ---

    main_corr = np.corrcoef(imp_main, gt_imp_main)[0, 1]

    # --- Step3: Top-K 恢复率 ---
    topk_main_idx = np.argsort(-imp_main)[:topk]
    topk_main_gt = np.argsort(-gt_imp_main)[:topk]
    main_topk_acc = len(set(topk_main_idx) & set(topk_main_gt)) / topk


    return {
        "main_corr": main_corr,

        "main_topk_acc": main_topk_acc,
    }

results = evaluate_importance_alignment(ig_vals,  gt_main, topk=5)
print("ig",results)

results = evaluate_importance_alignment(shap_vals,  gt_main, topk=5)
print("shap",results)
results = evaluate_importance_alignment(lime_vals, gt_main, topk=5)
print("lime",results)

