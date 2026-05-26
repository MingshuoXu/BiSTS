import os
import sys
itemPth = os.path.dirname(os.path.dirname(__file__))
sys.path.append(itemPth)
file_name_without_ext = os.path.splitext(os.path.basename(__file__))[0]
import textwrap
import random   

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib as mpl
# 全局设置marker大小
mpl.rcParams['lines.markersize'] = 3  # 设置默认marker大小
# 全局设置图表边框
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False
# 全局设置字体
plt.rcParams['font.family'] = 'Times New Roman'  # 设置字体家族
plt.rcParams['font.size'] = 12                   # 设置字体大小
plt.rcParams['axes.titlesize'] = 13              # 标题字体大小
plt.rcParams['axes.labelsize'] = 12              # 坐标轴标签字体大小
plt.rcParams['xtick.labelsize'] = 12             # x轴刻度字体大小
plt.rcParams['ytick.labelsize'] = 12             # y轴刻度字体大小
plt.rcParams['legend.fontsize'] = 11             # 图例字体大小



from utils import calculate_IoU_for_PSP, calculate_ap, evaluate_cbp, calculate_auc_roc # type: ignore


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def main():
    ground_truth = np.ones((30)) * 0.5
    model1_layer_opt = np.zeros((30))
    for i in range(30):
        if 20 <= i <= 25:
            ground_truth[i] = 1
            model1_layer_opt[i] += i * 0.05
        else:
            model1_layer_opt[i] += random.uniform(-0.1, 0.1)
        
    model1_layer_opt += 0.4
    model2_layer_opt = model1_layer_opt*2.4
    model3_layer_opt = model1_layer_opt+0.5
    model4_layer_opt = model1_layer_opt/2.4
    model5_layer_opt = model1_layer_opt-0.5


    model1_response = sigmoid(model1_layer_opt)
    model2_response = sigmoid(model2_layer_opt)
    model3_response = sigmoid(model3_layer_opt)
    model4_response = sigmoid(model4_layer_opt)
    model5_response = sigmoid(model5_layer_opt)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3))
    colors = ['#5BB318', 'm', '#2E5EAA', '#FF7F0E', '#8C564B'] #['blue', 'c', 'orange', 'green', 'red']
    markers = ['o', '*', '^', 's', 'D']

    ax = axes[0]
    for k, model_idx in enumerate(range(1, 6)):
        model_name = f'model{model_idx}'
        if k > 0:
            ax.plot(eval(f'model{model_idx}_response'), label= f'model{model_idx}',
                    color=colors[k], linestyle='--',
                    )
        else:
            ax.plot(eval(f'model{model_idx}_response'), label= f'model{model_idx}',
                    color=colors[k], 
                    )
    ax.fill_between(range(20, 26), 1, 0, color='red', alpha=0.1)
    ax.set_xlabel('Frame')
    ax.set_ylabel('Response')
    ax.set_title('Model Responses')
    ax.set_ylim((0.5, 1.05))


    model_performance = {}
    ## prepare data
    data = {f'model{i}': {'threshold_list': [],
                          'precision_list': [],
                          'recall_list': [],
                          'tp list': [],
                          'fp list': [],
                          'fn list': [],
                          'tn list': [],
                          } \
                          for i in range(1, 6)}
    for model_idx in range(1, 6):
        model_response = eval(f'model{model_idx}_response')
        thresholds = [float(res) for res in model_response]
        model_name = f'model{model_idx}'
        thresholds = list(set(thresholds))
        for threshold in sorted(thresholds):
            eva = evaluate_cbp(model_response, ground_truth==1, threshold)
            data[model_name]['threshold_list'].append(threshold)
            data[model_name]['precision_list'].append(eva['Precision'])
            data[model_name]['recall_list'].append(eva['Recall'])
            data[model_name]['tp list'].append(eva['True Positives'])
            data[model_name]['fp list'].append(eva['False Positives'])
            data[model_name]['fn list'].append(eva['False Negatives'])
            data[model_name]['tn list'].append(eva['True Negatives'])
        AP = calculate_ap(model_response, ground_truth==1)
        IoU = calculate_IoU_for_PSP(model_response, ground_truth)
        AUC = calculate_auc_roc(data[model_name]['tp list'], 
                                data[model_name]['fp list'],
                                data[model_name]['fn list'],
                                data[model_name]['tn list'])
        model_performance[f'model{model_idx}'] = {
            'AP': AP,
            'IoU': IoU,
            'AUC': AUC,
        }

    
    # 
    ax = axes[1]
    for k, idx in enumerate(range(1, 6)):
        model_name = f'model{idx}'
        ax.plot(data[model_name]['threshold_list'], 
                        data[model_name]['precision_list'], 
                        label=model_name, color=colors[k], marker=markers[k],
                        )
        
        
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Precision')
    ax.set_title('Model Performance')
    ax.set_xlim((0.5, 1))
    ax.set_ylim((0, 1.05))


    ax_inset = fig.add_axes([0.86, 0.45, 0.1, 0.3])  # 位置和大小
    ax_inset.set_facecolor('lightblue')  # 设置背景颜色
    ax_inset.patch.set_alpha(0.3)  # 设置透明度
    for idx in range(1, 6):
        # ax_inset.scatter(model_performance[f'model{idx}']['IoU'], model_performance[f'model{idx}']['AP'], color=colors[idx-1], marker=markers[idx-1], s=20)
        ax_inset.scatter(model_performance[f'model{idx}']['AUC'], model_performance[f'model{idx}']['IoU'], color=colors[idx-1], marker=markers[idx-1], s=22)
    ax_inset.set_xlabel('AUC', fontsize=11)
    ax_inset.set_ylabel('SA', fontsize=11)
    # ax_inset.yaxis.set_label_coords(-0.1, 1.15)  # 调整坐标位置 (x, y)
    ax_inset.set_xlim((0.5, 0.6))
    ax_inset.set_ylim((0.2, 0.4))

    # 创建统一的图例句柄
    legend_elements = [
        Line2D([0], [0], color=colors[0], marker=markers[0], linestyle='-', label='Original'),
        Line2D([0], [0], color=colors[2], marker=markers[2], linestyle='-', label='Linear +'),
        Line2D([0], [0], color=colors[4], marker=markers[4], linestyle='-', label='Linear -'),
        Line2D([0], [0], color=colors[1], marker=markers[1], linestyle='-', label='Non-linear +'),
        Line2D([0], [0], color=colors[3], marker=markers[3], linestyle='-', label='Non-linear -'),
        Patch(facecolor='red', alpha=0.3, label='Ground truth'),
    ]

    # 添加统一图例
    fig.legend(handles=legend_elements, 
            loc='lower center', 
            bbox_to_anchor=(0.5, -0.01),
            ncol=6,
            handletextpad=0.5,
            columnspacing=1,
            fancybox=True, # 圆角边框
            framealpha=0.8,
            )


    plt.tight_layout()
    plt.subplots_adjust(bottom=0.28)
    plt.savefig(os.path.join(os.path.dirname(__file__), f'{file_name_without_ext}.png'), 
                dpi=300)
    plt.show()
    ...




if __name__ == '__main__':
    main()