import os
import sys
itemPth = os.path.dirname(os.path.dirname(__file__))
sys.path.append(itemPth)
file_name_without_ext = os.path.splitext(os.path.basename(__file__))[0]
import textwrap


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
plt.rcParams['axes.titlesize'] = 12              # 标题字体大小
plt.rcParams['axes.labelsize'] = 11              # 坐标轴标签字体大小
plt.rcParams['xtick.labelsize'] = 10             # x轴刻度字体大小
plt.rcParams['ytick.labelsize'] = 10             # y轴刻度字体大小
plt.rcParams['legend.fontsize'] = 10             # 图例字体大小



from utils import calculate_IoU_for_PSP, custom_serialize, evaluate_cbp # type: ignore


def sigmoid(x):
    return 1 / (1 + np.exp(-x))



def main():
    ground_truth = np.ones((30)) * 0.5
    model1_layer_opt = np.zeros((30))
    for i in range(20, 26):
        ground_truth[i] = 1
        model1_layer_opt[i] = (i-15) * 0.1
    model2_layer_opt = model1_layer_opt+3
    model3_layer_opt = model1_layer_opt*4.5


    model1_response = sigmoid(model1_layer_opt)
    model2_response = sigmoid(model2_layer_opt)  - 0.12
    model3_response = sigmoid(model3_layer_opt)

    fig, axes = plt.subplots(3, 2, figsize=(6, 6))
    colors = ['#5BB318', 'm', '#2E5EAA',] #['blue', 'c', 'orange']
    markers = ['o', '*', '^']
    

    ## prepare data
    data = {f'model{i}': {'threshold_list': [],
                          'precision_list': [],
                          'recall_list': []} \
                          for i in range(1, 4)}
    for model_idx in range(1, 4):
        model_response = eval(f'model{model_idx}_response')
        thresholds = [float(res) for res in model_response]
        model_name = f'model{model_idx}'
        thresholds.append(min(max(model_response)-0.001, 1))
        thresholds.append(min(max(model_response)+0.001, 1))
        thresholds.append(max(min(model_response)-0.001, 0.5))
        thresholds.append(max(min(model_response)+0.001, 0.5))
        thresholds.append(0.5)
        thresholds.append(1)
        thresholds = list(set(thresholds))
        
        for threshold in sorted(thresholds):
            eva = evaluate_cbp(model_response, ground_truth==1, threshold)
            data[model_name]['threshold_list'].append(threshold)
            data[model_name]['precision_list'].append(eva['Precision'])
            data[model_name]['recall_list'].append(eva['Recall'])


    ## left plot
    # axes[0, 0]
    ax = axes[0, 0]
    for k, model_idx in enumerate(range(1, 4)):
        model_name = f'model{model_idx}'
        ax.plot(eval(f'model{model_idx}_response'), label= f'model{model_idx}',
                color=colors[k], marker=markers[k])
    ax.set_xlabel('Frame')
    ax.set_ylabel('Response')
    ax.set_title('Model Responses Over Time')
    ax.set_ylim((0.5, 1.05))

    # axes[1, 0]
    ax = axes[1, 0]
    for k, model_idx in enumerate(range(1, 4)):
        model_name = f'model{model_idx}'
        ax.plot(data[model_name]['recall_list'], 
                    data[model_name]['precision_list'], 
                    label=model_name, color=colors[k], marker=markers[k])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('P-R Curve')
    ax.set_ylim((0, 1.05))

        
    # axes[2, 0]
    ax = axes[2, 0]
    for k, model_idx in enumerate(range(1, 4)):
        model_name = f'model{model_idx}'
        ax.plot(data[model_name]['threshold_list'], 
                        data[model_name]['precision_list'], 
                        label=model_name, color=colors[k], marker=markers[k],
                        markersize=6,
                        )
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Precision')
    ax.set_title('Precision by Threshold')
    ax.set_ylim((0, 1.05))


    x_axis = np.ones((30))*0.5
    ## right plot
    # axes[0, 1]
    ax = axes[0, 1]
    ax.plot(ground_truth, label='GroundTruth', color='red', linestyle='--')
    ax.plot(model1_response, linewidth=2, color=colors[0], marker=markers[0])
    ax.fill_between(range(len(ground_truth)), ground_truth, x_axis,
                    where=(ground_truth > x_axis), 
                    color='red', alpha=0.1, interpolate=True)
    ax.fill_between(range(len(ground_truth)), model1_response, x_axis,
                    where=(model1_response > x_axis), 
                    hatch='xxx', alpha=0.1, interpolate=True)
    ax.set_xlabel('Frame')
    ax.set_ylabel('Response')
    ax.set_title('Model 1 Separability')
    ax.set_ylim((0.5, 1.05))
    Iou = calculate_IoU_for_PSP(model1_response, groundtruth=ground_truth)
    ax.text(0.5, 0.9, f'SE = {Iou:.2f}', bbox=dict(boxstyle='round', edgecolor='none', facecolor=colors[0], alpha=0.1) )

    # axes[1, 1]
    ax = axes[1, 1]
    ax.plot(ground_truth, label='GroundTruth', color='red', linestyle='--')
    ax.plot(model2_response, linewidth=2, color=colors[1], marker=markers[1])
    union_set = np.maximum(model2_response, ground_truth)
    ax.fill_between(range(len(ground_truth)), union_set, x_axis,
                    color='red', alpha=0.1, interpolate=True)
    intersection_set = np.minimum(model2_response, ground_truth)
    ax.fill_between(range(len(ground_truth)), intersection_set, x_axis,
                    where=(model1_response < ground_truth), 
                    hatch='xxx', alpha=0.1, interpolate=True)
    ax.set_xlabel('Frame')
    ax.set_ylabel('Response')
    ax.set_title('Model 2 Separability')
    ax.set_ylim((0.5, 1.05))
    Iou = calculate_IoU_for_PSP(model2_response, groundtruth=ground_truth)
    ax.text(0.5, 0.9, f'SE = {Iou:.2f}', bbox=dict(boxstyle='round', edgecolor='none', facecolor=colors[1], alpha=0.1) )

    # axes[2, 1]
    ax = axes[2, 1]
    ax.plot(ground_truth, color='red', linestyle='--')
    ax.plot(model3_response, linewidth=2, color=colors[2], marker=markers[2], )
    ax.fill_between(range(len(ground_truth)), ground_truth, x_axis,
                    where=(ground_truth > x_axis), 
                    color='red', alpha=0.1, interpolate=True)
    ax.fill_between(range(len(ground_truth)), model3_response, x_axis,
                    where=(model1_response > x_axis), 
                    hatch='xxx', alpha=0.1, interpolate=True)
    ax.set_xlabel('Frame')
    ax.set_ylabel('Response')
    ax.set_title('Model 3 Separability')
    ax.set_ylim((0.5, 1.05))
    Iou = calculate_IoU_for_PSP(model3_response, groundtruth=ground_truth)
    ax.text(0.5, 0.9, f'SE = {Iou:.2f}', bbox=dict(boxstyle='round', edgecolor='none', facecolor=colors[2], alpha=0.1) )


    # 创建统一的图例句柄
    legend_elements = [
        Line2D([0], [0], color=colors[0], marker=markers[0], linestyle='-', label='Model 1'),
        Patch(facecolor='none', alpha=0.3, hatch='xxx', edgecolor='black', label='Intersection Part'),
        Line2D([0], [0], color=colors[1], marker=markers[1], linestyle='-', label='Model 2'),
        Patch(facecolor='red', alpha=0.3, label='Union Part'),
        Line2D([0], [0], color=colors[2], marker=markers[2], linestyle='-', label='Model 3'),
        Line2D([0], [0], color='red', linestyle='--', label='Ideal Model (Ground Truth)'),
    ]

    # 添加统一图例
    fig.legend(handles=legend_elements, 
            loc='lower center', 
            bbox_to_anchor=(0.5, 0),
            ncol=3,
            fancybox=True, # 圆角边框
            framealpha=0.8)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)
    plt.savefig(os.path.join(os.path.dirname(__file__), f'{file_name_without_ext}.png'), 
                dpi=300)
    plt.show()
    ...




if __name__ == '__main__':
    main()