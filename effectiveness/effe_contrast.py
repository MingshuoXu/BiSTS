import os
import sys
itemPth = os.path.dirname(os.path.dirname(__file__))
sys.path.append(itemPth)
file_name_without_ext = os.path.splitext(os.path.basename(__file__))[0]
import concurrent.futures
from prettytable import PrettyTable

import cv2
from tqdm import tqdm
import json
import numpy as np
import matplotlib.pyplot as plt
# 全局设置字体
plt.rcParams['font.family'] = 'Times New Roman'  # 设置字体家族
plt.rcParams['font.size'] = 12                   # 设置字体大小
plt.rcParams['axes.titlesize'] = 14              # 标题字体大小
plt.rcParams['axes.labelsize'] = 12              # 坐标轴标签字体大小
plt.rcParams['xtick.labelsize'] = 10             # x轴刻度字体大小
plt.rcParams['ytick.labelsize'] = 10             # y轴刻度字体大小
plt.rcParams['legend.fontsize'] = 10             # 图例字体大小
import seaborn as sns
import pandas as pd
import textwrap
import torch

from model.bists_lgmd import BiSTS_LGMD # type: ignore
from model.existing_lgmd import BasicLGMD, LGMD_P_ON_OFF # type: ignore
from utils import calculate_IoU_for_PSP, custom_serialize


BENCHMARK_FOLDER = os.path.join(os.path.dirname(itemPth), '7_Dataset', 'lgmd_benchmark')

RAW_DATA_FILE_NAME = os.path.join(os.path.dirname(__file__), f'{file_name_without_ext}.json')

MODEL_LIST = ['BasicLGMD', 'LGMD_P_ON_OFF', 'BiSTS_LGMD']
INPUT_DATASETS = {
    'contrast_looming': os.path.join(os.path.dirname(itemPth), '7_Dataset', 'BiSTS_extend', 'contrast looming'),
    'looming_square_against_nature_scenarios': os.path.join(BENCHMARK_FOLDER, 'looming square against nature scenarios'),
    'looming_ball_against_driving_sense': os.path.join(BENCHMARK_FOLDER, 'looming ball against driving sense'),
}
groundtruth_pth = os.path.join(itemPth, '..', '7_Dataset', 'lgmd_benchmark', 'annotations.json')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def inference_model(model_api, max_workers=10):
    results = {}
    for dataset_key, dataset_name in INPUT_DATASETS.items():
        results[dataset_key] = {}
        for name in os.listdir(dataset_name):
            if name.endswith('.mp4'):
                results[dataset_key][name] = None
    
    features = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        for dataset_key, dataset_name in INPUT_DATASETS.items():
            for name in results[dataset_key].keys():
                
                future = executor.submit(_inference, os.path.join(dataset_name, name), model_api)
                future.dataset_key = dataset_key  # Attach dataset name to the future
                future.name = name  # Attach video name to the future
                features.append(future)

        for future in tqdm(concurrent.futures.as_completed(features), total=len(features), desc=f"Effectiveness Contrast Inference - {model_api.__name__}"):
            res = future.result()
            results[future.dataset_key][future.name] = res

    return results


def _inference(vidPth, model_api=None):
    objVid = cv2.VideoCapture(vidPth)

    # get process function
    class_obj, process_fun_name = model_api()
    # process_func = class_obj.process
    process_func = getattr(class_obj, process_fun_name)

    result = []

    while objVid.isOpened():
        ret, colorImg = objVid.read()
        if not ret:
            break

        gray = cv2.cvtColor(colorImg, cv2.COLOR_BGR2GRAY)
        gray_tensor = torch.from_numpy(gray).float().unsqueeze(0).unsqueeze(0).to(DEVICE)  # Add batch and channel dimensions

        k = process_func(gray_tensor)
        if isinstance(k, torch.Tensor):
            k = k.item()
        result.append(k)

    objVid.release()
    return result

  
def visulize_var():
    with open(RAW_DATA_FILE_NAME, 'r') as f:
        results = json.load(f)

    fig, axs = plt.subplots(3, len(MODEL_LIST), figsize=(9, 5))
    for i, dataset_key in enumerate(INPUT_DATASETS.keys()):
        for j, model in enumerate(MODEL_LIST):
            ax = axs[i, j]
            responses= []
            for vid_name, res in results[model][dataset_key].items():
                if len(res) > 0 and max(res) != 0.5:
                    responses.append(res)
            if i == 0:
                ax.set_title(model, fontsize=12)

            responses = np.array(responses)

            # 计算均值和置信区间
            mean = np.mean(responses, axis=0)
            std = np.std(responses, axis=0)

            ax.plot(mean, label='Mean')
            ax.fill_between(range(len(mean)), np.maximum(mean - std, 0.5), np.minimum(mean + std, 1), color='b', alpha=0.2)

            ax.set_xlabel('Frame', fontsize=8)
            if j == 0:
                ax.set_ylabel('Response', fontsize=8)
            
            ax.grid()


    plt.tight_layout()
    plt.subplots_adjust(top = 0.9, right = 0.90)

    # 设置行标题
    for i, title in enumerate(INPUT_DATASETS):
        # 在第一列左侧添加行标题
        # 限制每行字符数并自动换行
        title = title.split('\\')[-1].replace('-', ' ').replace('_', ' ').title()  # 提取最后一部分作为标题，并格式化
        max_chars_per_line = 20
        wrapped = '\n'.join(textwrap.wrap(title, max_chars_per_line))
        # 在左侧添加竖直行标题（自动换行）
        fig.text(0.95, 0.8 - i*0.3, wrapped, ha='center', va='center', fontsize=10, rotation=270)
    # plt.savefig(json_name.replace('.json', '_var.png'), dpi=300)
    plt.show()


def visulize_violin_plot():
    with open(RAW_DATA_FILE_NAME, 'r') as f:
        results = json.load(f)

    
    # 准备数据用于绘图
    plot_data = []
    
    for dataset_key in INPUT_DATASETS.keys():
        for model in MODEL_LIST:
            responses= []
            for vid_name, res in results[model][dataset_key].items():
                yData = res
                if len(yData) > 2 and max(yData) > 0.5:
                    responses.append(yData)
                
            # 计算均值和置信区间
            responses = np.array(responses)
            meanVal = np.mean(responses, axis=0)
            stdVal = np.std(responses, axis=0)
            CoVs = stdVal / meanVal
            CoVs = CoVs.tolist()
                

            # 为每个数据点添加标签
            for CoV in CoVs:
                plot_data.append({
                    'Dataset': dataset_key,
                    'Model': model,
                    'CoV': CoV,
                })
    
    # 转换为DataFrame
    df = pd.DataFrame(plot_data)
    
    # 为每个模型创建小提琴图
    ax = sns.violinplot(
            x='Model', 
            y='CoV',
            hue='Dataset',
            split=True,
            data=df, 
        )
    # ax.set_ylim((-0.01, 0.2))
    # 添加数据点
    sns.stripplot(
        x='Model', 
        y='CoV', 
        hue='Dataset',
        dodge=True,
        data=df,
        palette='dark:black',
        alpha=0.5,   # 设置透明度
        size=6,      # 点的大小
        ax=ax
    )
    # ax.set_title('Distribution of CoV per model', fontsize=20)
    # ax.set_xlabel('Model', fontsize=14, fontweight='bold')
    # ax.set_xlabel('')
    # ax.set_ylabel('Coefficient of Variation (CoV)', fontsize=20, fontweight='bold')

    # ax.set_yticks([0, 0.1, 0.2])
    # ax.set_yticklabels([0, 0.1, 0.2], fontsize=16)
    # ax.tick_params(axis='x', labelsize=16)
    # ax.tick_params(axis='y', labelsize=12)
    
    # handles, labels = ax.get_legend_handles_labels()
    # ax.legend(handles[:2], ('Background 0', 'Background 255'), fontsize=14,)  # 只显示前两个图例项（背景类型）

    plt.tight_layout()

    # plt.savefig(json_name.replace('.json', '_violin.png'), dpi=300)
    plt.show()


def evaluate_for_simulated_square(results, models):
    
    with open(groundtruth_pth, 'r') as f:
        groundtruth_dict = json.load(f)
        gt = groundtruth_dict['basic test\\looming square\\light_looming.mp4']['ground_truth']


    C_ge_0 = {model: [] for model in models}
    C_le_0 = {model: [] for model in models}
    dataset_key = 'contrast_looming'
    for vid_name in results[models[0]][dataset_key].keys():
        fore_contrast, back_contrast = vid_name[19:-4].split('_bgr=')[:2]
        _contrast = (int(fore_contrast) - int(back_contrast))/255

        for model in models:
            iou = calculate_IoU_for_PSP(results[model][dataset_key][vid_name], groundtruth=gt)
            
            if _contrast > 0:
                C_ge_0[model].append(iou)
            elif _contrast < 0:
                C_le_0[model].append(iou)

    return C_ge_0, C_le_0


def evaluate_for_natural_seq(results, models):
    def _calculate_bg_contrast(bg_name):
        BG_FOLDER = os.path.abspath(os.path.join(itemPth, '..', '7_Dataset', 
                                                  'data-from-LGMD_P_ON_OFF'))
        bg_path = os.path.join(BG_FOLDER, bg_name+'.jpg')
        bg_img = cv2.imread(bg_path, cv2.IMREAD_GRAYSCALE).astype(float)
        return np.mean(bg_img)
    
    with open(groundtruth_pth, 'r') as f:
        groundtruth_dict = json.load(f)
        gt = groundtruth_dict['looming square against nature scenarios\\natural-0001_120515-142521_tonemapped-fgr_gray=0.mp4']['ground_truth']


    C_ge_0 = {model: [] for model in models}
    C_le_0 = {model: [] for model in models}
    dataset_key = 'looming_square_against_nature_scenarios'
    for vid_name in results[models[0]][dataset_key].keys():
        bg_name, _fore = vid_name.split('-fgr_gray=')
        fore_contrast = (float(_fore[:-4]) + 1)/2 * 255
        back_contrast = _calculate_bg_contrast(bg_name[8:])
        _contrast = (int(fore_contrast) - int(back_contrast))/255
        for model in models:
            iou = calculate_IoU_for_PSP(results[model][dataset_key][vid_name], groundtruth=gt)

            if _contrast > 0:
                C_ge_0[model].append(iou)
            elif _contrast < 0:
                C_le_0[model].append(iou)

    return C_ge_0, C_le_0


def evaluate_for_driving_seq(results, models):
    with open(groundtruth_pth, 'r') as f:
        groundtruth_dict = json.load(f)
        gt = groundtruth_dict['looming ball against driving sense\\c02-dark2.mp4']['ground_truth']
    with open(os.path.join(os.path.dirname(__file__), 
                           'each_contrast_for_looming_ball_against_driving_sense.json'), 'r') as f:
        each_contrast = json.load(f)

    C_ge_0 = {model: [] for model in models}
    C_le_0 = {model: [] for model in models}
    dataset_key = 'looming_ball_against_driving_sense'
    for vid_name in results[models[0]][dataset_key].keys():
        contrast = each_contrast[vid_name]

        for model in models:
            iou = calculate_IoU_for_PSP(results[model][dataset_key][vid_name], groundtruth=gt)
            if contrast > 0:
                C_ge_0[model].append(iou)
            elif contrast < 0:
                C_le_0[model].append(iou)

    return C_ge_0, C_le_0


def calculate_contrast_for_looming_ball_against_driving_sense():
    vid_folder = os.path.abspath(os.path.join(itemPth, '..', '7_Dataset', 
                                              'lgmd_benchmark', 'looming ball against driving sense'))
    img_folder = os.path.abspath(os.path.join(itemPth, '..', '7_Dataset', 'color balls approaching'))


    def _calculate_contrast_for_single_video(vid_name, vid_folder, img_folder):
        _, fg_name = vid_name.split('-')
        raw_vid = cv2.VideoCapture(os.path.join(vid_folder, vid_name))
        contrasts = []
        for i in range(71):
            ret, frame = raw_vid.read()
            if not ret:
                break
            fg_path = os.path.join(img_folder, f'l-{fg_name[:-4]}-PNGs', f'frame_{i:06d}.png')
            fg_img = cv2.imread(fg_path, cv2.IMREAD_UNCHANGED)  # 读入所有通道，包括alpha
            if fg_img is not None and fg_img.shape[2] == 4:
                mask = fg_img[:, :, 3]  # alpha通道作为mask
                # mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
            else:
                mask = None  # 没有alpha通道时为None
                
            fg_contrast = np.mean(frame[mask==255]) if mask is not None else np.mean(frame)
            bg_contrast = np.mean(frame[mask==0]) if mask is not None else np.mean(frame)

            contrast = (fg_contrast - bg_contrast) / 255  # 归一化到[-1, 1]
            contrasts.append(contrast)

        raw_vid.release()
        
        return np.mean(contrasts)
    
    
    dataset_contrasts = {}
    for vid_name in tqdm(os.listdir(vid_folder)):
        if not vid_name.endswith('.mp4'):
            continue
        contrast = _calculate_contrast_for_single_video(vid_name, vid_folder, img_folder)
        dataset_contrasts[vid_name] = contrast

    with open(os.path.join(os.path.dirname(__file__), 'each_contrast_for_looming_ball_against_driving_sense.json'), 'w') as f:
        json.dump(dataset_contrasts, f, indent=4)


def comparison_performance(results = None, models=MODEL_LIST, is_print=True):
    if results is None:
        file_name = os.path.join(os.path.dirname(__file__), f'{file_name_without_ext}.json')
        with open(file_name, 'r') as f:
            results = json.load(f)

    iou1_gt_0, iou1_lt_0 = evaluate_for_simulated_square(results, models)
    iou2_gt_0, iou2_lt_0 = evaluate_for_natural_seq(results, models)
    iou3gt_0, iou3lt_0 = evaluate_for_driving_seq(results, models)

    mean_list = {model: [] for model in models}
    output_table = PrettyTable()
    output_table.max_width = 15  # 所有列最大宽度为20个字符
    output_table.title = "Performance Comparison under Different Contrast Conditions with IoU(%)"
    output_table.field_names = ["Sense", "Contrast", "Num of videos"] + models

    # square against static
    row0 = ["Square", "C>0", len(iou1_gt_0[models[0]]),]
    row1 = ["Static", "C<0", len(iou1_lt_0[models[0]]),]
    row2 = ["Square", "C>0", len(iou2_gt_0[models[0]]),]
    row3 = ["Nature", "C<0", len(iou2_lt_0[models[0]]),]
    row4 = ["Ball", "C>0", len(iou3gt_0[models[0]]),]
    row5 = ["Driving", "C<0", len(iou3lt_0[models[0]]),]
    


    for model in models:
        row0.append(f"{np.mean(iou1_gt_0[model])*100:.1f}±{np.std(iou1_gt_0[model])*100:.1f}")
        mean_list[model].extend(iou1_gt_0[model])

        row1.append(f"{np.mean(iou1_lt_0[model])*100:.1f}±{np.std(iou1_lt_0[model])*100:.1f}")
        mean_list[model].extend(iou1_lt_0[model])

        row2.append(f"{np.mean(iou2_gt_0[model])*100:.1f}±{np.std(iou2_gt_0[model])*100:.1f}")
        mean_list[model].extend(iou2_gt_0[model])

        row3.append(f"{np.mean(iou2_lt_0[model])*100:.1f}±{np.std(iou2_lt_0[model])*100:.1f}")
        mean_list[model].extend(iou2_lt_0[model])

        row4.append(f"{np.mean(iou3gt_0[model])*100:.1f}±{np.std(iou3gt_0[model])*100:.1f}")
        mean_list[model].extend(iou3gt_0[model])

        row5.append(f"{np.mean(iou3lt_0[model])*100:.1f}±{np.std(iou3lt_0[model])*100:.1f}")
        mean_list[model].extend(iou3lt_0[model])

    output_table.add_rows([row0, row1, row2, row3, row4, row5])
        

    # mean
    row6 = ["Weighted Average", "-", len(mean_list[models[0]]),]
    contrast_weighted_ave = {}
    for model in models:
        _mean = np.mean(mean_list[model]) * 100
        row6.append(f"{_mean:.1f}")
        contrast_weighted_ave[model] = _mean
    output_table.add_row(row6)

    if is_print:
        print(output_table)

    return contrast_weighted_ave



def BasicLGMD_class_api():
    model = BasicLGMD()
    model.to(DEVICE)
    return model, 'forward'

def BiSTS_LGMD_class_api():
    model = BiSTS_LGMD()
    model.to(DEVICE)
    return model, 'forward'

def LGMD_P_ON_OFF_class_api():
    model = LGMD_P_ON_OFF()
    model.to(DEVICE)
    return model, 'forward'


def main():
    # 创建函数映射字典
    api_functions = {
        'BasicLGMD': BasicLGMD_class_api,
        'LGMD_P_ON_OFF': LGMD_P_ON_OFF_class_api,
        'BiSTS_LGMD': BiSTS_LGMD_class_api,
    }

    results = {model_name: None for model_name in MODEL_LIST}
    for model_name in MODEL_LIST:
        class_api = api_functions[model_name]
        results[model_name] = inference_model(class_api)

    save_file = custom_serialize(results, indent=2)
    with open(RAW_DATA_FILE_NAME, 'w') as f:
        f.write(save_file)


if __name__ == "__main__":
    # main()

    # visulize_var()

    # visulize_violin_plot()

    # calculate_contrast_for_looming_ball_against_driving_sense()

    comparison_performance()

    