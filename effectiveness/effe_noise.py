import os
import sys
itemPth = os.path.dirname(os.path.dirname(__file__))
sys.path.append(itemPth)
file_name_without_ext = os.path.splitext(os.path.basename(__file__))[0]
import textwrap

import cv2
from tqdm import tqdm
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
# 全局设置字体
plt.rcParams['font.family'] = 'Times New Roman'  # 设置字体家族
plt.rcParams['font.size'] = 12                   # 设置字体大小
plt.rcParams['axes.titlesize'] = 12              # 标题字体大小
plt.rcParams['axes.labelsize'] = 12              # 坐标轴标签字体大小
plt.rcParams['xtick.labelsize'] = 10             # x轴刻度字体大小
plt.rcParams['ytick.labelsize'] = 10             # y轴刻度字体大小
plt.rcParams['legend.fontsize'] = 10             # 图例字体大小
from skimage.util import random_noise
import concurrent.futures
from prettytable import PrettyTable
import torch


from model.bists_lgmd import BiSTS_LGMD, BiSTS_LGMD_N # type: ignore
from model.existing_lgmd import BasicLGMD, pLGMD # type: ignore
from utils import calculate_IoU_for_PSP, custom_serialize # type: ignore

BENCHMARK_FOLDER = os.path.abspath(os.path.join(itemPth, '..', '7_Dataset', 
                                                  'lgmd_benchmark'))
NOISE_VIDEOS_FOLDER = os.path.abspath(os.path.join(itemPth, '..', '7_Dataset', 'BiSTS_extend',
                                                   'effe_noise_videos'))

RAW_DATA_FILE_NAME = os.path.join(os.path.dirname(__file__), file_name_without_ext+'.json')

NOISE_CLASSES = ['gaussian', 'salt-and-pepper']
NOISE_INTENSITIES = [0.001, 0.003, 0.005, 0.01, 0.1]

MODEL_LIST = ['BasicLGMD', 'pLGMD', 'BiSTS_LGMD']
RAW_INPUT_VIDEOS = {
    'ha-dark1': os.path.join(BENCHMARK_FOLDER, 'looming ball against static bg', 'ha-dark1.mp4'),
    'dark_looming': os.path.join(BENCHMARK_FOLDER, 'basic test', 'looming square', 'dark_looming.mp4'),
    'tc08': os.path.join(BENCHMARK_FOLDER, 'vehicle against driving sense', 'tc08.mp4'),
}
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

GT_PTH = os.path.join(itemPth, '..', '7_Dataset', 'lgmd_benchmark', 
                                  'annotations.json')

def _inference(vidPth, model_api=None):
    objVid = cv2.VideoCapture(vidPth)

    # get process function
    class_obj, process_fun_name = model_api()
    # process_func = class_obj.process
    process_func = getattr(class_obj, process_fun_name)

    results = []

    while objVid.isOpened():
        ret, colorImg = objVid.read()
        if not ret:
            break

        gray = cv2.cvtColor(colorImg, cv2.COLOR_BGR2GRAY)
        gray_tensor = torch.from_numpy(gray).float().unsqueeze(0).unsqueeze(0).to(DEVICE)

        k = process_func(gray_tensor)
        if isinstance(k, torch.Tensor):
            k = k.item()
        results.append(k)

    objVid.release()

    return results


def inference_model(model_api, max_workers=10):

    results = {f'{vid_key}-{noise}-{intensy}': None for noise in NOISE_CLASSES \
               for intensy in NOISE_INTENSITIES for vid_key in RAW_INPUT_VIDEOS.keys()}
    results['raw'] = None
               
    features = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:

        for vid_key, vid_name in RAW_INPUT_VIDEOS.items():
            feature = executor.submit(_inference, vid_name, model_api)
            feature.key = f'{vid_key}-raw'
            features.append(feature)
   
            for noise in NOISE_CLASSES:
                for intensy in NOISE_INTENSITIES:
                        vid_pth = os.path.join(NOISE_VIDEOS_FOLDER, vid_key,
                                            f'{vid_key}_{noise}_{intensy}.mp4')
                        feature = executor.submit(_inference, vid_pth, model_api)
                        feature.key = f'{vid_key}-{noise}-{intensy}'

                        features.append(feature)

        for feature in tqdm(concurrent.futures.as_completed(features), total=len(features), desc=f'Effectiveness Noise Inference - {model_api.__name__}'):
            res = feature.result()
            results[feature.key] = res

    return results


# Shade ground-truth alert regions where gt1 == 1 (apply to first column and noise subplots)
def _shade_gt(ax, gt_arr, color='red', alpha=0.15):
    gt_bool = np.array(gt_arr) == 1
    if gt_bool.size == 0:
        return
    padded = np.concatenate([[0], gt_bool.astype(int), [0]])
    diff = np.diff(padded)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    for s, e in zip(starts, ends):
        ax.axvspan(s, e, color=color, alpha=alpha, zorder=0)


def visulize_response(vid_name):
    
    json_name = os.path.join(os.path.dirname(__file__), 
                             f'{file_name_without_ext}.json')
    with open(json_name, 'r') as f:
        results = json.load(f)

    
    with open(GT_PTH, 'r') as f:
        groundtruth_dict = json.load(f)
        gt1 = groundtruth_dict['looming ball against static bg\\ha-dark1.mp4']['ground_truth']


    fig, axs = plt.subplots(len(NOISE_INTENSITIES), 3, figsize=(9, 5))

    ax = axs[0, 0]

    for model in MODEL_LIST:
        
        yData = results[model][f'{vid_name}-raw']

        ax.plot(yData)

    ax.set_ylim([0.5, 1.02])
    # ax.set_xlim([0, 40])
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    
    # 添加网格并设置透明度
    ax.grid(True, alpha=0.3)  # alpha值范围0-1，0完全透明，1完全不透明


    # 去除顶部和右边的边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
            

    ax.set_xlabel('Frame', fontsize=14)
    ax.set_ylabel('Response', fontsize=14)

    

    for j, intensy in enumerate(NOISE_INTENSITIES):
        for i, noise in enumerate(NOISE_CLASSES):
        
            ax = axs[j, i+1] 

            for model in MODEL_LIST:
                yData = results[model][f'{vid_name}-{noise}-{intensy}']
                ax.plot(yData)

            ax.set_ylim([0.5, 1.02])
            # ax.set_xlim([0, 40])
            ax.tick_params(axis='x', labelsize=12)
            ax.tick_params(axis='y', labelsize=12)
            
            # 添加网格并设置透明度
            ax.grid(True, alpha=0.3)  # alpha值范围0-1，0完全透明，1完全不透明

    
            # 去除顶部和右边的边框
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
                    
            # if i > 0:
            #     ax.set_xlabel('Frame', fontsize=14)
            # if j == 0:
            #     ax.set_ylabel('Response', fontsize=14)
            
    # apply shading to the raw column and all noise subplots
    _shade_gt(axs[0, 0], gt1)
    for row in range(len(NOISE_INTENSITIES)):
        for col in (1, 2):
            _shade_gt(axs[row, col], gt1)


    for _ in range(1, len(NOISE_INTENSITIES)):
        axs[_, 0].axis('off')

    # 设置图例（包含 Groundtruth 阴影标记）
    handles = [plt.Line2D([0], [0.5], color=f"C{i}", lw=2) for i in range(len(MODEL_LIST))]
    gt_patch = Patch(facecolor='red', alpha=0.3)
    handles.append(gt_patch)
    _legend = [model.replace('_LGMD', ' (Ours)') for model in MODEL_LIST] + ['Alarm Area (GT)']
    plt.figlegend(handles, _legend,  
               fontsize=14, title_fontsize=14,
               bbox_to_anchor=(0.1, 0.35, 0.1, 0.3),
               loc='lower left', ncol=1,
               handlelength=1.5,  # 图例线条长度
               fancybox=True,  # 圆角边框
               framealpha=0.9, # 透明度
            )

    plt.tight_layout()
    plt.subplots_adjust(top = 0.9, right = 0.96)

    
    # 设置列标题
    title_loca = [0.2, 0.5, 0.8]
    for j, title in enumerate(['without noise', 'Gaussian Noise', 'Salt & Pepper Noise']):
        fig.text(title_loca[j], 0.95, title, ha='center', va='center', 
                fontsize=16)
    
    # 设置行标题
    for i, title in enumerate(NOISE_INTENSITIES):
        # 在第一列左侧添加行标题
        fig.text(0.98, 0.84 - i*0.18, title, ha='center', va='center', 
                fontsize=16, rotation=-90)

    plt.savefig(os.path.join(os.path.dirname(__file__), file_name_without_ext+'_response.png'), dpi=300)
    plt.show()


def visulize_comparison():
    _MODEL_LIST = ['BasicLGMD', 'pLGMD', 'BiSTS_LGMD']

    json_name = os.path.join(os.path.dirname(__file__), f'{file_name_without_ext}.json')
    with open(json_name, 'r') as f:
        results = json.load(f)

    fig, axs = plt.subplots(1, 3, figsize=(11, 2.6))
    with open(GT_PTH, 'r') as f:
        groundtruth_dict = json.load(f)
        gt1 = groundtruth_dict['looming ball against static bg\\ha-dark1.mp4']['ground_truth']

    ax = axs[0]
    for i, model in enumerate(_MODEL_LIST):
        yData = results[model]['ha-dark1-raw']
        ax.plot(yData[30:])
            
    ax.set_ylim([0.5, 1.02])
    ax.tick_params(axis='x')
    ax.tick_params(axis='y')
        

    _shade_gt(ax, gt1[30:])
    
    # 添加网格并设置透明度
    ax.grid(True, alpha=0.3)  # alpha值范围0-1，0完全透明，1完全不透明


    # 去除顶部和右边的边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
            
    ax.set_title('Without Noise')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Response')

     
    for i, noise in enumerate(NOISE_CLASSES):

        ax = axs[i+1] 

        for k, model in enumerate(_MODEL_LIST):
            yData = results[model][f'ha-dark1-{noise}-0.1']
            ax.plot(yData[30:])

        ax.set_ylim([0.5, 1.02])
        ax.tick_params(axis='x')
        ax.tick_params(axis='y')
        
        # 添加网格并设置透明度
        ax.grid(True, alpha=0.3)  # alpha值范围0-1，0完全透明，1完全不透明


        # 去除顶部和右边的边框
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
                
        ax.set_xlabel('Frame')
        if i == 0:
            ax.set_title('10% Gaussian Noise')
        else:
            ax.set_title('10% Salt & Pepper Noise')

        _shade_gt(ax, gt1[30:])
            
    # 设置图例
    handles = [plt.Line2D([0], [0.5], color=f"C{i}", lw=2) for i in range(len(_MODEL_LIST))]
    gt_patch = Patch(facecolor='red', alpha=0.3)
    handles.append(gt_patch)
    _legend = [model.replace('_LGMD', ' (Ours)') for model in _MODEL_LIST] + ['Alarm Area (GT)']
    plt.figlegend(handles, _legend,   
               fontsize=11, title_fontsize=14,
               bbox_to_anchor=(0.05, -0.02, 0.9, 0.1),
               loc='lower center', ncol=4,
               handlelength=1.5,  # 图例线条长度
               fancybox=True,  # 圆角边框
               framealpha=0.9, # 透明度
            )

    plt.tight_layout()
    plt.subplots_adjust(top = 0.88, bottom=0.35)


    plt.savefig(os.path.join(os.path.dirname(__file__), file_name_without_ext+'_comparison.png'), dpi=300)
    plt.show()


def comparison_performance(results=None, models=MODEL_LIST, is_print=True):
    

    if results is None:
        file_name = os.path.join(os.path.dirname(__file__), f'{file_name_without_ext}.json')
        with open(file_name, 'r') as f:
            results = json.load(f)

    with open(GT_PTH, 'r') as f:
        groundtruth_dict = json.load(f)
        gt1 = groundtruth_dict['looming ball against static bg\\ha-dark1.mp4']['ground_truth']
        gt2 = groundtruth_dict['basic test\\looming square\\dark_looming.mp4']['ground_truth']
        gt3 = groundtruth_dict['vehicle against driving sense\\tc08.mp4']['ground_truth']

    output_table = PrettyTable()
    output_table.title = "Performance Comparison under Different Noise Types and Intensities with IoU(%)"
    output_table.field_names = ["Video", "Noise", "Intensity"] + models
    mean_iou = {model: [] for model in models}
    vid_keys = ['ha-dark1', 'dark_looming', 'tc08']
    vid_names = [RAW_INPUT_VIDEOS[key] for key in vid_keys]
    for vid_key, vid_name, groundtruth in zip(vid_keys, vid_names, [gt1, gt2, gt3]):
        
        # raw
        table_line = [vid_key, None, 0]
        for model in models:
            iou = calculate_IoU_for_PSP(results[model][f'{vid_key}-raw'], groundtruth=groundtruth)
            mean_iou[model].append(iou)
            table_line.append(f'{iou*100:.1f}')
        output_table.add_row(table_line)

        # with noise
        for noise in NOISE_CLASSES:
            for intensy in NOISE_INTENSITIES:
                table_line = ['', noise, intensy]
                for model in models:
                    iou = calculate_IoU_for_PSP(results[model][f'{vid_key}-{noise}-{intensy}'], groundtruth=groundtruth)
                    mean_iou[model].append(iou)
                    table_line.append(f'{iou*100:.1f}')
                output_table.add_row(table_line)
    output_table.add_row(['Weighted Average', '', ''] + [f'{np.mean(mean_iou[model])*100:.1f}' for model in models])

    if is_print:
        print(output_table)

    noise_weighted_ave = {model: np.mean(mean_iou[model])*100 for model in models}

    return noise_weighted_ave


def BasicLGMD_class_api():
    model = BasicLGMD()
    model.to(DEVICE)
    return model, 'forward'

def pLGMD_class_api():
    model = pLGMD()
    model.to(DEVICE)
    return model, 'forward'

def BiSTS_LGMD_class_api():
    model = BiSTS_LGMD()
    model.to(DEVICE)
    return model, 'forward'


def main():
    
    # 创建函数映射字典
    api_functions = {
        'BasicLGMD': BasicLGMD_class_api,
        'pLGMD': pLGMD_class_api,
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

    visulize_response('ha-dark1')

    visulize_comparison()

    # comparison_performance()

