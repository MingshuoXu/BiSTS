import os
import sys
itemPth = os.path.dirname(os.path.dirname(__file__))
sys.path.append(itemPth)
file_name_without_ext = os.path.splitext(os.path.basename(__file__))[0]
import concurrent.futures

import cv2
from tqdm import tqdm
import json
import numpy as np
import statsmodels.api as sm
from skimage.util import random_noise
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
plt.rcParams['font.family'] = 'Times New Roman'  # 设置字体家族
plt.rcParams['font.size'] = 14                   # 设置字体大小
plt.rcParams['axes.titlesize'] = 15              # 标题字体大小
plt.rcParams['axes.labelsize'] = 15              # 坐标轴标签字体大小
plt.rcParams['xtick.labelsize'] = 14             # x轴刻度字体大小
plt.rcParams['ytick.labelsize'] = 14             # y轴刻度字体大小
plt.rcParams['legend.fontsize'] = 13  

from model.bists_lgmd import BiSTS_LGMD, BiSTS_LGMD_N # type: ignore

VID_FOLDER = os.path.abspath(os.path.join(itemPth, '..', '7_Dataset', 
                                                  'lgmd_benchmark', 'balanced one'))
RAW_DATA_FILE_NAME = os.path.join(os.path.dirname(__file__), file_name_without_ext+'.json')
NOISE_CLASSES = ['gaussian', 's&p']
NOISE_INTENSITIES = [i*0.02 for i in range(51)]
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def _add_noise_and_inference(vidPth, noiseType=None, noiseIntensity=None):
    objVid = cv2.VideoCapture(vidPth)

    model = BiSTS_LGMD()
    model.to(DEVICE)

    result = []

    while objVid.isOpened():
        ret, colorImg = objVid.read()
        if not ret:
            break

        gray = cv2.cvtColor(colorImg, cv2.COLOR_BGR2GRAY).astype(float) / 255.0  # Normalize to [0, 1]
        
        if noiseType == 'gaussian':
            gray_with_noise = random_noise(gray, mode='gaussian', var=noiseIntensity)
        elif noiseType == 's&p':
            gray_with_noise = random_noise(gray, mode='s&p', amount=noiseIntensity)
        gray_with_noise *= 255.0

        gray_tensor = torch.from_numpy(gray_with_noise).float().to(DEVICE).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, H, W)
        k = model(gray_tensor)
        result.append(k.item())

    objVid.release()
    return result


def main(max_workers = 1):

    results = {f'{noise}-{intensy}': [None for _ in range(10)] for noise in NOISE_CLASSES for intensy in NOISE_INTENSITIES}

    vidPth = os.path.join(itemPth, '..', '7_Dataset', 
                          'lgmd_benchmark', 'looming ball against static bg', f'ha-dark1.mp4')


    features = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        
        for noise in NOISE_CLASSES:
            if noise == 's&p':
                intensities = NOISE_INTENSITIES
            else:
                intensities = NOISE_INTENSITIES
            for intensy in intensities:
                for repeat in range(10):
                    feature = executor.submit(_add_noise_and_inference, vidPth, noise, intensy)
                    feature.noise = noise
                    feature.intensy = intensy
                    feature.repeat = repeat
                    features.append(feature)


        for future in tqdm(concurrent.futures.as_completed(features), total=len(features)):
            res = future.result()
            results[f'{future.noise}-{future.intensy}'][future.repeat] = res

    with open(RAW_DATA_FILE_NAME, 'w') as f:
        json.dump(results, f, indent=4)


def visualize_boundary():
    from utils import calculate_IoU_for_PSP

    with open(RAW_DATA_FILE_NAME, 'r') as f:
        results = json.load(f)

    groundtruth_pth = os.path.join(itemPth, '..', '7_Dataset', 'lgmd_benchmark', 
                                  'annotations.json')
    with open(groundtruth_pth, 'r') as f:
        groundtruth_dict = json.load(f)
        gt = groundtruth_dict['looming ball against static bg\\ha-dark1.mp4']['ground_truth']

    _, ax = plt.subplots(1, 1, figsize=(6, 3.5))

    curves = {noise:[] for noise in NOISE_CLASSES}
    std_curves = {noise:[] for noise in NOISE_CLASSES}
    # 为每种噪声类型定义颜色和标记
    colors = {'gaussian': 'blue', 's&p': 'red'}
    markers = {'gaussian': '^', 's&p': 'v'}

    for noise in NOISE_CLASSES:
        for i, intensy in enumerate(NOISE_INTENSITIES):

            iou_list = []
            for repeat in range(10):
                iou = calculate_IoU_for_PSP(results[f'{noise}-{intensy}'][repeat], groundtruth=gt)
                iou_list.append(iou)
            
            if i == 0:
                first_value = np.mean(iou_list)
                curves[noise].append(0)
                std_curves[noise].append(np.std(iou_list))
            else:  
                related_iou_list = [(val - first_value) / first_value for val in iou_list]
                curves[noise].append(np.mean(related_iou_list))
                std_curves[noise].append(np.std(related_iou_list))

        # 拟合曲线

        lowess = sm.nonparametric.lowess
        _fix = lowess(curves[noise], NOISE_INTENSITIES, frac=0.2)  # frac 控制局部范围
        curves[f'{noise}_fix'] = _fix[:, 1]
        curves[f'{noise}_fix'][0] = 0  
        curves[f'{noise}_fix'][-1] = curves[noise][-1]


        color = colors[noise]
        marker = markers[noise]
        
        # 绘制拟合曲线
        fitted_data = curves[f'{noise}_fix']
        ax.plot(NOISE_INTENSITIES, fitted_data, 
                label=f'{noise.capitalize()} noise', linewidth=2, color=color)
        
        # 绘制虚化的原始数据点（使用不同标记）
        ax.scatter(NOISE_INTENSITIES, curves[noise], s=20, 
                  alpha=0.4, linestyle='--', color=color, marker=marker) # , label=f'{noise.capitalize()} mean')
        
        # 使用局部标准差创建阴影区域
        local_std = std_curves[noise]
        ax.fill_between(NOISE_INTENSITIES, 
                       np.maximum(-1, fitted_data - local_std), 
                       np.minimum(fitted_data + local_std, 1), 
                       alpha=0.2, color=color,
                    #    label=f'{noise.capitalize()} Var'
                       )
        

    # ax.set_ylim([0, 100])
    ax.set_xlim([0, 1])
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    
    # 添加网格并设置透明度
    ax.grid(True, alpha=0.3)
    
    # 去除顶部和右边的边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_xlabel('Noise intensities') # (var for Gaussian noise or amount for S&P noise)
    ax.set_ylabel('Performance drop') # change compared to noiseless
    ax.set_title('Boundory Study')

    plt.legend(title_fontsize=12, loc='best')
    
    plt.tight_layout()
    # plt.subplots_adjust(left=0.11)
    plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    plt.savefig(os.path.join(os.path.dirname(__file__), file_name_without_ext+'_curve.png'), dpi=300)

    plt.show()


if __name__ == "__main__":

    main() 

    visualize_boundary()

