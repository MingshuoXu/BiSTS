import os
import sys
itemPth = os.path.dirname(os.path.dirname(__file__))
sys.path.append(itemPth)
file_name_without_ext = os.path.splitext(os.path.basename(__file__))[0]
import textwrap
from copy import deepcopy

import cv2
import matplotlib.pyplot as plt
# 全局设置字体
plt.rcParams['font.family'] = 'Times New Roman'  # 设置字体家族
plt.rcParams['font.size'] = 12                   # 设置字体大小
plt.rcParams['axes.titlesize'] = 12              # 标题字体大小
plt.rcParams['axes.labelsize'] = 12              # 坐标轴标签字体大小
plt.rcParams['xtick.labelsize'] = 10             # x轴刻度字体大小
plt.rcParams['ytick.labelsize'] = 10             # y轴刻度字体大小
plt.rcParams['legend.fontsize'] = 10             # 图例字体大小
import torch
import numpy as np



from model.bists_lgmd import BiSTS_LGMD # type: ignore
from model.existing_lgmd import BasicLGMD, pLGMD # type: ignore

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
}
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

GT_PTH = os.path.join(itemPth, '..', '7_Dataset', 'lgmd_benchmark', 
                                  'annotations.json')

def _inference(vidPth, model):
    objVid = cv2.VideoCapture(vidPth)

    i = 0
    while objVid.isOpened():
        ret, colorImg = objVid.read()
        if not ret:
            break

        gray = cv2.cvtColor(colorImg, cv2.COLOR_BGR2GRAY)
        gray_tensor = torch.from_numpy(gray).float().unsqueeze(0).unsqueeze(0).to(DEVICE)

        model(gray_tensor)
        i += 1
        if i == 100:
            break


    objVid.release()

    return model.S_opt.cpu().squeeze(0).squeeze(0).numpy()


def inference_model(model):

    results = {f'{vid_key}-{noise}-{intensy}': None for noise in NOISE_CLASSES \
               for intensy in NOISE_INTENSITIES for vid_key in RAW_INPUT_VIDEOS.keys()}

    for vid_key, vid_name in RAW_INPUT_VIDEOS.items():
        results['raw'] = _inference(vid_name, model)

        for noise in NOISE_CLASSES:
            for intensy in NOISE_INTENSITIES:
                vid_pth = os.path.join(NOISE_VIDEOS_FOLDER, vid_key,
                                    f'{vid_key}_{noise}_{intensy}.mp4')
                results[f'{vid_key}-{noise}-{intensy}'] = _inference(vid_pth, model)

    return results


def get_raw_img():
    def _get_raw_img(vidPth):
        objVid = cv2.VideoCapture(vidPth)

        i = 0
        while objVid.isOpened():
            ret, colorImg = objVid.read()
            if not ret:
                break

            i += 1
            if i == 100:
                break

        objVid.release()

        return colorImg

    raw_img = {f'{vid_key}-{noise}-{intensy}': None for noise in NOISE_CLASSES \
               for intensy in NOISE_INTENSITIES for vid_key in RAW_INPUT_VIDEOS.keys()}  
    
    for vid_key, vid_name in RAW_INPUT_VIDEOS.items():
        raw_img['raw'] = _get_raw_img(vid_name)

        for noise in NOISE_CLASSES:
            for intensy in NOISE_INTENSITIES:
                vid_pth = os.path.join(NOISE_VIDEOS_FOLDER, vid_key,
                                    f'{vid_key}_{noise}_{intensy}.mp4')
                raw_img[f'{vid_key}-{noise}-{intensy}'] = _get_raw_img(vid_pth)

    return raw_img


def calculate_psnr_extended(original, processed, max_pixel=255.0):
    # 确保是浮点数运算，避免溢出
    original_new = deepcopy(original).astype(np.float64)
    processed_new = deepcopy(processed).astype(np.float64)
    
    mse = np.mean((original_new - processed_new) ** 2)
    if mse == 0:
        return float('inf')
    
    psnr_val = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr_val


def format_psnr_text(psnr_value):
    if np.isinf(psnr_value):
        return 'PSNR=inf'
    return f'{psnr_value:.2f} dB'



def main():
    BiSTS_LGMD_model = BiSTS_LGMD().to(DEVICE)
    BasicLGMD_model = BasicLGMD().to(DEVICE)
    pLGMD_model = pLGMD().to(DEVICE)
    
    results = {}
    for model, name in zip([BasicLGMD_model, pLGMD_model, BiSTS_LGMD_model], MODEL_LIST):
        print(f"Running inference for {name}...")
        results[name] = inference_model(model)
    raw_img = get_raw_img()

    # 1. 提取所有图像的 key，并将最后一个（原图/无噪声图）挪到第一个
    keys = list(results[MODEL_LIST[0]].keys())
    keys = [keys[-1]] + keys[:-1]

    # 2. 计算右侧三列模型输出的全局最大值和最小值
    # 这样能保证所有彩图的颜色映射范围是相同的，具有绝对的对比意义
    all_outputs = [results[model][key] for model in MODEL_LIST for key in keys]
    global_vmin = min([np.min(out) for out in all_outputs])
    global_vmax = max([np.max(out) for out in all_outputs])
    
    print(f"Visualization unified limits: vmin={global_vmin:.4f}, vmax={global_vmax:.4f}")

    fig, axes = plt.subplots(11, 4, figsize=(7.8, 10.4))
    
    psnr_results = {model: {} for model in MODEL_LIST}
    psnr_input = {}
    for i, key in enumerate(keys):
        psnr_input[key] = calculate_psnr_extended(raw_img[keys[0]], raw_img[key])
        for model in MODEL_LIST:
            psnr_results[model][key] = calculate_psnr_extended(results[model][keys[0]], results[model][key])



    for i, key in enumerate(keys):
        # 原图列保持灰度
        axes[i, 0].imshow(raw_img[key], cmap='gray')
        axes[i, 0].axis('off')
        axes[i, 0].text(
            0.98,
            0.03,
            format_psnr_text(psnr_input[key]),
            fontsize=14,
            color='white',
            ha='right',
            va='bottom',
            transform=axes[i, 0].transAxes,
            bbox=dict(facecolor='black', alpha=0.95, edgecolor='none', pad=1.5),
        )
        
        # 处理最左侧文字标签
        # 做一个防错处理：因为第一行变成了无噪声原图，key 可能不包含特殊字符
        if 'gaussian' in key:
            # 提取强度并格式化，避免浮点数精度导致的超长小数
            noise_intensity = float(key.split('gaussian-')[-1])
            # 使用 :g 自动去掉多余的 0，或者用 .1f 保留一位小数
            show_text = f"Gaussian\n({noise_intensity*100:g}%)"
        elif 'salt-and-pepper' in key:
            noise_intensity = float(key.split('salt-and-pepper-')[-1])
            show_text = f"S&P\n({noise_intensity*100:g}%)"
        else:
            show_text = 'Original'
            
        # 注意：因为你手动加了 \n，textwrap.fill 可能会破坏你的手动换行
        # 如果 width 足够，直接使用 show_text 即可，Ha 设置为 'right' 会自动处理换行对齐
        axes[i, 0].text(-0.25, 0.5, show_text, fontsize=15, 
                        ha='center', va='center', transform=axes[i, 0].transAxes,
                        linespacing=1.2) # 稍微增加行间距，更清晰

        for j, model in enumerate(MODEL_LIST):
            # 3. 换一种更高级的 cmap，比如 'magma', 'viridis', 或 'coolwarm'
            axes[i, j+1].imshow(results[model][key], cmap='coolwarm', vmin=global_vmin, vmax=global_vmax)
            axes[i, j+1].axis('off')
            axes[i, j+1].text(
                0.98,
                0.03,
                format_psnr_text(psnr_results[model][key]),
                fontsize=14,
                color='white',
                ha='right',
                va='bottom',
                transform=axes[i, j+1].transAxes,
                bbox=dict(facecolor='black', alpha=0.95, edgecolor='none', pad=1.5),
            )

        # 竖线位置设在第一列图片右侧外侧 10% 宽度的位置
        line_x_pos = 1.05
        line_color = 'orange'
        dot_color = 'orange'

        # 默认上下延伸短线，以连接同组内的上下行
        y_top = 1.15     
        y_bottom = -0.15 

        # 用于存储该行端点的标记样式和位置，默认为空
        top_dot_type = None
        bottom_dot_type = None

        if i == 0:          # === 第1区块 (Original, 共1行) ===
            y_top = 0.9       # 顶部齐平图片，不向上延伸
            top_dot_type = 'open'
            y_bottom = 0.1   # 向下延伸，正好碰到位于 -0.1 的水平线
            bottom_dot_type = 'open'
        elif 1 <= i <= 5:   # === 第2区块 (Gaussian, 共5行) ===
            if i == 1: 
                y_top = 0.7
                top_dot_type = 'filled'    # 区块第一行，不向上延伸
            if i == 5: 
                y_bottom = 0.3
                bottom_dot_type = 'filled' # 区块最后一行，向下碰到水平线
        elif 6 <= i <= 10:  # === 第3区块 (S&P, 共5行) ===
            if i == 6: 
                y_top = 0.7
                top_dot_type = 'filled'     # 区块第一行，不向上延伸
            if i == 10: 
                y_bottom = 0.3
                bottom_dot_type = 'filled' # 整张图的最后一行，底部齐平图片，不向下延伸

        # 在第一列图片(c=0)的右侧外侧约 10% 宽度的位置 (x=1.1) 画出竖线
        axes[i, 0].plot([1.05, 1.05], [y_bottom, y_top], color='orange', linestyle='--', lw=1.5, 
                        transform=axes[i, 0].transAxes, clip_on=False)
        
        if top_dot_type:
            # 标记点位置 (x=line_x_pos, y=y_top)
            m_style = 'o' # 圆形
            m_size = 6     # 标记大小
            # 如果是空心圈，用深灰边框白色填充；如果是实心点，用深灰填充。
            fc = 'white' if top_dot_type == 'open' else dot_color
            
            axes[i, 0].plot(line_x_pos, y_top, marker=m_style, markersize=m_size, 
                            markeredgecolor=dot_color, markerfacecolor=fc, 
                            transform=axes[i, 0].transAxes, clip_on=False, zorder=2) # 标记点显示在线条上方

        # 3. 画出该行底部的圆圈（如果设置了）
        if bottom_dot_type:
            m_style = 'o'
            m_size = 6
            fc = 'white' if bottom_dot_type == 'open' else dot_color
            
            axes[i, 0].plot(line_x_pos, y_bottom, marker=m_style, markersize=m_size, 
                            markeredgecolor=dot_color, markerfacecolor=fc, 
                            transform=axes[i, 0].transAxes, clip_on=False, zorder=2)

    axes[0, 0].set_title('Input', fontsize=15)
    for k, name in enumerate(MODEL_LIST):
        axes[0, k+1].set_title(name.replace('_LGMD', ''), fontsize=15)

    # plt.tight_layout()
    plt.subplots_adjust(left=0.1, right=1, top=0.97, bottom=0, wspace=0.01, hspace=0.03)
    plt.savefig(os.path.join(os.path.dirname(__file__), 'EI_layer_comparison.png'), dpi=300)
    plt.show()



if __name__ == "__main__":
    main()

