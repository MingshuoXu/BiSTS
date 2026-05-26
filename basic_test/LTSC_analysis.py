import os
import sys
CUR_DIR = os.path.dirname(__file__) # 当前文件所在目录
sys.path.append(os.path.dirname(CUR_DIR)) # 添加上级目录到路径中


import cv2
import numpy as np
from copy import deepcopy
from tqdm import tqdm
import torch
import prettytable
from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter
import json


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# DEVICE = 'cpu'
from model import bists_lgmd, existing_lgmd
    
def run(vidDict):
    '''
    主函数，读取视频并处理
    Args:
        BiSTS_modelDict: 模型字典，包含模型名称和参数
        vidPth: 视频路径
    '''

    res = {}

    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    
    fig, axs = plt.subplots(2, 5, figsize=(15, 6))
    panel_axes = np.empty((2, 5), dtype=object)
    ax_mid_right_dark = None
    ax_mid_right_light = None

    for i in range(5):
        axs[1, i].axis('off')  # 隐藏第二行的坐标轴

    for i, (vidName, vidPth) in tqdm(enumerate(vidDict.items()), total=len(vidDict), desc="Processing Videos"):
        vid_name_lower = vidName.lower()
        file_name = os.path.basename(vid_name_lower)

        # 行：dark在上(light在下)；列：5种运动类型
        row = 0 if 'dark' in file_name else 1
        if 'looming' in vid_name_lower:
            col = 0
        elif 'receding' in vid_name_lower or 'reveding' in vid_name_lower:
            col = 1
        elif 'translating' in vid_name_lower:
            col = 2
        elif 'elongating' in vid_name_lower:
            col = 3
        elif 'shortening' in vid_name_lower:
            col = 4
        else:
            raise ValueError(f'Unknown motion type in video name: {vidName}')

        ax = axs[row, col]

        # 将每个大子图拆成3行
        panel_spec = ax.get_subplotspec()
        ax.remove()
        # 通过增加一个中间空白行，仅拉开第1行和第2行的间距
        subgs = panel_spec.subgridspec(5, 1, hspace=0.05, height_ratios=[1.5, 0.15, 1, 0.05, 1])
        ax_top = fig.add_subplot(subgs[0, 0])
        ax_mid = fig.add_subplot(subgs[2, 0], sharex=ax_top)
        ax_bot = fig.add_subplot(subgs[4, 0], sharex=ax_top)
        panel_axes[row, col] = {
            'top': ax_top,
            'mid': ax_mid,
            'bot': ax_bot,
        }
        is_first_panel = (col == 0)

        for _ax in (ax_top, ax_mid, ax_bot):
            _ax.spines['top'].set_visible(False)
            _ax.spines['right'].set_visible(False)
            _ax.grid(True, linestyle='--', linewidth=0.7, alpha=0.45)

        # 第一、二行不显示x轴
        ax_top.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
        ax_mid.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
        ax_bot.tick_params(axis='x', which='both', bottom=True, labelbottom=True)



        objVid = cv2.VideoCapture(vidPth)

        # 创建模型对象
        BiSTS_model = bists_lgmd.BiSTS_LGMD()
        BiSTS_model.to(DEVICE)
        LGMD_model = existing_lgmd.BasicLGMD()
        LGMD_model.to(DEVICE)

        # 读取视频
        temporal_distinctiveness_list = []
        spatiotemporal_continuity_list = []
        scalar_value_list = []

        td_list = []
        sc_list = []
        sd_list = []
        BiSTS_response = []
        LGMD_response = []
        last_frame = None
        while objVid.isOpened():
            ret, colorImg = objVid.read()
            if not ret:
                break

            gray = cv2.cvtColor(colorImg, cv2.COLOR_BGR2GRAY)
            if row == 0:  # dark
                scalar_value_list.append(1-np.mean(gray)/255)  # 记录当前帧的平均亮度作为对比度指标
            else:  # light
                scalar_value_list.append(np.mean(gray)/255)
            gray_tensor = torch.from_numpy(gray).float().unsqueeze(0).unsqueeze(0).to(DEVICE)

            BiSTS_res = BiSTS_model.forward(gray_tensor)
            BiSTS_response.append(BiSTS_res)
            LGMD_res = LGMD_model.forward(gray_tensor)
            LGMD_response.append(LGMD_res.item())

            if last_frame is None:
                last_frame = deepcopy(colorImg)
                td_list.append(0)
                sc_list.append(0)
                sd_list.append(0)
            else:
                frame_diff = cv2.absdiff(colorImg, last_frame)
                last_frame = deepcopy(colorImg)
                sum_diff = np.mean(frame_diff)
                if sum_diff > 0.01:  # 仅在帧有变化时记录指标
                    temporal_distinctiveness_list.append(BiSTS_model.LTSC.invNumOfTem)
                    spatiotemporal_continuity_list.append(BiSTS_model.LTSC.seqNumOfSTem) 

                td_list.append(BiSTS_model.LTSC.invNumOfTem)
                sc_list.append(BiSTS_model.LTSC.seqNumOfSTem)
                sd_list.append(sum_diff/255)

        objVid.release() 

        res[vidName] = {
            'temporal_distinctiveness_list': temporal_distinctiveness_list,
            'spatiotemporal_continuity_list': spatiotemporal_continuity_list,
            'scalar_value_list': scalar_value_list
        }

        # 第一行：不同的输出
        ax_top.plot(LGMD_response, label='LGMD response (Conventional)', 
                    color='#0072B2', linestyle='--', alpha=0.7, linewidth=1.5)
        ax_top.plot(BiSTS_response, label='BiSTS response (Proposed)', 
                    color='#D55E00', linestyle='-', linewidth=2)
        ax_top.set_ylim(0.4, 1.1)

        # 第二行：SV / SD
        ax_mid.plot(scalar_value_list, label=f'Object Size (%)', 
                    color='black', linestyle='-', linewidth=1.5)
        ax_mid.set_ylim(-0.1, 1.1)
        ax_mid_right = ax_mid.twinx()
        ax_mid_right.plot(sd_list, label='Change Size (%)', 
                          color='red', linestyle='-', linewidth=1.5)
        ax_mid_right.set_ylim(-0.05, 0.3)
        ax_mid_right.spines['top'].set_visible(False)
        # ax_mid_right.spines['right'].set_visible(is_first_panel)
        ax_mid_right.spines['right'].set_color('red')
        ax_mid_right.tick_params(axis='y', which='both', colors='red', right=True, labelright=True)
        if not is_first_panel:
            ax_mid_right.tick_params(axis='y', which='both', right=True, labelright=False)
            # ax_mid_right.tick_params(axis='y', which='both', left=True, labelleft=False)

        # 第三行：TD / SC
        ax_bot.plot(td_list, label='Temporal Distinctiveness (inverse num of Tem)', 
                    color='#009E73', linestyle='-.', linewidth=1.5)
        ax_bot.plot(sc_list, label='Spatiotemporal Continuity (sequence num of SpaTem)', 
                    color='#CC79A7', linestyle='-.', linewidth=1.5)
        td_arr = np.asarray(td_list)
        sc_arr = np.asarray(sc_list)
        effective_x = ((td_arr >= 2 / 3) & (sc_arr >= 0.05)) | \
                      ((td_arr >= 0.5) & (sc_arr >= 2 / 3)) | \
                      (td_arr >= 0.95)

        if len(td_arr) > 0:
            x_idx = np.arange(len(td_arr))
            ax_bot.fill_between(
                x_idx,
                -0.1,
                1.02,
                where=effective_x,
                color='#F0E442',
                alpha=0.5,
                label='Activation Region'
            )

        ax_bot.set_ylim(-0.1, 1.1)
        ax_bot.set_xlim(0, 50)
        if col == 0:  
            ax_bot.set_xticks([10, 20, 30, 40, 50])
        else:  
            ax_bot.set_xticks([10, 20, 30, 40, 50])
            ax_bot.set_xticklabels([])

        if is_first_panel and row == 0:
            ax_mid_right_dark = ax_mid_right
        if is_first_panel and row == 1:
            ax_mid_right_light = ax_mid_right


        if col == 0:
            ax_mid.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
            ax_mid_right.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
            ax_bot.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))

        # 仅保留第一个面板的y轴标记
        if not is_first_panel:
            ax_top.tick_params(axis='y', which='both', labelleft=False)
            ax_mid.tick_params(axis='y', which='both', labelleft=False)
            # ax_mid_right.tick_params(axis='y', which='both', labelleft=False)
            ax_bot.tick_params(axis='y', which='both', labelleft=False)
        
        
        
    # 为每一列添加标题（5种运动）
    col_labels = ['Looming', 'Receding', 'Translation', 'Elongating', 'Shortening']
    for col, label in enumerate(col_labels):
        panel_axes[0, col]['top'].text(0.5, 1.1, label, transform=panel_axes[0, col]['top'].transAxes,
                                       ha='center', fontsize=16)

    # 第一行第3列面板添加x轴标题
    panel_axes[0, 0]['bot'].set_xlabel('Time (Frames)', fontsize=12, labelpad=3)
    panel_axes[0, 0]['bot'].set_yticks([0, 1])
    panel_axes[0, 0]['bot'].set_yticklabels(['0', '100%'])

    # 第一行第1列面板添加y轴标题
    panel_axes[0, 0]['top'].set_ylabel('Response', fontsize=12, labelpad=15)
    panel_axes[0, 0]['mid'].set_ylabel('Object\nSize', fontsize=12, rotation=90)
    panel_axes[0, 0]['mid'].yaxis.set_label_coords(-0.15, 0.5)
    panel_axes[0, 0]['mid'].set_yticks([0, 1])
    panel_axes[0, 0]['mid'].set_yticklabels(['0', '100%'])
    if ax_mid_right_dark is not None:
        ax_mid_right_dark.set_ylabel('Change Size', fontsize=12, color='red', rotation=270)
        ax_mid_right_dark.yaxis.set_label_coords(1.2, 0.5)
        ax_mid_right_dark.set_yticks([0, 0.25])
        ax_mid_right_dark.set_yticklabels(['0', '25%'])
    panel_axes[0, 0]['bot'].set_ylabel('Ratio', fontsize=12, labelpad=3)

    # 第一行第3列面板添加x轴标题
    panel_axes[1, 0]['bot'].set_xlabel('Time (Frames)', fontsize=12, labelpad=3)
    panel_axes[1, 0]['bot'].set_yticks([0, 1])
    panel_axes[1, 0]['bot'].set_yticklabels(['0', '100%'])

    # 第一行第1列面板添加y轴标题
    panel_axes[1, 0]['top'].set_ylabel('Response', fontsize=12, labelpad=15)
    panel_axes[1, 0]['mid'].set_ylabel('Object\nSize', fontsize=12, rotation=90)
    panel_axes[1, 0]['mid'].yaxis.set_label_coords(-0.15, 0.5)
    panel_axes[1, 0]['mid'].set_yticks([0, 1])
    panel_axes[1, 0]['mid'].set_yticklabels(['0', '100%'])
    if ax_mid_right_light is not None:
        ax_mid_right_light.set_ylabel('Change Size', fontsize=12, color='red', rotation=270)
        ax_mid_right_light.yaxis.set_label_coords(1.2, 0.5)
        ax_mid_right_light.set_yticks([0, 0.25])
        ax_mid_right_light.set_yticklabels(['0', '25%'])
    panel_axes[1, 0]['bot'].set_ylabel('Ratio', fontsize=12, labelpad=3)
    

    # 将Dark / Light移动到整图最右侧
    fig.text(0.992, 0.75, 'Dark', ha='right', va='center', fontsize=16, rotation=270)
    fig.text(0.992, 0.35, 'Light', ha='right', va='center', fontsize=16, rotation=270)
    
    # 获取第一个子图三行的lines用于创建统一legend
    handles, labels = panel_axes[0, 0]['top'].get_legend_handles_labels()
    handles_mid, labels_mid = panel_axes[0, 0]['mid'].get_legend_handles_labels()
    handles.extend(handles_mid)
    labels.extend(labels_mid)
    if ax_mid_right_dark is not None:
        handles_right, labels_right = ax_mid_right_dark.get_legend_handles_labels()
        handles.extend(handles_right)
        labels.extend(labels_right)

    handles_bot, labels_bot = panel_axes[0, 0]['bot'].get_legend_handles_labels()
    handles.extend(handles_bot)
    labels.extend(labels_bot)

    
    fig.legend(handles, labels, loc='lower center', 
               bbox_to_anchor=(0.5, 0), ncol=4, columnspacing=5.0,
               fontsize=13)
    plt.tight_layout(rect=[0, 0.12, 0.985, 1])
    plt.savefig(os.path.join(CUR_DIR, 'LTSC_analysis_results.png'), dpi=600)
    # plt.show()
    

    with open(os.path.join(CUR_DIR, 'LTSC_analysis_results.json'), 'w') as f:
        json.dump(res, f, indent=4)



def show_table():
    with open(os.path.join(CUR_DIR, 'LTSC_analysis_results.json'), 'r') as f:
        import json
        res = json.load(f)

    table = prettytable.PrettyTable()
    table.field_names = ["Video Name", "TD (mean)", "TD (max)", "SC (mean)", "SC (max)", "GAP (mean)", "GAP (max)"]

    for vidName, metrics in res.items():
        temporal_distinctiveness_list = metrics['temporal_distinctiveness_list']
        spatiotemporal_continuity_list = metrics['spatiotemporal_continuity_list']
        scalar_value_list = metrics['scalar_value_list']
        table.add_row([vidName, 
                    f'{np.mean(temporal_distinctiveness_list): .3f}', 
                    f'{np.max(temporal_distinctiveness_list): .3f}', 
                    f'{np.mean(spatiotemporal_continuity_list): .3f}',
                    f'{np.max(spatiotemporal_continuity_list): .3f}',
                    f'{np.mean(scalar_value_list): .3f}',
                    f'{np.max(scalar_value_list): .3f}'
                    ])

    print(table)
    print(table.get_latex_string())


def _update_dict(original_dict, vidFatherPth, subPth):
    """更新字典，添加新键值对"""
    original_dict.update({
        subPth: os.path.join(vidFatherPth, subPth)
    })        # 更新副本


def selete_basic_video():
    # 视频路径字典
    vidFatherPth = os.path.join(os.path.dirname(os.path.dirname(CUR_DIR)), '7_Dataset',
                                'lgmd_benchmark', 'basic test')
    vidDict = {}

    # approaching
    _update_dict(vidDict, vidFatherPth, os.path.join('looming square', 'dark_looming.mp4'))
    _update_dict(vidDict, vidFatherPth, os.path.join('looming square', 'light_looming.mp4'))
    # receding
    _update_dict(vidDict, vidFatherPth, os.path.join('receding square', 'dark_reveding.mp4'))
    _update_dict(vidDict, vidFatherPth, os.path.join('receding square', 'light_receding.mp4'))
    # translation
    _update_dict(vidDict, vidFatherPth, os.path.join('translating bar', 'light_translating_vertical.mp4'))
    _update_dict(vidDict, vidFatherPth, os.path.join('translating bar', 'dark_translating_vertical.mp4'))
    # elongating
    _update_dict(vidDict, vidFatherPth, os.path.join('linear deformation', 'dark_elongating_vertical.mp4'))
    _update_dict(vidDict, vidFatherPth, os.path.join('linear deformation', 'light_elongating_vertical.mp4'))
    # shortening
    _update_dict(vidDict, vidFatherPth, os.path.join('linear deformation', 'dark_shortening_vertical.mp4'))
    _update_dict(vidDict, vidFatherPth, os.path.join('linear deformation', 'light_shortening_vertical.mp4'))


    return vidDict




if __name__ == '__main__':
    
    vidDict = selete_basic_video()

    run(vidDict)

    show_table()


