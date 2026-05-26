import os
import sys
ITEM_PTH = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ITEM_PTH)
file_name_without_ext = os.path.splitext(os.path.basename(__file__))[0]

import json
import torch
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Rectangle
import cv2

plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 14,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
})


RAW_DATA_FILE_NAME = os.path.join(ITEM_PTH, 'lgmd_benchmark', 'run_benchmark.json')
BENCHMARK_PTH = os.path.join(ITEM_PTH, '..', '7_Dataset', 'lgmd_benchmark')
GT_PTH = os.path.join(BENCHMARK_PTH, 'annotations.json')

MODEL_LIST = ['BasicLGMD', 'LGMD_P_ON_OFF', 'pLGMD', 
              'SFA_LGMD',
              'LGMD1_LGMD2_CascadeNetwork_Dual_Channel', 'LGMD2_LGMD1_CascadeNetwork_Dual_Channel',
              'ALGMD', 
              'EMD_LPLC2_GF',
              'BiSTS_LGMD',
              ]


model_name_to_display_name = {
    'BasicLGMD': 'BasicLGMD',
    'LGMD_P_ON_OFF': 'MC-LGMD',
    'pLGMD': 'pLGMD',
    'SFA_LGMD': 'SFA-LGMD',
    'LGMD1_LGMD2_CascadeNetwork_Dual_Channel': 'LGMD1-LGMD2',
    'LGMD2_LGMD1_CascadeNetwork_Dual_Channel': 'LGMD2-LGMD1',
    'ALGMD': 'ALGMD',
    'EMD_LPLC2_GF': 'EMD-LPLC2-GF',
    'BiSTS_LGMD': 'BiSTS (Ours)',
}


def main(video_list):
    with open(RAW_DATA_FILE_NAME, 'r') as f:
        model_response = json.load(f)

    if not os.path.exists(GT_PTH):
        raise FileNotFoundError(f"Groundtruth file not found at {GT_PTH}")
    with open(GT_PTH, 'r') as f:
        groundtruth_dict = json.load(f)

    n_rows = 1 + len(MODEL_LIST)
    n_cols = len(video_list)
    fig = plt.figure(figsize=(3 * n_cols, n_rows))
    outer = fig.add_gridspec(n_rows, n_cols, hspace=0.37, wspace=0.12)

    def read_frame(path, frm_idx):
        try:
            cap = cv2.VideoCapture(path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frm_idx)
            ret, frame = cap.read()
            cap.release()
            if not ret:
                return None
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except Exception:
            return None

    def hide_top_right_spines(ax):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    for col_idx, video_name in enumerate(video_list):
        if video_name not in groundtruth_dict:
            print(f"Warning: {video_name} not found in groundtruth. Skipping.")
            continue
        gt = groundtruth_dict[video_name]['ground_truth']
        video_pth = os.path.join(BENCHMARK_PTH, video_name)

        # determine indices for the 4 special frames
        first_idx = 0
        last_idx = len(gt) - 1
        gt_indices = [i for i, v in enumerate(gt) if v==1]
        first_gt_idx = gt_indices[0] if gt_indices else int(last_idx/3)
        last_gt_idx = gt_indices[-1] if gt_indices else int(last_idx*2/3)
        

        # Determine frame indices based on GT structure
        if last_gt_idx is not None and last_gt_idx == last_idx:
            # If last GT frame is the video end frame
            mid_idx = (first_idx + first_gt_idx) // 2 
            frame_indices = [first_idx, mid_idx, first_gt_idx, last_gt_idx]
        elif last_gt_idx is not None and last_idx < last_gt_idx + 5:
            mid_idx = (first_gt_idx + last_gt_idx) // 2 
            frame_indices = [first_idx, first_gt_idx, mid_idx, last_gt_idx]
        else:
            # Original logic
            frame_indices = [first_idx, first_gt_idx, last_gt_idx, last_idx]

        # top row: four frames in a nested grid for this video column
        top_spec = outer[0, col_idx].subgridspec(1, 4, wspace=0.05)
        for frame_col in range(4):
            ax = fig.add_subplot(top_spec[0, frame_col])
            frm_idx = frame_indices[frame_col]
            frame = read_frame(video_pth, frm_idx) if frm_idx is not None else None

            if frame is not None:
                ax.imshow(frame)
                ax.axis('off')
            else:
                ax.text(0.5, 0.5, 'Frame not available', ha='center', va='center')
                ax.set_xticks([])
                ax.set_yticks([])

            if frame_col == 0:
                ax.set_ylabel(video_name, rotation=0, labelpad=55, va='center')
            ax.set_title(f'{frm_idx+1}', fontsize=14)

        # bottom rows: one model per row for this video column
        for row_idx, model_name in enumerate(MODEL_LIST, start=1):
            ax = fig.add_subplot(outer[row_idx, col_idx])
            display_name = model_name_to_display_name.get(model_name, model_name)
            if model_name not in model_response or video_name not in model_response[model_name]['results']:
                ax.text(0.5, 0.5, f'{display_name}\nno data', ha='center', va='center')
                ax.set_xticks([])
                ax.set_yticks([])
                hide_top_right_spines(ax)
                if col_idx == 0:
                    ax.text(-0.17, 0.5, display_name, transform=ax.transAxes,
                            va='center', ha='center')
                continue

            response = model_response[model_name]['results'][video_name]
            min_len = min(len(response), len(gt))
            time_steps = list(range(1, min_len+1))
            resp = response[:min_len]
            gt_trim = gt[:min_len]

            ax.plot(time_steps, resp, label='response')
            gt_mask = [value == 1 for value in gt_trim]
            ax.fill_between(time_steps, 0.5, 1.0, where=gt_mask, color='red', alpha=0.15, label='GT')
            ax.set_xlim(0, max(1, min_len - 1))
            ax.set_ylim([.5, 1.05])
            hide_top_right_spines(ax)
            if row_idx == len(MODEL_LIST):
                ax.set_xlabel('Frame')
            else:
                ax.tick_params(labelbottom=False)
            if col_idx > 0:
                ax.tick_params(labelleft=False)
            if row_idx == 1 and col_idx == 0:
                ax.set_ylabel('Response')
                ax.set_xlabel('Frame')
            if col_idx == 0:
                ax.text(-0.8, 0.45, display_name, transform=ax.transAxes,
                        va='center', ha='left')

    plt.subplots_adjust(left=0.13, right=0.98, top=0.97, bottom=0.03)
    plt.savefig(os.path.join(os.path.dirname(__file__), f'{file_name_without_ext}.png'), dpi=300)
    plt.show()





if __name__ == '__main__':
    video_list = [
        "looming ball against driving sense\\tc05-green2.mp4",
        "looming square against nature scenarios\\natural-0079_120619-141033_tonemapped-fgr_gray=255.mp4",
        "looming ball against static bg\\ha-dark1.mp4",
        "vehicle against driving sense\\tc09.mp4",
        "pure driving sense\\tc10.mp4",

    ]
    main(video_list)
