import os
import sys
CUR_DIR = os.path.dirname(__file__) # 当前文件所在目录
sys.path.append(os.path.dirname(CUR_DIR)) # 添加上级目录到路径中
import time

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch

from utils import visualize_LGMD, get_input_by_GUI
from model.existing_lgmd import pLGMD

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == '__main__':
    # 读取视频
    vidPath = os.path.join(os.path.dirname(os.path.dirname(CUR_DIR)), 
                                                    '7_Dataset', 'lgmd_benchmark', 'basic test', 'looming square',
                                                    'dark_looming.mp4')
    # vidPath = get_input_by_GUI(initDir=os.path.join(os.path.dirname(os.path.dirname(CUR_DIR)), 
                                                    # '7_Dataset', 'color balls approaching'),)

    objVid = cv2.VideoCapture(vidPath)

    model = pLGMD()
    model.to(DEVICE)

    # 创建可视化对象
    visualizer = visualize_LGMD()

    while objVid.isOpened():
        ret, colorImg = objVid.read()
        if not ret:
            break
        

        gray = cv2.cvtColor(colorImg, cv2.COLOR_BGR2GRAY).astype(np.float32) # LGMD 的输入不除255.0
        gray_tensor = torch.from_numpy(gray).unsqueeze(0).unsqueeze(0).to(DEVICE)  # 转换为4D张量并移动到GPU
        colorImg = cv2.cvtColor(colorImg, cv2.COLOR_BGR2RGB)
        if DEVICE == 'cuda':
            torch.cuda.synchronize()  # 确保GPU上的所有操作完成
        start_time = time.perf_counter()
        k = model.forward(gray_tensor)
        if DEVICE == 'cuda':
            torch.cuda.synchronize()  # 
        timeCost = time.perf_counter() - start_time  # ms

        visualizer.update(colorImg, k.item(), timeCost, objVid.get(cv2.CAP_PROP_POS_FRAMES))

    # 等待用户输入
    print("Press any key to close the window...")
    plt.waitforbuttonpress()  # 等待用户按下键盘或鼠标按键

    # 关闭图形
    visualizer.close()
    objVid.release()
