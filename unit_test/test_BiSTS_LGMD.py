import os
import sys
CUR_DIR = os.path.dirname(__file__) # 当前文件所在目录
sys.path.append(os.path.dirname(CUR_DIR)) # 添加上级目录到路径中

import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import torch

from utils import visualize_LGMD, get_input_by_GUI
from model.bists_lgmd import (BiSTS_LGMD, BiSTS_LGMD_N,
                              BiSTS_LGMD_with_Original_EI, # Original E-I layer without pooling
                              BiSTS_LGMD_with_Randomly_Dropout, # Randomly Dropout mechanism for STSA module - from pLGMD
                              BiSTS_LGMD_Remove_LTSC, # remove LTSC
                              BiSTS_LGMD_Remove_LP_Opt, # remove Layer pooling opt
                              BiSTS_LGMD_Adding_G_before_LP, # Adding G before Layer Pooling
                              BiSTS_LGMD_Adding_CN_before_LP, # Adding G before CN
                              BiSTS_LGMD_Adding_CN_in_PP, # Adding CN in PP (pre-processing)
                              BiSTS_LGMD_Adding_CN_in_PP_and_Adding_G_before_LP, # Adding CN in PP and Adding G
                              BiSTS_LGMD_Adding_G_CN_before_LP, # Adding G and CN before LP
                              BiSTS_LGMD_Adding_CN_G_before_LP, # Adding CN and G (contrast normalization)
                              BiSTS_LGMD_with_STSA_remove_Conv,
                              BiSTS_LGMD_Pooling_before_Conv
                            ) # type: ignore

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


if __name__ == '__main__':
    # 读取视频
    # vidPath = os.path.join(os.path.dirname(os.path.dirname(CUR_DIR)), 
    # vidPath = get_input_by_GUI(initDir=os.path.join(os.path.dirname(os.path.dirname(CUR_DIR)), 
    #                                                 '7_Dataset', '100x100 synthetic stimuli',
    #                                                 ))
    vidPath = os.path.join(os.path.dirname(os.path.dirname(CUR_DIR)), 
                                                    '7_Dataset', 'lgmd_benchmark', 'basic test', 'looming square',
                                                    'dark_looming.mp4')
    # vidPath = os.path.join(os.path.dirname(os.path.dirname(CUR_DIR)), 
    #                                             '7_Dataset', 'lgmd_benchmark', 'looming ball against driving sense',
    #                                             'c41-dark4.mp4')

    objVid = cv2.VideoCapture(vidPath)

    # model = BiSTS_LGMD()
    # model = BiSTS_LGMD_N()
    # model = BiSTS_LGMD_with_Original_EI()
    # model = BiSTS_LGMD_with_Randomly_Dropout()
    # model = BiSTS_LGMD_Remove_LTSC()
    # model = BiSTS_LGMD_Remove_LP_Opt()
    model = BiSTS_LGMD_Adding_G_before_LP()
    # model = BiSTS_LGMD_Adding_CN_before_LP()
    # model = BiSTS_LGMD_Adding_CN_in_PP()
    # model = BiSTS_LGMD_Adding_CN_in_PP_and_Adding_G_before_LP()
    # model = BiSTS_LGMD_Adding_G_CN_before_LP()
    # model = BiSTS_LGMD_Adding_CN_G_before_LP()
    # model = BiSTS_LGMD_with_STSA_remove_Conv()
    # model = BiSTS_LGMD_Pooling_before_Conv()
    model.to(DEVICE)

    # 创建可视化对象
    visualizer = visualize_LGMD()

    total_time = 0.0
    while objVid.isOpened():
        ret, colorImg = objVid.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(colorImg, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gray_tensor = torch.from_numpy(gray).unsqueeze(0).unsqueeze(0).to(DEVICE)  # 转换为4D张量 (1, 1, H, W)
        colorImg = cv2.cvtColor(colorImg, cv2.COLOR_BGR2RGB)
        if DEVICE == 'cuda':
            torch.cuda.synchronize()  # 确保GPU计算完成
        start_time = time.perf_counter()
        k = model.forward(gray_tensor)
        if DEVICE == 'cuda':
            torch.cuda.synchronize()  # 确保GPU计算完成
        timeCost = time.perf_counter() - start_time  # ms
        total_time += timeCost

        # if k > 0.6:
        #     print('\n=================!!! WARNING !!!=================')
        # print('Frame:%3d, opt: %.2f; inv: %.1f, seq: %.1f; \n\ttemEig: %s, \n\tspaEig: %s, \n\tspatemEig: %s' % (
        #     int(objVid.get(cv2.CAP_PROP_POS_FRAMES)), 
        #     k,
        #     model.smrom.invNumOfTem, 
        #     model.smrom.seqNumOfSTem, 
        #     str(model.smrom.temEig),
        #     str(model.smrom.spaEig),
        #     str(model.smrom.spatemEig) ))
        # showEig = model.smrom.temEig
        
        
        # print('Frame:%3d, inv: %.2f, seq: %.2f;\t tem0: %.2f, \t gain_coeff: %.1f, \t k: %.2f' % (
        #     int(objVid.get(cv2.CAP_PROP_POS_FRAMES)), 
        #     model.smrom.invNumOfTem, 
        #     model.smrom.seqNumOfSTem,
        #     model.smrom.temEig[0], 
        #     model.gain_coeff,
        #     k))
            

        visualizer.update(colorImg, k.item(), timeCost, objVid.get(cv2.CAP_PROP_POS_FRAMES))
        # 模拟实时数据间隔
        time.sleep(0.001)


    # 等待用户输入
    print(f'total time: {total_time:.4f} s')
    print("Press any key to close the window...")
    plt.waitforbuttonpress()  # 等待用户按下键盘或鼠标按键

    # 关闭图形
    visualizer.close()
    objVid.release()
