import os
import sys
CUR_DIR = os.path.dirname(__file__) # 当前文件所在目录
sys.path.append(os.path.dirname(CUR_DIR)) # 添加上级目录到路径中


import cv2
from copy import deepcopy
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm
import torch
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

from model import bists_lgmd

    
def main(vidDict, savePDFPth=None):
    '''
    主函数，读取视频并处理
    Args:
        modelDict: 模型字典，包含模型名称和参数
        vidPth: 视频路径
    '''
    with PdfPages(savePDFPth) as pdf: 

        court = 0
        fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(11, 8)) 

        for vidName, vidPth in tqdm(vidDict.items(), total=len(vidDict), desc="Processing Videos"):

            objVid = cv2.VideoCapture(vidPth)

            responses = [None for _ in range(int(objVid.get(cv2.CAP_PROP_FRAME_COUNT)))]
            # 重新创建模型对象
            model = bists_lgmd.BiSTS_LGMD()
            model.to(DEVICE)

            # 读取视频
            while objVid.isOpened():
                ret, colorImg = objVid.read()
                _idx = int(objVid.get(cv2.CAP_PROP_POS_FRAMES))-1
                if not ret:
                    break
                if _idx == 0:
                    firstImg = deepcopy(colorImg)

                gray = cv2.cvtColor(colorImg, cv2.COLOR_BGR2GRAY)
                gray_tensor = torch.from_numpy(gray).float().unsqueeze(0).unsqueeze(0).to(DEVICE)

                k = model.forward(gray_tensor)

                # 获取输出
                responses[_idx] = k.item()           

            objVid.release() 


            # 可视化输出
            if court < 3:
                id_X_1 = 0
                id_X_2 = 1
                id_Y = court
            else:
                id_X_1 = 3
                id_X_2 = 4
                id_Y = court % 3

            for i in range(3):
                axes[2, i].axis('off')

            axes[id_X_1, id_Y].imshow(firstImg)
            axes[id_X_1, id_Y].set_title(f"{vidName}", fontsize=7)
            axes[id_X_1, id_Y].axis('off')
            axes[id_X_2, id_Y].plot(responses)
            axes[id_X_2, id_Y].set_xlabel("Frame", fontsize=6)
            axes[id_X_2, id_Y].set_ylabel("Response", fontsize=6)
            axes[id_X_2, id_Y].tick_params(axis='both', labelsize=6)
            axes[id_X_2, id_Y].set_ylim(0.4, 1.1)
            
            court += 1
                
            if court == 6:
            # 保存图像到PDF
                pdf.savefig()
                plt.close(fig)
                fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(11, 8)) 
                court = 0

        # 保存最后一页
        if court < 6:   
            pdf.savefig()
            plt.close(fig)


def select_basic_video(): 
    # 视频路径字典
    Benchmark_PTH = os.path.join(os.path.dirname(os.path.dirname(CUR_DIR)), '7_Dataset', 'lgmd_benchmark')
    vidDict = {}

    for root, dirs, files in os.walk(Benchmark_PTH):
        for fileName in files:
            if fileName.endswith('.mp4'):
                # 核心修正：
                # 键应该是 fileName (字符串)，值应该是完整的绝对路径
                full_path = os.path.join(root, fileName)
                vidDict[fileName] = full_path
                
    return vidDict
    

if __name__ == '__main__':
    
    vidDict = select_basic_video()
    
    # 保存PDF路径
    PDFPth = os.path.join(CUR_DIR, 'basic-test for LCR_STMD.pdf')
    open(PDFPth, 'wb').close()

    # 执行主函数
    main(vidDict, savePDFPth=PDFPth)


