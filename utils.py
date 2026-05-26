''' util.py: utility 一些工具函数 '''

# python自带库
from collections import deque # deque是一个双端队列，可以从队列的两端快速的增加和删除元素
from copy import deepcopy

# 第三方库
import matplotlib.pyplot as plt
import numpy as np
from tkinter import Tk, filedialog
import json


class visualize_LGMD:
    """
        可视化 LGMD 模型输出的类。
    """
    def __init__(self, modelName, maxLen=100, isIOn=True):
        self.isIOn = isIOn
        self.maxLen = maxLen
        self.modelName = modelName
        self.count = 0  # 计数器，用于记录帧数

        # 创建图形和子图
        self.fig, self.ax = plt.subplots(2, 1)

        # 初始化曲线数据
        if self.maxLen == -1:
            self.x_data = deque()  # 存储 x 轴数据(时间或帧数)
            self.y_data = deque()  # 存储 y 轴数据(k 值)
        else:
            self.x_data = deque(maxlen=self.maxLen)  # 存储 x 轴数据(时间或帧数)
            self.y_data = deque(maxlen=self.maxLen)  # 存储 y 轴数据(k 值)
        self.line, = self.ax[1].plot([], [], color='blue')  # 初始化曲线

        # 设置子图属性
        self.ax[0].set_title("Image")
        self.ax[0].axis('off')  # 关闭坐标轴
        self.ax[1].set_title(f"{self.modelName} Output")  # 设置标题
        self.ax[1].set_xlabel("Frame")
        self.ax[1].set_ylabel("Response")
        self.ax[1].set_xlim(0, self.maxLen)  # 设置 x 轴范围
        self.ax[1].set_ylim(0.4, 1.1)  # 设置 y 轴范围

        # 开启交互模式
        if self.isIOn:
            plt.ion()
            plt.show()

    def update(self, img, k, timeCost=None, frameNum=None):
        """
        更新图像和曲线数据。
        """

        # 更新总标题
        if timeCost is not None:
            self.fig.suptitle(f"timeCost: {timeCost*1000: .1f} ms")
        # 更新图像
        self.ax[0].clear()
        self.ax[0].imshow(img, cmap='gray')
        self.ax[0].set_title(f"Frame: {int(frameNum)}" if frameNum is not None else "Image")
        self.ax[0].axis('off')

        # 更新曲线数据
        self.count += 1
        self.x_data.append(self.count)  # x 轴为时间或帧数
        self.y_data.append(k)  # y 轴为 k 值

        # 更新曲线
        self.line.set_data(self.x_data, self.y_data)

        # 调整 x 轴范围以动态显示最新数据
        if self.maxLen is None:
            self.ax[1].set_xlim(0, self.count+20)
        else:
            if self.count > self.maxLen:  # 如果数据点超过 maxLen 个，滑动窗口
                self.ax[1].set_xlim(self.count - self.maxLen, self.count)
            else:
                self.ax[1].set_xlim(0, self.maxLen)

        # 重绘图形
        if self.isIOn:
            plt.pause(0.001)
            plt.draw()


    def close(self):
        """
        关闭图形。
        """
        plt.ioff()  # 关闭交互模式
        plt.close(self.fig)


def get_input_by_GUI(initDir='D:/LGMD_Dataset'):
    ''' 交互窗口选择输入，用鼠标选择输入'''

    root = Tk()
    root.withdraw()  # 隐藏主窗口
    vidPath = filedialog.askopenfilename(title="选择视频文件", 
                                         filetypes=[("All files", "*.*")],
                                         initialdir=initDir,
                                        )
    root.destroy()

    if not vidPath:
        raise ValueError("未选择视频文件")

    return vidPath


## the following functions are only used for evaluation
def evaluate_cbp(modelOpt, groundtruth, threshold=0.7):
    ''' evaluate binary classification problem 
        modelOpt: 模型输出的列表
        groundtruth: 真实标签的列表
        threshold: 阈值，超过该值则认为模型检测到了目标
    '''

    assert len(modelOpt) == len(groundtruth), "模型输出和真实标签长度不一致"

    TP = FP = TN = FN = 0

    for pred, true in zip(modelOpt, groundtruth):
        if pred >= threshold and true == 1:
            TP += 1  # 真阳性
        elif pred >= threshold and true == 0:
            FP += 1  # 假阳性
        elif pred < threshold and true == 0:
            TN += 1  # 真阴性
        elif pred < threshold and true == 1:
            FN += 1  # 假阴性

    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'True Positives': TP,
        'False Positives': FP,
        'True Negatives': TN,
        'False Negatives': FN,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1_score
    }


def evaluate_mean_performance(modelOpt, groundtruth):
    ''' evaluate mean square error for regression problem
        modelOpt: 模型输出的列表
        groundtruth: 真实标签的列表
    '''

    assert len(modelOpt) == len(groundtruth), "模型输出和真实标签长度不一致"

    thresholdList = deepcopy(modelOpt)
    thresholdList = sorted(set(thresholdList), reverse=True)  # 去重并排序，作为阈值列表

    performanceList = {threshold: evaluate_cbp(modelOpt, groundtruth, threshold) for threshold in thresholdList}

    return performanceList


def calculate_ap(recalls, precisions):
    """使用所有点插值法计算AP (COCO标准) """
    # 按recall排序
    recalls = np.array(recalls)
    precisions = np.array(precisions)
    
    sorted_indices = np.argsort(recalls)
    recalls_sorted = recalls[sorted_indices]
    precisions_sorted = precisions[sorted_indices]
    
    # 确保recall从0开始，到1结束
    recalls_interp = np.concatenate(([0.], recalls_sorted, [1.]))
    precisions_interp = np.concatenate(([0.], precisions_sorted, [0.]))
    
    # 对precision进行单调递减处理
    for i in range(len(precisions_interp) - 1, 0, -1):
        precisions_interp[i - 1] = np.maximum(precisions_interp[i - 1], precisions_interp[i])
    
    # 计算AP（PR曲线下面积）
    ap = np.trapezoid(precisions_interp, recalls_interp)

    return ap


def calculate_auc_roc(tp_list, fp_list, tn_list, fn_list):
    """计算 AUC-ROC
    Args:
        tp_list: 真阳性列表
        fp_list: 假阳性列表  
        tn_list: 真阴性列表
        fn_list: 假阴性列表
    Returns:
        auc: AUC-ROC 值
    """
    # 转换为 numpy 数组以提高性能
    tp_array = np.array(tp_list)
    fp_array = np.array(fp_list) 
    tn_array = np.array(tn_list)
    fn_array = np.array(fn_list)
    
    # 计算 TPR 和 FPR（避免除零错误）
    denominator_tpr = tp_array + fn_array
    denominator_fpr = fp_array + tn_array
    
    tpr = np.divide(tp_array, denominator_tpr, out=np.zeros_like(tp_array, dtype=float), 
                   where=denominator_tpr > 0)
    fpr = np.divide(fp_array, denominator_fpr, out=np.zeros_like(fp_array, dtype=float),
                   where=denominator_fpr > 0)
    
    # 按 FPR 排序并确保单调性
    sort_idx = np.argsort(fpr)
    fpr_sorted = fpr[sort_idx]
    tpr_sorted = tpr[sort_idx]
    
    # 确保从 (0,0) 开始，到 (1,1) 结束
    fpr_final = np.concatenate(([0.0], fpr_sorted, [1.0]))
    tpr_final = np.concatenate(([0.0], tpr_sorted, [1.0]))
    
    # 计算 AUC
    auc = np.trapezoid(tpr_final, fpr_final)
    
    return auc


def calculate_auc_threshold_precision(thresholds, precisions):
    """计算 AUC of Threshold-Precision
    Args:
        thresholds: 阈值列表
        Precision: Precision列表
    Returns:
        auc: AUC of Threshold-Precision 值
    """
    # 转换为 numpy 数组以提高性能
    thresholds_array = np.array(thresholds)
    precisions_array = np.array(precisions)
    
    # 按阈值排序
    sort_idx = np.argsort(thresholds_array)
    thresholds_sorted = thresholds_array[sort_idx]
    precisions_sorted = precisions_array[sort_idx]
    
    # 确保从 (min_threshold,0) 开始，到 (max_threshold,0) 结束
    thresholds_final = np.concatenate(([0.5], thresholds_sorted, [1.0]))
    precisions_final = np.concatenate(([0.0], precisions_sorted, [0.0]))
    
    # 计算 AUC
    i = np.where(thresholds_final[1:] != thresholds_final[:-1])[0]
    auc = np.sum((thresholds_final[i + 1] - thresholds_final[i]) * precisions_final[i + 1])
    
    if thresholds_sorted[-1] == thresholds_sorted[0]:
        return 0.0  # 避免除以零
    else:
        return auc / (thresholds_final[-1] - thresholds_final[0]) # normalize to [0, 1] 


def calculate_IoU_for_PSP(modelOpt, groundtruth):
    """
    Calculate Intersection over Union (IoU) for probabilistic score prediction (PSP).
    Args:
        modelOpt: List or array of model outputs (probabilistic scores), ranging from 0.5 to 1
        groundtruth: List or array of ground truth values (probabilistic scores), ranging from 0.5 to 1
    Returns:
        iou: IoU value in the range [0, 1]
    """
    
    # Convert to numpy arrays
    modelOpt = np.asarray(modelOpt)
    groundtruth = np.asarray(groundtruth)

    min_gt = np.min(groundtruth)
    len_gt = len(groundtruth)

    # Calculate intersection and union
    intersection = np.trapezoid(np.minimum(modelOpt, groundtruth) - min_gt) / len_gt * 2 
    union = np.trapezoid(np.maximum(modelOpt, groundtruth) - min_gt) / len_gt * 2 

    if np.max(groundtruth) == 0.5:
        return 1 - union
    else:
        # Prevent division by zero
        if union == 0:
            return 0.0
        else:
            return intersection / union
        

def custom_serialize(obj, indent=2, current_level=0):
    if isinstance(obj, list):
        current_level += 1
        if any(isinstance(item, (list, dict)) for item in obj):
            # 包含嵌套结构，不是最后一级
            indent_str = ' ' * (indent * current_level)
            items = [f"{indent_str}{custom_serialize(item, indent, current_level)}" for item in obj]
            outer_indent = ' ' * (indent * (current_level - 1)) if current_level > 1 else ''
            return '[\n' + ',\n'.join(items) + '\n' + outer_indent + ']'
        else:
            # 最后一级，都是基本类型
            return '[' + ', '.join(json.dumps(item, ensure_ascii=False) for item in obj) + ']'
    
    elif isinstance(obj, dict):
        current_level += 1
        indent_str = ' ' * (indent * current_level)
        items = []
        for key, value in obj.items():
            serialized_value = custom_serialize(value, indent, current_level)
            items.append(f'{indent_str}{json.dumps(key, ensure_ascii=False)}: {serialized_value}')
        
        outer_indent = ' ' * (indent * (current_level - 1)) if current_level > 1 else ''
        return '{\n' + ',\n'.join(items) + '\n' + outer_indent + '}'
    
    else:
        return json.dumps(obj, ensure_ascii=False)
    