import os
import sys
ITEM_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ITEM_ROOT)  

import numpy as np
import cv2
import torch
import time
from tqdm import tqdm
import glob
from prettytable import PrettyTable
from jtop import jtop

from model.existing_lgmd import (
    BasicLGMD, pLGMD, LGMD_P_ON_OFF,
    SFA_LGMD, 
    LGMD1_LGMD2_CascadeNetwork_Dual_Channel,
    LGMD2_LGMD1_CascadeNetwork_Dual_Channel,
    ALGMD, EMD_LPLC2_GF
)
from model.bists_lgmd import BiSTS_LGMD, BiSTS_LGMD_N

CPU_DEVICE = torch.device("cpu")
GPU_DEVICE = torch.device("cuda")


def analyze_LGMD_fps(model, vid_dir, device, jetson):
    model.setup()
    model.to(device) 
    model.eval() 

    # 1. 获取目录下所有的视频文件并排序
    vid_list = []
    search_pattern = os.path.join(vid_dir, '**', '*.mp4')
    vid_list.extend(glob.glob(search_pattern, recursive=True))

    if not vid_list:
        raise FileNotFoundError(f"在路径下未找到视频文件: {vid_dir}")
    
    frame_time_list = []
    powers = []
    frame_count = 0
    
    # 使用 tqdm 监控处理进度
    pbar = tqdm(total=len(vid_list), desc=f"Testing {model.__class__.__name__} on {device.type.upper()}")


    # 2. 遍历视频文件读取帧
    for vid_path in vid_list:
        cap = cv2.VideoCapture(vid_path)
        
        if not cap.isOpened():
            print(f"警告: 无法打开视频 {vid_path}")
            continue

        counter = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break # 当前视频读取完毕，跳出内循环，继续读取下一个视频

            # BGR to RGB, HWC to CHW, Normalize
            gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            torch_img = torch.from_numpy(gray_img).unsqueeze(0).unsqueeze(0).to(device).float() / 255.0

            if counter < 5:
                model(torch_img)  # 预热模型，跳过前5帧的 FLOPs 和时间统计
                counter += 1
                continue # 跳过前10帧的不稳定状态（刚开始有零点突变）
            
            with torch.no_grad():
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                t0 = time.perf_counter()

                model(torch_img)

                if device.type == 'cuda':
                    torch.cuda.synchronize()
                t1 = time.perf_counter()

            frame_time_list.append(t1 - t0)
            frame_count += 1

            current_power = jetson.power['tot']['power'] 
            powers.append(current_power)

        cap.release()
        pbar.update(1)
                
    pbar.close()

    if len(frame_time_list) == 0:
        raise ValueError("视频中有效帧过少，未能成功计算 FLOPs/耗时。")

    avg_time_s = float(np.mean(frame_time_list))
    avg_power = float(np.mean(powers)) / 1000.0  # 瓦特 (W)

    return avg_time_s, avg_power


def systen_power(vid_dir, device, jetson):
    # 1. 获取目录下所有的视频文件并排序
    vid_list = []
    search_pattern = os.path.join(vid_dir, '**', '*.mp4')
    vid_list.extend(glob.glob(search_pattern, recursive=True))

    if not vid_list:
        raise FileNotFoundError(f"在路径下未找到视频文件: {vid_dir}")
    
    powers = []
    frame_count = 0
    
    # 遍历视频文件读取帧
    for vid_path in tqdm(vid_list, total=len(vid_list), desc=f"Measuring System Power on {device.type.upper()}"):
        cap = cv2.VideoCapture(vid_path)
        
        if not cap.isOpened():
            print(f"警告: 无法打开视频 {vid_path}")
            continue

        counter = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break # 当前视频读取完毕，跳出内循环，继续读取下一个视频

            # BGR to RGB, HWC to CHW, Normalize
            gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            torch_img = torch.from_numpy(gray_img).unsqueeze(0).unsqueeze(0).to(device).float() / 255.0
            

            if counter < 5:
                counter += 1
                continue # 跳过前10帧的不稳定状态（刚开始有零点突变）
        
            if device.type == 'cuda':
                torch.cuda.synchronize()
                time.sleep(0.001)  # 模拟系统空载状态，等待功耗稳定
            else:
                time.sleep(0.01)  # 模拟系统空载状态，等待功耗稳定

            current_power = jetson.power['tot']['power'] 
            powers.append(current_power)

            frame_count += 1

        cap.release()

    avg_power = float(np.mean(powers)) / 1000.0  # 瓦特 (W)

    return avg_power


def main():
    video_dir = os.path.join('/home', 'scranes1', 'Data', 'lgmd_benchmark')
    # video_dir = os.path.join('/home', 'scranes1', 'Data', 'lgmd_benchmark', 'basic test')
    
    models_to_test = {
        "LGMD": BasicLGMD(),
        "li-LGMD": LGMD_P_ON_OFF(),
        "pLGMD": pLGMD(),
        "LGMD1-LGMD2": LGMD1_LGMD2_CascadeNetwork_Dual_Channel(),
        "LGMD2-LGMD1": LGMD2_LGMD1_CascadeNetwork_Dual_Channel(),
        "SFA_LGMD": SFA_LGMD(),
        "ALGMD": ALGMD(),
        "EMD-LPLC2-GF": EMD_LPLC2_GF(),
        "BiSTS": BiSTS_LGMD(),
        "BiSTS-N": BiSTS_LGMD_N()
    }

    # ==========================================
    # 渲染学术表格
    # ==========================================
    table = PrettyTable()
    table.field_names = ["Model",  
                         "CPU FPS", "CPU Power", "CPU Pure",
                         "GPU FPS", "GPU Power", "GPU Pure", 
                         ]
    
    # 设置对齐方式
    table.align["Model"] = "l"


    print("开始执行复杂度对比实验 (Complexity Analysis)...")

    with jtop() as jetson:
        # 等待后台数据初始化完毕
        while not jetson.ok():
            time.sleep(0.1)

        cpu_video_dir = os.path.join('/home', 'scranes1', 'Data', 'lgmd_benchmark', 'basic test')  # CPU 只测试基本测试集
        for _ in range(5):
            cpu_system_power = systen_power(cpu_video_dir, CPU_DEVICE, jetson)
        cpu_system_power_str = f"{cpu_system_power:.2f}"
        print(f"系统空载功耗 (CPU System Power): {cpu_system_power_str} W")

        
        gpu_system_power = systen_power(video_dir, GPU_DEVICE, jetson)
        gpu_system_power_str = f"{gpu_system_power:.2f}"
        print(f"系统空载功耗 (GPU System Power): {gpu_system_power_str} W")
        # 新增 table1：仅包含 CPU FPS, CPU FPS/CPU Pure, GPU FPS, GPU FPS/GPU Pure
        table1 = PrettyTable()
        table1.field_names = ["Model", "CPU FPS", "GPU FPS", "CPU FPS/CPU Pure", "GPU FPS/GPU Pure"]
        table1.align["Model"] = "l"

        for model_name, model in models_to_test.items():

            if model_name == "EMD-LPLC2-GF":
                # this model is too slow to test all, so we only test it with a smaller video set
                gpu_video_dir = os.path.join('/home', 'scranes1', 'Data', 'lgmd_benchmark', 'basic test')
            else:
                gpu_video_dir = video_dir

            # CPU 测试
            cpu_vid_dir = os.path.join('/home', 'scranes1', 'Data', 'lgmd_benchmark', 'basic test')
            cpu_avg_time_s, cpu_power = analyze_LGMD_fps(model, cpu_vid_dir, CPU_DEVICE, jetson)
            cpu_fps_val = 1 / cpu_avg_time_s
            cpu_fps_str = f"{cpu_fps_val:.1f}"
            cpu_power_str = f"{cpu_power:.2f}"

            cpu_pure_val = cpu_power - cpu_system_power
            cpu_pure_power_str = f"{cpu_pure_val:.2f}"

            # GPU 测试
            gpu_avg_time_s, gpu_power = analyze_LGMD_fps(model, gpu_video_dir, GPU_DEVICE, jetson)
            gpu_fps_val = 1 / gpu_avg_time_s
            gpu_fps_str = f"{gpu_fps_val:.1f}"
            gpu_power_str = f"{gpu_power:.2f}"

            gpu_pure_val = gpu_power - gpu_system_power
            gpu_pure_power_str = f"{gpu_pure_val:.2f}"

            # 计算 FPS / PureRatio（避免除零）
            if cpu_fps_val is not None and cpu_pure_val is not None and abs(cpu_pure_val) > 1e-9:
                cpu_ratio_str = f"{cpu_fps_val / cpu_pure_val:.1f}"
            else:
                cpu_ratio_str = "N/A"

            if gpu_fps_val is not None and gpu_pure_val is not None and abs(gpu_pure_val) > 1e-9:
                gpu_ratio_str = f"{gpu_fps_val / gpu_pure_val:.1f}"
            else:
                gpu_ratio_str = "N/A"

            table.add_row([
                model_name,
                cpu_fps_str,
                cpu_power_str,
                cpu_pure_power_str,
                gpu_fps_str,
                gpu_power_str,
                gpu_pure_power_str
            ])

            table1.add_row([
                model_name,
                cpu_fps_str,
                gpu_fps_str,
                cpu_ratio_str,
                gpu_ratio_str
            ])

            print(f"完成 {model_name} 的测试: CPU FPS={cpu_fps_str}, GPU FPS={gpu_fps_str}",
                  f"CPU Efficiency={cpu_ratio_str}, GPU Efficiency={gpu_ratio_str}")

    print("\n" + "="*60)
    print("TABLE III: COMPLEXITY ANALYSIS ON HOST AND EDGE DEVICE")
    print("="*60)
    print(table)
    print(f"where CPU System Power: {cpu_system_power_str} W and GPU System Power: {gpu_system_power_str} W")
    print("="*60)
    print(table1)

    with open(os.path.join(os.path.dirname(__file__), 'complexity_analysis_results.txt'), 'w') as f:
        f.write("TABLE III: COMPLEXITY ANALYSIS ON HOST AND EDGE DEVICE\n")
        f.write("="*60 + "\n")
        f.write(str(table) + "\n")
        f.write(f"where CPU System Power: {cpu_system_power_str} W and GPU System Power: {gpu_system_power_str} W\n")
        f.write("="*60 + "\n")
        f.write(str(table1) + "\n")

    



if __name__ == "__main__":
    main()

    