import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import time

N = 8

def monte_carlo_simulation_batched(d=0.5, B=0.3, num_trials=1e10, chunk_size=10_000_000):
    """
    分批处理的蒙特卡洛模拟，防止内存溢出。
    
    参数:
    num_trials: 总模拟次数 (支持百亿级别)
    chunk_size: 每一批处理的样本量 (建议 1千万 到 5千万 之间)
    """
    num_trials = int(num_trials)
    print(f"--- 开始超级模拟: d={d}, B={B}, 总次数={num_trials:,} ---")
    print(f"--- 采用分批计算，每批 {chunk_size:,} 次，保护内存 ---")
    
    start_time = time.time()
    
    # 1. 初始化变量
    values = np.array([0, -1, 1, -B, B, 1-B, B-1])
    probs = np.array([
        1 - 2*d + 1.5*(d**2), (d**2)/4, (d**2)/4, 
        d/2 - (d**2)/2, d/2 - (d**2)/2, d/2 - (d**2)/2, d/2 - (d**2)/2
    ])
    probs = probs / np.sum(probs)
    
    # 理论值计算
    sigma2 = (d**2)/2 + d*(1-d) * (B**2 + (1-B)**2)
    var_S8 = N/2*5 * sigma2
    theoretical_abs_mean = np.sqrt(2 / np.pi * var_S8)
    
    # 2. 准备分批累加器
    total_abs_sum = 0.0
    trials_completed = 0
    
    # 为了画图，我们需要累积直方图的计数，而不是保存所有点
    # 理论最大值为 8 * 1 = 8，设定 bins 范围为 -10 到 10
    bins = np.linspace(-10, 10, 100) 
    hist_counts = np.zeros(len(bins) - 1)
    
    # 3. 分批执行循环
    while trials_completed < num_trials:
        # 确定这一批次要跑多少个（处理最后可能不足一个 chunk_size 的尾巴）
        current_chunk = min(chunk_size, num_trials - trials_completed)
        
        # 生成当前批次
        samples = np.random.choice(values, size=(current_chunk, N), p=probs)
        kernel = np.array([1,] * int(N/2) +  [2,] * int(N/2) )
        S8_chunk = np.sum(samples*kernel, axis=1)
        
        # 累加绝对值之和
        total_abs_sum += np.sum(np.abs(S8_chunk))
        
        # 累加直方图频数 (这一步非常巧妙，用极小的内存保留了分布形状)
        counts, _ = np.histogram(S8_chunk, bins=bins)
        hist_counts += counts
        
        trials_completed += current_chunk
        
        # 打印进度条
        if trials_completed % (chunk_size * 10) == 0 or trials_completed == num_trials:
            progress = (trials_completed / num_trials) * 100
            elapsed = time.time() - start_time
            print(f"进度: {progress:5.1f}% | 已完成 {trials_completed:,} 次 | 耗时: {elapsed:.1f} 秒")

    # 4. 计算最终平均值
    simulated_abs_mean = total_abs_sum / num_trials
    
    # 5. 打印结果对比
    print("\n" + "="*40)
    print(f"理论绝对值期望 (正态近似): {theoretical_abs_mean:.8f}")
    print(f"模拟绝对值期望 (100亿次):   {simulated_abs_mean:.8f}")
    print(f"相对误差: {abs(theoretical_abs_mean - simulated_abs_mean) / theoretical_abs_mean * 100:.6f}%")
    print("="*40 + "\n")
    
    # 6. 可视化 (使用累加好的频数绘制)
    plt.figure(figsize=(10, 6))
    
    # 将计数值转换为概率密度 (Density)
    bin_widths = np.diff(bins)
    hist_density = hist_counts / (num_trials * bin_widths)
    
    # 使用 bar 绘制我们自己计算的概率密度直方图
    plt.bar(bins[:-1], hist_density, width=bin_widths, alpha=0.6, color='skyblue', edgecolor='black', align='edge', label='Simulated $S_8$ Distribution')
    
    # 绘制正态曲线
    x = np.linspace(-10, 10, 1000)
    y = norm.pdf(x, loc=0, scale=np.sqrt(var_S8)) 
    plt.plot(x, y, 'r-', lw=2, label=f'Theoretical Normal $\\mathcal{{N}}(0, {var_S8:.2f})$')
    
    plt.title(f'Batched Monte Carlo: $S_{N}$ \n d={d}, B={B}, Trials={num_trials:.1e}')
    plt.xlabel('Value of $S_8$')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# 运行防崩版模拟 (测试 1 亿次很快，100 亿次需要耐心等一会儿)
# 这里先默认跑 1亿次 让你测试，想跑 100亿次 直接改回 1e10
monte_carlo_simulation_batched(d=0.5, B=0.3, num_trials=1e8)