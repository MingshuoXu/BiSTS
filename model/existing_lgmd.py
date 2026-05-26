#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from math import exp
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.transforms.functional import gaussian_blur


class BasicLGMD(nn.Module):
    '''
    Ref: 
        - Yue S, Rind F C. A collision detection system for a mobile robot inspired by the locust visual system[C]
            //Proceedings of the 2005 IEEE international conference on robotics and automation. IEEE, 2005: 3832-3837.
        - Yue S, Rind F C. Collision detection in complex dynamic scenes using an LGMD-based visual neural network 
            with feature enhancement[J]. IEEE transactions on neural networks, 2006, 17(3): 705-716.
    '''
    def __init__(self):
        super().__init__()

        self.p1 = 0
        self.p2 = 0
        self.alpha = 1
        self.Cde = 0.5
        self.Tde = 15
        w_tensor = torch.tensor([[0.125, 0.25, 0.125], 
                           [0.25, 0, 0.25],
                           [0.125, 0.25, 0.125]], dtype=torch.float32).unsqueeze(0).unsqueeze(0) # shape (1, 1, 3, 3)
        self.register_buffer('w', w_tensor)
        We_tensor = 1/9 * torch.ones((1, 1, 3, 3), dtype=torch.float32) # shape (1, 1, 3, 3)
        self.register_buffer('We', We_tensor)
        self.WI = 0.3
        self.Cw = 4
        self.deltaC = 0.01

        self.setup() # 初始化
    
    def setup(self):
        self.last_ipt = None
        self.last_P_opt = None
        self.last_last_P_opt = None

    def model_structure(self, ipt_tensor):
        ''' 模型的结构，清晰地定义了模型的处理流程
            输入: 灰度图像
            输出: K值
        '''
        
        ''' P Layer'''
        self.P_opt = ipt_tensor - self.last_ipt + \
            self.p1 * self.last_P_opt + self.p2 * self.last_last_P_opt

        ''' IE Layer'''
        self.exciOpt = self.P_opt.clone()
        self.inhiOpt = F.conv2d(self.last_P_opt, self.w, padding=1)

        ''' S Layer'''
        self.S_opt = torch.abs(self.exciOpt) - self.WI * torch.abs(self.inhiOpt)

        ''' G Layer'''
        self.Ce = F.avg_pool2d(self.S_opt, kernel_size=3, stride=1, padding=1) # F.conv2d(self.S_opt, self.We, padding=1) 这里的卷积核是全1的，等价于求局部平均
        omega = torch.max(torch.abs(self.Ce)) / self.Cw + self.deltaC

        self.G = self.S_opt * self.Ce / omega
        # equivalent to `self.G[self.G * self.Cde < self.Tde] = 0`
        self.G = torch.where(self.G * self.Cde < self.Tde, torch.zeros_like(self.G), self.G) 

        ''' LGMD Cell '''
        self.k = torch.mean(torch.abs(self.G))

        # membrane potential, that is K
        # / (self.imgH * self.imgW) 这个归一化放上面去了
        self.mp = 1 / ( 1 + torch.exp(-self.alpha * self.k)) 

        return self.mp
    
    def forward(self, ipt_tensor):
        ''' forward函数, 本质是在model_structure的基础上, 增加对运行时间以及对第一第二帧的特殊初始化，以及中间变量的保存
            
            输入: ipt_tensor, shape (1, 1, H, W), dtype torch.float32, device 'cpu' or 'cuda'
            输出: 
            - opt: 输出膜电位
            - timeCost: 模型运行时间
        '''

        if self.last_ipt is None:
            # 第一帧，获得图像尺寸
            self.imgH, self.imgW = ipt_tensor.shape[-2:] # 图像高宽
            opt = ipt_tensor.new_tensor([0.5])
        elif self.last_P_opt is None:
            # 第二帧, 只计算与上一帧的帧差
            self.last_P_opt = ipt_tensor - self.last_ipt
            self.last_last_P_opt = torch.zeros_like(ipt_tensor)
            opt = ipt_tensor.new_tensor([0.5])
        else:
            opt = self.model_structure(ipt_tensor)
            ''' 存储上一次的值'''
            self.last_last_P_opt = self.last_P_opt.detach().clone()
            self.last_P_opt = self.P_opt.detach().clone()
        
        self.last_ipt = ipt_tensor.detach().clone()
        return opt
 

class pLGMD(BasicLGMD):
    '''
    Ref:
        - Hong J, Fu Q, Sun X, et al. Boosting collision perception against noisy signals with a probabilistic 
            neural network[C]//2023 International Joint Conference on Neural Networks (IJCNN). IEEE, 2023: 1-8.
        - Hong J, Sun X, Peng J, et al. A bio-inspired probabilistic neural network model for noise-resistant 
            collision perception[J]. Biomimetics, 2024, 9(3): 136.
        - https://github.com/fuqinbing/pLGMD
    '''
    def __init__(self):
        super().__init__()
        self.p1 = 1/(1+exp(1))
        self.p2 = 1/(1+exp(2))
        self.Ts = 30

        self.probP = self.prob = self.probE = self.probI = self.probG = 0.8
        self.register_buffer('pMtxP2I', self.prob * torch.ones((1, 1, 3, 3), dtype=torch.float32))
        self.pMtxP2I[:, :, 1, 1] = 0

    def setup(self):
        super().setup()

    @staticmethod
    def _bernoulli_mask_like(tensor, prob):
        return torch.bernoulli(torch.full_like(tensor, prob))

    @staticmethod
    def prob_conv2(A, B, Prob):
        """ cconv2_with_probability in the raw version
        带概率的二维卷积函数 (same模式)
        
        参数:
        A: 输入矩阵 [1, 1, ma, na]
        B: 卷积核 [1, 1, mb, nb]，尺寸必须为奇数
        Prob: 概率矩阵，尺寸与卷积核相同
        
        返回:
        C: 卷积结果，尺寸与输入相同
        """

        # 检查输入尺寸
        ma, na = A.shape[-2:]
        mb, nb = B.shape[-2:]
        prob_m, prob_n = Prob.shape[-2:]
        
        if prob_m != mb or prob_n != nb:
            raise ValueError('卷积核的尺寸和概率参数尺寸不一致！')
        
        if mb % 2 == 0 or nb % 2 == 0:
            raise ValueError('卷积核的尺寸必须为奇数')
        
        # 初始化输出
        C = torch.zeros_like(A)

        # 扩展输入矩阵
        pad_h = (mb - 1) // 2
        pad_w = (nb - 1) // 2
        expansion_A = F.pad(A, (pad_w, pad_w, pad_h, pad_h))
        
        for i in range(mb):
            for j in range(nb):
                b_val = B[:, :, mb - 1 - i, nb - 1 - j]
                prob_val = float(Prob[:, :, mb - 1 - i, nb - 1 - j].reshape(-1)[0])

                # 提取对应的滑动窗口
                sliceA = expansion_A[:, :, i:i + ma, j:j + na]

                # 生成概率矩阵
                prob_matrix = torch.bernoulli(torch.full_like(sliceA, prob_val))

                # 累加结果
                C += b_val.view(1, 1, 1, 1) * sliceA * prob_matrix
        
        return C
    
    def model_structure(self, ipt):
        ''' 模型的结构，清晰地定义了模型的处理流程
            输入: 灰度图像
            输出: K值
        '''
        
        ''' Temporal Difference'''
        self.P_opt = ipt - self.last_ipt + \
            self.p1 * self.last_P_opt + self.p2 * self.last_last_P_opt

        ''' Lateral Inhibition'''
        pMtxP2E = self._bernoulli_mask_like(self.P_opt, self.probP)
        self.exciIpt = self.P_opt * pMtxP2E

        
        self.inhiIpt = self.prob_conv2(self.last_P_opt, self.w, self.pMtxP2I)

        pMtxE2S = self._bernoulli_mask_like(self.exciIpt, self.probE)
        self.exciOpt = self.exciIpt * pMtxE2S

        pMtxI2S = self._bernoulli_mask_like(self.inhiIpt, self.probI)
        self.inhiOpt = self.inhiIpt * pMtxI2S 

        ''' Spatial Summation'''
        self.S_opt = self.exciOpt - self.inhiOpt * self.WI

        ''' Grouping Mechanism'''
        self.Ce = F.avg_pool2d(self.S_opt, kernel_size=3, stride=1, padding=1)
        omega = torch.max(torch.abs(self.Ce)) / self.Cw + self.deltaC

        G = self.S_opt * self.Ce / omega
        # G[G < self.Ts] = 0
        G = torch.where(G < self.Ts, torch.zeros_like(G), G)


        pMtxG = self._bernoulli_mask_like(G, self.probG)
        self.G = G * pMtxG

        self.k = torch.mean(torch.abs(self.G))

        # membrane potential, that is K
        self.mp = 1 / ( 1 + torch.exp(- self.alpha * self.k) )

        return self.mp


class LGMD_P_ON_OFF(nn.Module):
    """
    path                   : path of the folder where the dataset is located
    beta_1                 : weight coefficient of inhibition unit
    I_size                 : size of convolution kernel
    alpha_1                : baseline sensitivity value of contrast normalization
    alpha_2,alpha_3        : influence coefficient of contrast and motion computation in ON channel
    alpha_4,alpha_5        : influence coefficient of contrast and motion computation in OFF channel
    theta_1,theta_2,theta_3: influence coefficient of ON/OFF channel
    gauss_size,gauss_div   : size and standard deviation of gaussian convolution kernel
    C_w                    : a constant used to calculate w
    delta_C                : a small real number used to prevent calculation errors
    C_de,T_de              : a constant used to calculate T_g
    T_g                    : threshold of grouping mechanism
    K_I,K_C,K_g            : Three convolution kernels in computation

    Ref:
        - Li Z, Fu Q, Li H, et al. Dynamic signal suppression increases the fidelity of looming perception 
            against input variability[C]//2022 International Joint Conference on Neural Networks (IJCNN). 
            IEEE, 2022: 1-8.
        - Fu Q, Li Z, Peng J. Harmonizing motion and contrast vision for robust looming detection[J]. 
            Array, 2023, 17: 100272.
        - https://github.com/fuqinbing/harmonizing-motion-and-contrast-vision-for-looming-detection
    """

    def __init__(self):
        super().__init__()
        self.last_gray_image = None
        self.beta = 0.4
        self.I_size = 3
        self.alpha_1 = 3
        self.alpha_2 = 1
        self.alpha_3 = 1
        self.alpha_4 = 1
        self.alpha_5 = 1
        self.theta_1 = 1
        self.theta_2 = 1
        self.theta_3 = 1
        self.gauss_size = 9
        self.gauss_div = 5
        self.C_w = 4
        self.delta_C = 0.01
        self.C_de = 0.5
        self.T_de = 0.7
        self.basic_weight = 0.125
        self.register_buffer('K_I', torch.zeros([1, 1, self.I_size, self.I_size], dtype=torch.float32))
        self.register_buffer('K_C', torch.zeros([1, 1, 3, 3], dtype=torch.float32))
        self.register_buffer('K_g', torch.zeros([1, 1, 3, 3], dtype=torch.float32))
        self.register_buffer('gauss_kernel', self._build_gaussian_kernel(self.gauss_size, self.gauss_div))
        self.init_kernel()
        self.setup()

    def init_kernel(self):
        for i in range(3):
            for j in range(3):
                self.K_g[0, 0, i, j] = 1 / 9
                self.K_C[0, 0, i, j] = -1 / 8
                if i == 1 and j == 1:
                    self.K_C[0, 0, i, j] = 1

        a = 0
        b = 0
        while a < self.I_size / 2:
            b = 0
            while b < self.I_size / 2:
                weight = self.basic_weight * (((a * a + b * b) ** 0.5) + 1)
                self.K_I[0, 0, a, b] = weight
                self.K_I[0, 0, self.I_size - 1 - a, b] = weight
                self.K_I[0, 0, a, self.I_size - 1 - b] = weight
                self.K_I[0, 0, self.I_size - 1 - a, self.I_size - 1 - b] = weight
                b += 1
            a += 1
        self.K_I[0, 0, a - 1, b - 1] = 0

    def setup(self):
        self.last_gray_image = None
        self.last_E_on = None
        self.last_E_off = None
        self.height = None
        self.width = None
        self.mp = torch.tensor(0.5, dtype=torch.float32)

    @staticmethod
    def _build_gaussian_kernel(kernel_size, sigma):
        coords = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2
        grid_y, grid_x = torch.meshgrid(coords, coords, indexing='ij')
        kernel = torch.exp(-(grid_x ** 2 + grid_y ** 2) / (2 * sigma ** 2))
        kernel = kernel / kernel.sum()
        return kernel.unsqueeze(0).unsqueeze(0)

    def looming_perception(self, frame_diff):
        self.P = frame_diff
        self.P_on = torch.clamp(self.P, min=0)
        self.P_off = -torch.clamp(self.P, max=0)

        self.M_hat_on = F.conv2d(self.P_on, self.gauss_kernel, padding=self.gauss_size // 2)
        self.M_hat_off = F.conv2d(self.P_off, self.gauss_kernel, padding=self.gauss_size // 2)
        self.M_on = torch.tanh(self.P_on / (self.M_hat_on + self.alpha_1))
        self.M_off = torch.tanh(self.P_off / (self.M_hat_off + self.alpha_1))

        self.C_on = torch.abs(F.conv2d(self.M_on, self.K_C, padding=1))
        self.C_off = torch.abs(F.conv2d(self.M_off, self.K_C, padding=1))

        self.E_on = self.M_on
        self.I_on = F.conv2d(self.last_E_on, self.K_I, padding=1)
        self.S_on = self.E_on - self.beta * self.I_on

        self.E_off = self.M_off
        self.I_off = F.conv2d(self.last_E_off, self.K_I, padding=1)
        self.S_off = self.E_off - self.beta * self.I_off

        self.S_on_out = torch.clamp(self.alpha_2 * self.S_on - self.alpha_3 * self.C_on, min=0)
        self.S_off_out = -torch.clamp(self.alpha_4 * self.S_off - self.alpha_5 * self.C_off, max=0)

        self.S = self.theta_1 * self.S_on_out + self.theta_2 * self.S_off_out + self.theta_3 * self.S_on_out * self.S_off_out
        self.Ce = F.conv2d(self.S, self.K_g, padding=1)
        w = torch.max(torch.abs(self.Ce)) / self.C_w + self.delta_C
        self.G = self.S * self.Ce / w
        self.G[self.G * self.C_de < self.T_de] = 0
        k = torch.mean(self.G)
        self.mp = 1 / (1 + torch.exp(-k * 10))

        self.last_E_on = self.E_on.detach().clone()
        self.last_E_off = self.E_off.detach().clone()
        return self.mp

    def forward(self, gray_image):

        if self.last_gray_image is not None:
            frame_diff = gray_image - self.last_gray_image
            self.mp = self.looming_perception(frame_diff)
        else:
            self.height, self.width = gray_image.shape[-2:]
            self.last_E_on = torch.zeros_like(gray_image)
            self.last_E_off = torch.zeros_like(gray_image)
            self.mp = gray_image.new_tensor(0.5)

        self.last_gray_image = gray_image.detach().clone()
        return self.mp


class SFA_LGMD(BasicLGMD):
    '''
    LGMD1 with ON-OFF pathways and spike frequency adaptation.
    
    Ref:
        - Fu Q, Hu C, Peng J, et al. Shaping the collision selectivity in a looming sensitive neuron model with 
            parallel on and off pathways and spike frequency adaptation[J]. Neural Networks, 2018, 106: 127-143.
    '''
    
    def __init__(self, fps=30):
        super().__init__()
        
        # Temporal parameters
        tau_in = 1000 / fps
        tau_1tmp = torch.tensor([0, 30, 60], dtype=torch.float32)
        tau_inm = tau_in * torch.ones(3, dtype=torch.float32)
        self.alp = tau_inm / (tau_inm + tau_1tmp)  # shape (3,)
        
        self.tau_f = 1/tau_in # tau_in / (tau_in + 10)
        self.sig_slow = 700 / (700 + tau_in)
        self.sig_fast = 300 / (300 + tau_in)
        
        # Spatial parameters
        W_on_tensor = torch.tensor([[0.125, 0.25, 0.125], 
                                    [0.25, 0, 0.25],
                                    [0.125, 0.25, 0.125]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        w_g_tensor = (1/9) * torch.ones((1, 1, 3, 3), dtype=torch.float32)
        self.register_buffer('w_g', w_g_tensor)
        
        # Thresholds
        self.T_g = 10
        self.T_ffi = 10
        self.T_sfa = 0.0015
        self.w1 = 0.3
        self.w2 = 0.6

        alpha_mask_tensor = torch.tensor([
            self.alp[2], self.alp[1], self.alp[2],  # top row: diagonal, adjacent, diagonal
            self.alp[1], self.alp[0], self.alp[1],  # middle row: adjacent, center, adjacent
            self.alp[2], self.alp[1], self.alp[2]   # bottom row: diagonal, adjacent, diagonal
        ], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('W_cur', W_on_tensor * alpha_mask_tensor)
        self.register_buffer('W_pre', W_on_tensor * (1 - alpha_mask_tensor) )
        
        self.setup()
    
    def setup(self):
        super().setup()
        # ON pathway states
        self.last_P_on = None
        self.last_E_on = None
        self.last_I_on = None
        
        # OFF pathway states
        self.last_P_off = None
        self.last_E_off = None
        self.last_I_off = None
        
        # Frequency adaptation states
        self.last_U = None
        self.last_U_diff = None
        self.last_U_diff2 = None
        self.last_F_judge = 0
        self.mp = torch.tensor(0.5, dtype=torch.float32)
    
    def delay_conv2d(self, cur_p, pre_p):
        out_cur = F.conv2d(cur_p, self.W_cur, padding=1)
        out_pre = F.conv2d(pre_p, self.W_pre, padding=1)
        return out_cur + out_pre
    
    def model_structure(self, ipt_tensor):
        """
        LGMD1 model structure with ON-OFF pathways and spike frequency adaptation.
        
        Args:
            ipt_tensor: input image, shape (1, 1, H, W)
        
        Returns:
            mp: membrane potential (output)
        """
        # P Layer - frame difference
        self.P = ipt_tensor - self.last_ipt
        
        # ON-OFF mechanisms
        self.P_on = torch.clamp(self.P, min=0) + 0.1 * self.last_P_on
        self.P_off = -torch.clamp(self.P, max=0)+ 0.1 * self.last_P_off
        
        # IE Layer - ON pathway
        self.E_on = self.P_on.clone()
        self.I_on = self.delay_conv2d(self.P_on, self.last_I_on)
        
        # IE Layer - OFF pathway
        self.E_off = self.delay_conv2d(self.P_off, self.last_E_off)
        self.I_off = self.P_off.clone()
        
        # S Layer - spatial summation
        self.S_on = self.E_on - self.w1 * self.I_on
        self.S_off = self.E_off - self.w2 * self.I_off
        self.S = self.S_on + self.S_off
        
        # G Layer - grouping mechanism
        self.Ce = F.conv2d(self.S, self.w_g, padding=1)
        self.G_hat = self.Ce.clone()
        self.G_hat = torch.where(self.G_hat < self.T_g, torch.zeros_like(self.G_hat), self.G_hat)
        
        # LGMD cell computation
        self.MP = torch.mean(torch.abs(self.G_hat))
        
        # Membrane potential
        self.U = 1 / (1 + torch.exp(-torch.abs(self.MP)))
        
        # Activity level
        self.F = torch.mean(torch.abs(self.P))
        self.F_judge = self.tau_f * self.F + (1 - self.tau_f) * self.last_F_judge
        
        ''' Frequency adaptation (spiking frequency adaptation) '''
        if self.F_judge >= self.T_ffi:
            self.U = torch.tensor(0.5, dtype=torch.float32, device=self.U.device)
        
        self.U_diff = self.U - self.last_U
        U_diff2 = self.U_diff - self.last_U_diff
        
        if U_diff2 >= self.T_sfa:
            self.mp = self.sig_slow * self.U
        elif U_diff2 < self.T_sfa and self.U_diff >= self.T_sfa:
            self.mp = self.sig_fast * self.U
        elif self.U_diff < self.T_sfa:
            self.mp = self.sig_fast * (self.mp + self.U_diff)
        
        return self.mp.clamp(min=0.5)
    
    def forward(self, ipt_tensor):
        if self.last_ipt is None:
            # 第一帧：初始化所有状态为全0
            self.mp = ipt_tensor.new_tensor([0.5])
            self.last_U = torch.tensor(0.5, dtype=torch.float32, device=ipt_tensor.device)
            self.last_P_on = torch.zeros_like(ipt_tensor)
            self.last_P_off = torch.zeros_like(ipt_tensor)
            self.last_E_on = torch.zeros_like(ipt_tensor)
            self.last_E_off = torch.zeros_like(ipt_tensor)
            self.last_I_on = torch.zeros_like(ipt_tensor)
            self.last_I_off = torch.zeros_like(ipt_tensor)
            self.last_U_diff = torch.tensor(0.0, dtype=torch.float32, device=ipt_tensor.device)
            self.last_F_judge = torch.tensor(0.0, dtype=torch.float32, device=ipt_tensor.device)
        else:
            # 第二帧及以后：正常走模型逻辑
            self.mp = self.model_structure(ipt_tensor)
            
            # 保存当前状态给下一帧使用
            self.last_P_on = self.P_on.detach().clone()
            self.last_P_off = self.P_off.detach().clone()
            self.last_E_on = self.E_on.detach().clone()
            self.last_E_off = self.E_off.detach().clone()
            self.last_I_on = self.I_on.detach().clone()
            self.last_I_off = self.I_off.detach().clone()
            self.last_U = self.U.detach().clone()
            self.last_U_diff = self.U_diff.detach().clone()
            self.last_F_judge = self.F_judge.item()

        self.last_mp = self.mp.detach().clone()
        self.last_ipt = ipt_tensor.detach().clone()
        return self.mp


class LGMD1_LGMD2_CascadeNetwork_Dual_Channel(nn.Module):
    """
    PyTorch implementation converted from Matlab LGMD1-LGMD2 cascade dual-channel model.
    Ref:
        - Li J, Sun X, Li H, et al. On the ensemble of collision perception neuron models towards 
            ultra-selectivity[C]//2023 International Joint Conference on Neural Networks 
            (IJCNN). IEEE, 2023: 1-8.
        - https://github.com/fuqinbing/Ensemble-of-LGMD-Models/blob/main/LGMD1_LGMD2_CascadeNetwork_Dual_Channel.m
    """

    def __init__(
        self,
        gaussian_size=7,
        gaussian_sigma=1.5,
        dc=0.1,
        clip_point=0.0,
        on_exp=1.0,
        off_exp=1.0,
        ccn=1.0,
        tsfa=0.01,
        tffi=40.0,
        base_on=0.3,
        base_off=0.15,
        vth=0.9,
        tin=6.0,
        vrest=0.5,
        spike_window=64,
    ):
        super().__init__()
        self.dc = dc
        self.clip_point = clip_point
        self.on_exp = on_exp
        self.off_exp = off_exp
        self.ccn = ccn
        self.tsfa = tsfa
        self.tffi = tffi
        self.base_on = base_on
        self.base_off = base_off
        self.vth = vth
        self.tin = tin
        self.vrest = vrest
        self.gaussian_size = gaussian_size
        self.gaussian_sigma = gaussian_sigma

        self.register_buffer("delay_hp", torch.tensor(0.1, dtype=torch.float32))
        self.register_buffer("delay_ffi", torch.tensor([0.2, 0.5, 0.3], dtype=torch.float32))
        self.register_buffer("delay_on", torch.tensor([0.2, 0.5, 0.3], dtype=torch.float32))
        self.register_buffer("delay_off", torch.tensor([0.2, 0.5, 0.3], dtype=torch.float32))

        kernel_e = torch.tensor(
            [[0.125, 0.25, 0.125], [0.25, 0.0, 0.25], [0.125, 0.25, 0.125]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        kernel_on = torch.tensor(
            [[0.05, 0.1, 0.05], [0.1, 0.0, 0.1], [0.05, 0.1, 0.05]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        kernel_off = torch.tensor(
            [[0.125, 0.25, 0.125], [0.25, 0.0, 0.25], [0.125, 0.25, 0.125]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        kernel_g = torch.ones((1, 1, 3, 3), dtype=torch.float32) / 9.0
        kernel_contrast = torch.ones((1, 1, 3, 3), dtype=torch.float32) / 9.0

        self.register_buffer("kernel_e", kernel_e)
        self.register_buffer("kernel_on", kernel_on)
        self.register_buffer("kernel_off", kernel_off)
        self.register_buffer("kernel_g", kernel_g)
        self.register_buffer("kernel_contrast", kernel_contrast)

        self.spike_window = int(spike_window)
        self.setup()

    def setup(self):
        self.initialized = False
        self.t = 0
        self.last_spike_age = 1e9

        self.last_lgmd1_photo = None
        self.last_lgmd1_ons = None
        self.last_lgmd1_offs = None
        self.last_lgmd1_ons_delay = None
        self.last2_lgmd1_ons_delay = None
        self.last_lgmd1_offs_delay = None
        self.last2_lgmd1_offs_delay = None
        self.last_lgmd1_sum = None

        self.last_lgmd2_ons = None
        self.last_lgmd2_offs = None
        self.last_lgmd2_ons_delay = None
        self.last2_lgmd2_ons_delay = None
        self.last_lgmd2_offs_delay = None
        self.last2_lgmd2_offs_delay = None
        self.last_lgmd2_group = None
        self.last_lgmd2_sfa = None

        self.last_ffi = None
        self.last2_ffi = None
        self.last_vlgmd = None
        self.spi_buffer = None
        self.spi_idx = 0

    def _ensure_state(self, x):
        if self.initialized:
            return
        z = torch.zeros_like(x)
        zs = x.new_tensor(0.0)

        self.last_lgmd1_photo = z.clone()
        self.last_lgmd1_ons = z.clone()
        self.last_lgmd1_offs = z.clone()
        self.last_lgmd1_ons_delay = z.clone()
        self.last2_lgmd1_ons_delay = z.clone()
        self.last_lgmd1_offs_delay = z.clone()
        self.last2_lgmd1_offs_delay = z.clone()
        self.last_lgmd1_sum = z.clone()

        self.last_lgmd2_ons = z.clone()
        self.last_lgmd2_offs = z.clone()
        self.last_lgmd2_ons_delay = z.clone()
        self.last2_lgmd2_ons_delay = z.clone()
        self.last_lgmd2_offs_delay = z.clone()
        self.last2_lgmd2_offs_delay = z.clone()
        self.last_lgmd2_group = z.clone()
        self.last_lgmd2_sfa = z.clone()

        self.last_ffi = zs.clone()
        self.last2_ffi = zs.clone()
        self.last_vlgmd = zs.clone()
        self.spi_buffer = torch.zeros(self.spike_window, dtype=torch.float32, device=x.device)
        self.initialized = True

    def _highpass(self, pre_in, cur_in, pre_out, a):
        return a * (pre_out + cur_in - pre_in)

    @staticmethod
    def _highpass2(pre_in, cur_in):
        return cur_in - pre_in

    @staticmethod
    def _lowpass(cur_in, pre_out, prepre_out, a3):
        return a3[0] * cur_in + a3[1] * pre_out + a3[2] * prepre_out

    def _halfwave_on(self, a, b):
        mask = (a - self.clip_point >= 0).to(a.dtype)
        return a * mask + self.dc * b

    def _halfwave_off(self, a, b):
        mask = (a - self.clip_point < 0).to(a.dtype)
        return torch.abs(a) * mask + self.dc * b

    @staticmethod
    def _competing(exc, inh, wi):
        gate = (exc * inh >= 0).to(exc.dtype)
        return (exc - inh * wi) * gate

    def _polarity_sum(self, on_exc, off_exc):
        return torch.pow(torch.clamp(on_exc, min=0), self.on_exp) + torch.pow(torch.clamp(off_exc, min=0), self.off_exp)

    def _normalize_contrast(self, signal):
        local_abs = F.conv2d(torch.abs(signal), self.kernel_contrast, padding='same')
        return torch.tanh(signal / (self.ccn + local_abs))

    def _thresholding(self, a):
        return torch.where(a >= self.tsfa, a, torch.zeros_like(a))

    def _sfa_profile(self, pre_out, pre_in, cur_in):
        diff_in = cur_in - pre_in
        c = (diff_in <= self.tsfa).to(cur_in.dtype)
        cur_tmp = self.delay_hp * cur_in + self.delay_hp * (pre_out - pre_in) * c
        return torch.clamp(cur_tmp, min=0)

    def _spiking(self, vpre, vcur):
        return torch.where(vpre + vcur - self.vth > 0, vcur.new_tensor(1.0), vcur.new_tensor(0.0))

    def _lif_neuron(self, vpre, vcur, spike):
        if spike.item() >= 1.0:
            self.last_spike_age = 0.0
        else:
            self.last_spike_age += 1.0
        return vpre * torch.exp(vpre.new_tensor(-self.last_spike_age / self.tin)) + vcur - spike * self.vth

    def forward(self, img_prev, img_cur=None):
        """
        Args:
            img_prev: previous gray frame, shape (N, 1, H, W).
            img_cur: current gray frame, shape (N, 1, H, W). If None, img_prev is treated as current frame.

        Returns:
            membrane_potential: scalar tensor.
        """
        if img_cur is None:
            img_cur = img_prev
            img_prev = self.last_lgmd1_photo if self.initialized else torch.zeros_like(img_cur)

        self._ensure_state(img_cur)

        lgmd1_photo = self._highpass(img_prev, img_cur, self.last_lgmd1_photo, self.delay_hp)
        ncell = float(img_cur.shape[-2] * img_cur.shape[-1])
        tmp_ffi = torch.sum(torch.abs(lgmd1_photo)) / ncell
        ffi = self._lowpass(tmp_ffi, self.last_ffi, self.last2_ffi, self.delay_ffi)

        wi_off = torch.maximum(ffi / self.tffi, img_cur.new_tensor(self.base_off))
        wi_on = torch.maximum(2.0 * wi_off, img_cur.new_tensor(self.base_on))

        blurred = gaussian_blur(lgmd1_photo, self.gaussian_size, self.gaussian_sigma)
        lgmd1_ons = self._halfwave_on(blurred, self.last_lgmd1_ons)
        lgmd1_offs = self._halfwave_off(blurred, self.last_lgmd1_offs)

        lgmd1_exc_on = F.conv2d(lgmd1_ons, self.kernel_e, padding='same')
        lgmd1_exc_off = F.conv2d(lgmd1_offs, self.kernel_e, padding='same')
        lgmd1_ons_delay = self._lowpass(lgmd1_exc_on, self.last_lgmd1_ons_delay, self.last2_lgmd1_ons_delay, self.delay_off)
        lgmd1_offs_delay = self._lowpass(lgmd1_exc_off, self.last_lgmd1_offs_delay, self.last2_lgmd1_offs_delay, self.delay_off)

        lgmd1_inh_on = F.conv2d(lgmd1_ons_delay, self.kernel_off, padding='same')
        lgmd1_inh_off = F.conv2d(lgmd1_offs_delay, self.kernel_off, padding='same')
        s1_on = torch.clamp(self._competing(lgmd1_exc_on, lgmd1_inh_on, wi_off), min=0)
        s1_off = torch.clamp(self._competing(lgmd1_exc_off, lgmd1_inh_off, wi_off), min=0)
        lgmd1_sum = self._polarity_sum(s1_on, s1_off)

        lgmd2_photo = self._highpass2(self.last_lgmd1_sum, lgmd1_sum)
        lgmd2_ons = self._halfwave_on(lgmd2_photo, self.last_lgmd2_ons)
        lgmd2_offs = self._halfwave_off(lgmd2_photo, self.last_lgmd2_offs)

        lgmd2_exc_on = F.conv2d(lgmd2_ons, self.kernel_e, padding='same')
        lgmd2_exc_off = F.conv2d(lgmd2_offs, self.kernel_e, padding='same')
        lgmd2_ons_delay = self._lowpass(lgmd2_exc_on, self.last_lgmd2_ons_delay, self.last2_lgmd2_ons_delay, self.delay_on)
        lgmd2_offs_delay = self._lowpass(lgmd2_exc_off, self.last_lgmd2_offs_delay, self.last2_lgmd2_offs_delay, self.delay_off)

        lgmd2_inh_on = F.conv2d(lgmd2_ons_delay, self.kernel_on, padding='same')
        lgmd2_inh_off = F.conv2d(lgmd2_offs_delay, self.kernel_off, padding='same')
        s2_on = torch.clamp(self._competing(lgmd2_exc_on, lgmd2_inh_on, wi_on), min=0)
        s2_off = torch.clamp(self._competing(lgmd2_exc_off, lgmd2_inh_off, wi_off), min=0)
        lgmd2_sum = self._polarity_sum(s2_on, s2_off)

        lgmd1_compressed = self._normalize_contrast(lgmd2_sum)
        lgmd2_group = self._thresholding(F.conv2d(lgmd1_compressed, self.kernel_g, padding='same'))
        lgmd2_sfa = self._sfa_profile(self.last_lgmd2_sfa, self.last_lgmd2_group, lgmd2_group)

        vlgmd = torch.sum(lgmd2_sfa)
        spike = self._spiking(self.last_vlgmd, vlgmd)
        self.spi_buffer[self.spi_idx] = spike
        self.spi_idx = (self.spi_idx + 1) % self.spike_window
        membrane_potential = self._lif_neuron(self.last_vlgmd, vlgmd, spike) + self.vrest

        self.last2_ffi = self.last_ffi.detach().clone()
        self.last_ffi = ffi.detach().clone()
        self.last_lgmd1_photo = lgmd1_photo.detach().clone()
        self.last_lgmd1_ons = lgmd1_ons.detach().clone()
        self.last_lgmd1_offs = lgmd1_offs.detach().clone()
        self.last2_lgmd1_ons_delay = self.last_lgmd1_ons_delay.detach().clone()
        self.last_lgmd1_ons_delay = lgmd1_ons_delay.detach().clone()
        self.last2_lgmd1_offs_delay = self.last_lgmd1_offs_delay.detach().clone()
        self.last_lgmd1_offs_delay = lgmd1_offs_delay.detach().clone()
        self.last_lgmd1_sum = lgmd1_sum.detach().clone()
        self.last_lgmd2_ons = lgmd2_ons.detach().clone()
        self.last_lgmd2_offs = lgmd2_offs.detach().clone()
        self.last2_lgmd2_ons_delay = self.last_lgmd2_ons_delay.detach().clone()
        self.last_lgmd2_ons_delay = lgmd2_ons_delay.detach().clone()
        self.last2_lgmd2_offs_delay = self.last_lgmd2_offs_delay.detach().clone()
        self.last_lgmd2_offs_delay = lgmd2_offs_delay.detach().clone()
        self.last_lgmd2_group = lgmd2_group.detach().clone()
        self.last_lgmd2_sfa = lgmd2_sfa.detach().clone()
        self.last_vlgmd = vlgmd.detach().clone()
        self.t += 1

        return 1/(1+torch.exp(-200*torch.mean(lgmd2_sfa)))


class LGMD2_LGMD1_CascadeNetwork_Dual_Channel(LGMD1_LGMD2_CascadeNetwork_Dual_Channel):
    """
    PyTorch implementation converted from Matlab LGMD2-LGMD1 cascade dual-channel model.
    Ref:
        - Li J, Sun X, Li H, et al. On the ensemble of collision perception neuron models towards
            ultra-selectivity[C]//2023 International Joint Conference on Neural Networks
            (IJCNN). IEEE, 2023: 1-8.
        - https://github.com/fuqinbing/Ensemble-of-LGMD-Models/blob/main/LGMD2_LGMD1_CascadeNetwork_Dual_Channel.m
    """

    def setup(self):
        self.initialized = False
        self.t = 0
        self.last_spike_age = 1e9

        # First stage is LGMD2
        self.last_lgmd2_photo = None
        self.last_lgmd2_ons = None
        self.last_lgmd2_offs = None
        self.last_lgmd2_ons_delay = None
        self.last2_lgmd2_ons_delay = None
        self.last_lgmd2_offs_delay = None
        self.last2_lgmd2_offs_delay = None
        self.last_lgmd2_sum = None

        # Second stage is LGMD1
        self.last_lgmd1_ons = None
        self.last_lgmd1_offs = None
        self.last_lgmd1_ons_delay = None
        self.last2_lgmd1_ons_delay = None
        self.last_lgmd1_offs_delay = None
        self.last2_lgmd1_offs_delay = None
        self.last_lgmd2_group = None
        self.last_lgmd2_sfa = None

        self.last_ffi = None
        self.last2_ffi = None
        self.last_vlgmd = None
        self.spi_buffer = None
        self.spi_idx = 0

    def _ensure_state(self, x):
        if self.initialized:
            return
        z = torch.zeros_like(x)
        zs = x.new_tensor(0.0)

        self.last_lgmd2_photo = z.clone()
        self.last_lgmd2_ons = z.clone()
        self.last_lgmd2_offs = z.clone()
        self.last_lgmd2_ons_delay = z.clone()
        self.last2_lgmd2_ons_delay = z.clone()
        self.last_lgmd2_offs_delay = z.clone()
        self.last2_lgmd2_offs_delay = z.clone()
        self.last_lgmd2_sum = z.clone()

        self.last_lgmd1_ons = z.clone()
        self.last_lgmd1_offs = z.clone()
        self.last_lgmd1_ons_delay = z.clone()
        self.last2_lgmd1_ons_delay = z.clone()
        self.last_lgmd1_offs_delay = z.clone()
        self.last2_lgmd1_offs_delay = z.clone()
        self.last_lgmd1_group = z.clone()
        self.last_lgmd1_sfa = z.clone()

        self.last_ffi = zs.clone()
        self.last2_ffi = zs.clone()
        self.last_vlgmd = zs.clone()
        self.spi_buffer = torch.zeros(self.spike_window, dtype=torch.float32, device=x.device)
        self.initialized = True

    def forward(self, img_prev, img_cur=None):
        if img_cur is None:
            img_cur = img_prev
            img_prev = self.last_lgmd2_photo if self.initialized else torch.zeros_like(img_cur)

        self._ensure_state(img_cur)

        # Stage 1: LGMD2
        lgmd2_photo = self._highpass(img_prev, img_cur, self.last_lgmd2_photo, self.delay_hp)
        ncell = float(img_cur.shape[-2] * img_cur.shape[-1])
        tmp_ffi = torch.sum(torch.abs(lgmd2_photo)) / ncell
        ffi = self._lowpass(tmp_ffi, self.last_ffi, self.last2_ffi, self.delay_ffi)

        wi_off = torch.maximum(ffi / self.tffi, img_cur.new_tensor(self.base_off))
        wi_on = torch.maximum(2.0 * wi_off, img_cur.new_tensor(self.base_on))

        blurred = gaussian_blur(lgmd2_photo, self.gaussian_size, self.gaussian_sigma)
        lgmd2_ons = self._halfwave_on(blurred, self.last_lgmd2_ons)
        lgmd2_offs = self._halfwave_off(blurred, self.last_lgmd2_offs)

        lgmd2_exc_on = F.conv2d(lgmd2_ons, self.kernel_e, padding='same')
        lgmd2_exc_off = F.conv2d(lgmd2_offs, self.kernel_e, padding='same')
        lgmd2_ons_delay = self._lowpass(lgmd2_exc_on, self.last_lgmd2_ons_delay, self.last2_lgmd2_ons_delay, self.delay_on)
        lgmd2_offs_delay = self._lowpass(lgmd2_exc_off, self.last_lgmd2_offs_delay, self.last2_lgmd2_offs_delay, self.delay_off)

        lgmd2_inh_on = F.conv2d(lgmd2_ons_delay, self.kernel_on, padding='same')
        lgmd2_inh_off = F.conv2d(lgmd2_offs_delay, self.kernel_off, padding='same')
        s2_on = torch.clamp(self._competing(lgmd2_exc_on, lgmd2_inh_on, wi_on), min=0)
        s2_off = torch.clamp(self._competing(lgmd2_exc_off, lgmd2_inh_off, wi_off), min=0)
        lgmd2_sum = self._polarity_sum(s2_on, s2_off)

        # Stage 2: LGMD1
        lgmd1_photo = self._highpass2(self.last_lgmd2_sum, lgmd2_sum)
        lgmd1_ons = self._halfwave_on(lgmd1_photo, self.last_lgmd1_ons)
        lgmd1_offs = self._halfwave_off(lgmd1_photo, self.last_lgmd1_offs)

        lgmd1_exc_on = F.conv2d(lgmd1_ons, self.kernel_e, padding='same')
        lgmd1_exc_off = F.conv2d(lgmd1_offs, self.kernel_e, padding='same')
        lgmd1_ons_delay = self._lowpass(lgmd1_exc_on, self.last_lgmd1_ons_delay, self.last2_lgmd1_ons_delay, self.delay_off)
        lgmd1_offs_delay = self._lowpass(lgmd1_exc_off, self.last_lgmd1_offs_delay, self.last2_lgmd1_offs_delay, self.delay_off)

        lgmd1_inh_on = F.conv2d(lgmd1_ons_delay, self.kernel_off, padding='same')
        lgmd1_inh_off = F.conv2d(lgmd1_offs_delay, self.kernel_off, padding='same')
        s1_on = torch.clamp(self._competing(lgmd1_exc_on, lgmd1_inh_on, wi_off), min=0)
        s1_off = torch.clamp(self._competing(lgmd1_exc_off, lgmd1_inh_off, wi_off), min=0)
        lgmd1_sum = self._polarity_sum(s1_on, s1_off)

        lgmd1_compressed = self._normalize_contrast(lgmd1_sum)
        lgmd1_group = self._thresholding(F.conv2d(lgmd1_compressed, self.kernel_g, padding='same'))
        lgmd1_sfa = self._sfa_profile(self.last_lgmd1_sfa, self.last_lgmd1_group, lgmd1_group)

        vlgmd = torch.sum(lgmd1_sfa)
        spike = self._spiking(self.last_vlgmd, vlgmd)
        self.spi_buffer[self.spi_idx] = spike
        self.spi_idx = (self.spi_idx + 1) % self.spike_window
        membrane_potential = self._lif_neuron(self.last_vlgmd, vlgmd, spike) + self.vrest

        self.last2_ffi = self.last_ffi.detach().clone()
        self.last_ffi = ffi.detach().clone()
        self.last_lgmd2_photo = lgmd2_photo.detach().clone()
        self.last_lgmd2_ons = lgmd2_ons.detach().clone()
        self.last_lgmd2_offs = lgmd2_offs.detach().clone()
        self.last2_lgmd2_ons_delay = self.last_lgmd2_ons_delay.detach().clone()
        self.last_lgmd2_ons_delay = lgmd2_ons_delay.detach().clone()
        self.last2_lgmd2_offs_delay = self.last_lgmd2_offs_delay.detach().clone()
        self.last_lgmd2_offs_delay = lgmd2_offs_delay.detach().clone()
        self.last_lgmd2_sum = lgmd2_sum.detach().clone()
        self.last_lgmd1_ons = lgmd1_ons.detach().clone()
        self.last_lgmd1_offs = lgmd1_offs.detach().clone()
        self.last2_lgmd1_ons_delay = self.last_lgmd1_ons_delay.detach().clone()
        self.last_lgmd1_ons_delay = lgmd1_ons_delay.detach().clone()
        self.last2_lgmd1_offs_delay = self.last_lgmd1_offs_delay.detach().clone()
        self.last_lgmd1_offs_delay = lgmd1_offs_delay.detach().clone()
        self.last_lgmd1_group = lgmd1_group.detach().clone()
        self.last_lgmd1_sfa = lgmd1_sfa.detach().clone()
        self.last_vlgmd = vlgmd.detach().clone()
        self.t += 1

        return 1/(1+torch.exp(-200*torch.mean(lgmd1_sfa)))


class ALGMD(nn.Module):
    """
    Acceleration-based Looming Sensitive Motion Detection model using Izhikevich neuron.
    
    Combines velocity and acceleration detection with spiking neural response.
    
    Ref:
        - Zhao J, Xie Q, Shuang F, et al. An angular acceleration based looming detector 
            for moving UAVs[J]. Biomimetics, 2024, 9(1).
        - https://github.com/chasen-xqs/ALGMD
    """
    
    def __init__(self, max_frames=300, r1=1, r2=1, device='cpu'):
        super().__init__()
        self.max_frames = max_frames
        self.device = device
        
        # Izhikevich neuron parameters
        self.a = 0.02
        self.b = 0.2
        self.c = -65.0
        self.d = 2.0
        self.v_threshold = 30.0
        self.v_rest = -25.0
        
        self.r1 = r1
        self.r2 = r2
        
        self.conv_I2 = nn.Conv2d(1, 1, kernel_size=2*r1+1, padding='same', bias=False)
        self.conv_I2.weight.data = self._generate_kernel(r1, r1, p=0)
        self.conv_I2.weight.requires_grad = False # 冻结参数，不参与训练更新
        
        self.conv_I3 = nn.Conv2d(1, 1, kernel_size=2*r2+1, padding='same', bias=False)
        self.conv_I3.weight.data = self._generate_kernel(r2, r2, p=1)
        self.conv_I3.weight.requires_grad = False
        
        r_A = 2 * r2 - r1 + 1
        self.conv_A = nn.Conv2d(1, 1, kernel_size=2*r_A+1, padding='same', bias=False)
        self.conv_A.weight.data = torch.ones((1, 1, 2*r_A+1, 2*r_A+1), dtype=torch.float32)
        self.conv_A.weight.requires_grad = False
        
        self.setup()
    
    def _generate_kernel(self, sigma, r, p=0):
        x = torch.arange(-r, r+1, dtype=torch.float32)
        y = torch.arange(-r, r+1, dtype=torch.float32)
        grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
        
        amplitude = 1.0 / (2 * torch.tensor(torch.pi) * sigma**2)
        exponent = (grid_x**2 + grid_y**2) / (2 * sigma**2)
        
        if p == 0:
            Z = amplitude * (1 - torch.exp(-exponent))
        else:
            Z = amplitude * torch.exp(-exponent)
        
        Z = torch.where(Z == 0, torch.tensor(10.0), Z)
        Z_min = torch.min(Z)
        k_scale = torch.round(1.0 / Z_min).int()
        Z = torch.where(Z == 10.0, torch.tensor(0.0), Z)
        Z = Z * k_scale.float()
        
        for i in range(2*r+1):
            for j in range(2*r+1):
                if (i - r)**2 + (j - r)**2 > r**2:
                    Z[i, j] = 0
        
        Z[r, r] = 1.0
        return Z.unsqueeze(0).unsqueeze(0)
    
    def _izhikevich_step(self, v_prev, u_prev, I, dt):
        dv_dt = 0.04 * v_prev**2 + 5*v_prev + 140 - u_prev + I
        du_dt = self.a * (self.b * v_prev - u_prev)
        
        v = v_prev + dt * dv_dt
        u = u_prev + dt * du_dt
        
        # 确保全部是 Tensor 计算，不打断梯度图
        spike_mask = (v >= self.v_threshold).float()
        v = torch.where(v >= self.v_threshold, torch.tensor(self.c, dtype=v.dtype, device=v.device), v)
        u = u + spike_mask * self.d
        
        return v, u
    
    def setup(self):
        """Initialize state variables."""
        self.img_past = None
        self.img_diff1 = None
        self.img_diff2 = None
        
        self.v_prev = None
        self.u_prev = None
        self.v_window = None  # 记录最近3帧电位，用于检测 spike 峰值
        
        self.spike_times = []
        self.spike_amplitudes = []
        
        self.frame_count = 0
        self.valid_frame_count = 0
        self.mp = torch.tensor(0.5, dtype=torch.float32)
    
    def forward(self, img_cur):
        img_cur = img_cur / 255.0

        # 延迟初始化 (Lazy initialization)，适应输入张量的 shape 和 device
        if self.img_past is None:
            self.img_past = torch.zeros_like(img_cur)
            self.img_diff1 = torch.zeros_like(img_cur)
            self.img_diff2 = torch.zeros_like(img_cur)
            self.v_prev = torch.tensor(-65.0, device=img_cur.device)
            self.u_prev = torch.tensor(0.0, device=img_cur.device)
            self.v_window = torch.full((3,), -65.0, device=img_cur.device)

        img_diff3 = torch.abs(img_cur - self.img_past)
        self.valid_frame_count += 1
        
        # 使用配置好的 nn.Conv2d 层进行计算
        layer_V2 = self.img_diff2 - self.conv_I2(self.img_diff1)
        layer_V23 = self.img_diff2 - self.conv_I3(self.img_diff1)
        layer_V3 = img_diff3 - self.conv_I3(self.img_diff2)
        
        # 保证 device 一致性
        layer_V3 = torch.where(layer_V3 < 0.1, torch.tensor(0.0, device=img_cur.device), layer_V3)
        layer_temp = self.conv_A(layer_V3)
        
        binary_V2 = (layer_V2.abs() > 0.03).float()
        binary_V23 = (layer_V23.abs() > 0.1).float()
        layer_A = layer_temp * binary_V2 * (1.0 - binary_V23)
        
        A_num = torch.sum(layer_A > 0.1).float()
        V3_num = torch.sum(layer_V3 > 0).float()
        
        # 修复4: 直接将 soma 作为 Tensor 传入，移除 soma.item() 切断计算图的问题
        soma = A_num * V3_num 
        
        # dt 更新也保持在张量系统内（这里也可以是纯 float，但传入 Izhikevich 必须和张量兼容）
        dt = 1e-3 * float(self.valid_frame_count) / 30.0
        
        v_curr, u_curr = self._izhikevich_step(self.v_prev, self.u_prev, soma, dt)
        
        # 更新神经元状态
        self.v_prev = v_curr
        self.u_prev = u_curr
        
        # 滑动窗口维护最近3帧数据，用于 Spike 检测，避免操作 Python List
        self.v_window = torch.cat([self.v_window[1:], v_curr.unsqueeze(0)])
        
        if self.valid_frame_count > 3:
            if (self.v_window[1] > self.v_window[2] and 
                self.v_window[1] > self.v_window[0] and 
                self.v_window[1] > self.v_rest):
                self.spike_times.append(self.valid_frame_count - 2)
                self.spike_amplitudes.append(self.v_window[1].item())
        
        v_normalized = torch.clamp(v_curr - self.c, min=0)
        self.mp = torch.sigmoid(v_normalized)
        
        self.img_past = img_cur.clone()
        self.img_diff1 = self.img_diff2.clone()
        self.img_diff2 = img_diff3.clone()
        self.frame_count += 1
        
        return self.mp

    def get_spike_info(self):
        return {
            'spike_times': self.spike_times,
            'spike_amplitudes': self.spike_amplitudes,
            'total_spikes': len(self.spike_times)
        }


class EMD_LPLC2_GF(nn.Module):
    """Torch implementation of the Matlab GF pipeline for EMD-LPLC2-GF.

    `forward()` is online: it processes one frame at a time and updates all
    internal states.

    Expected input for `forward()`:
    - (1, 1, H, W)

    Ref:
        - Zhao J, Xi S, Li Y, et al. A fly inspired solution to looming detection for 
            collision avoidance[J]. IScience, 2023, 26(4).
        - https://github.com/zhaojunyu01/EMD-LPLC2-GF-model
    """

    def __init__(
        self,
        image_scale=0.25,
        frame_axis=0,
        step=0.01,
        substeps=20,
        normal_i1=2500.0,
        current_gain=250.0,
        v0=-60.0,
        vth=-50.0,
        vre=-70.0,
        normalize_input=True,
        device=None,
    ):
        super().__init__()
        self.image_scale = float(image_scale)
        self.frame_axis = frame_axis
        self.step = float(step)
        self.substeps = int(substeps)
        self.normal_i1 = float(normal_i1)
        self.current_gain = float(current_gain)
        self.v0 = float(v0)
        self.vth = float(vth)
        self.vre = float(vre)
        self.normalize_input = bool(normalize_input)

        self.lif_rest = -60.0
        self.lif_r = 10.0
        self.lif_tau = 0.30
        self.h = self.step / self.substeps

        i1_99_5ms = 2116.0 / self.normal_i1
        i1_98_5ms = 676.0 / self.normal_i1
        self.normal_vel = (i1_99_5ms - i1_98_5ms) / self.step

        d_low = 0.050 / self.step
        d_high = 0.250 / self.step
        self.register_buffer(
            "a_low",
            torch.tensor([1.0 / (d_low + 1.0), 1.0 - 1.0 / (d_low + 1.0)], dtype=torch.float32),
        )
        self.register_buffer(
            "b_high",
            torch.tensor([d_high / (d_high + 1.0), d_high / (d_high + 1.0)], dtype=torch.float32),
        )
        self.register_buffer("lplc2_kernel", self._build_lplc2_kernels(side=100))

        if device is not None:
            self.to(device)

        self.setup()

    def setup(self):
        self.reset_state()

    def reset_state(self):
        self.last_frame = None
        self.last_patt = None
        self.fh_before = None
        self.fd_on_before = None
        self.fd_off_before = None
        self.prev_count = None
        self.v_state = None
        self.last_output = None
        self.last_nums = None
        self.last_vel = None
        self.last_current = None
        self.last_lp = None
        self.last_lplc2 = None
        self.last_spike_times = []
        self.frame_index = 0
        self.valid_frame_index = 0
        return self.lplc2_kernel.device

    @staticmethod
    def _build_lplc2_kernels(side=100):
        if side % 2 != 0:
            raise ValueError("side must be even")

        hside = side // 2
        kernels = torch.zeros((4, 1, side, side), dtype=torch.float32)
        row_start = hside - 20
        row_end = hside + 20
        col_start = hside - 20
        col_end = hside + 20

        kernels[0, 0, row_start:row_end, hside:] = 1.0
        kernels[1, 0, row_start:row_end, :hside] = 1.0
        kernels[2, 0, hside:, col_start:col_end] = 1.0
        kernels[3, 0, :hside, col_start:col_end] = 1.0
        return kernels

    @staticmethod
    def _on_rect(input_tensor, threshold=0.0):
        return torch.where(input_tensor > threshold, torch.abs(input_tensor - threshold), torch.zeros_like(input_tensor))

    @staticmethod
    def _off_rect(input_tensor, threshold=0.05):
        return torch.where(input_tensor < threshold, torch.abs(input_tensor - threshold), torch.zeros_like(input_tensor))

    def _emd_step(self, picture, newpicture, fh_before, fd_on_before, fd_off_before):
        # picture/newpicture: (1,1,H,W)
        fh = self.b_high[0] * (newpicture - picture) + self.b_high[1] * fh_before
        on = self._on_rect(fh, 0.0)
        off = self._off_rect(fh, 0.05)

        fd_on = self.a_low[0] * on + self.a_low[1] * fd_on_before
        fd_off = self.a_low[0] * off + self.a_low[1] * fd_off_before

        # compute directional pairs over spatial dims (keep batch and channel dims)
        he_on = fd_on[..., :-1, :-1] * on[..., :-1, 1:]
        hi_on = on[..., :-1, :-1] * fd_on[..., :-1, 1:]
        ve_on = fd_on[..., :-1, :-1] * on[..., 1:, :-1]
        vi_on = on[..., :-1, :-1] * fd_on[..., 1:, :-1]

        he_off = fd_off[..., :-1, :-1] * off[..., :-1, 1:]
        hi_off = off[..., :-1, :-1] * fd_off[..., :-1, 1:]
        ve_off = fd_off[..., :-1, :-1] * off[..., 1:, :-1]
        vi_off = off[..., :-1, :-1] * fd_off[..., 1:, :-1]

        # remove the singleton channel dim so stacking yields (batch, 4, H-1, W-1)
        hi = (hi_on + hi_off).squeeze(1)
        he = (he_on + he_off).squeeze(1)
        vi = (vi_on + vi_off).squeeze(1)
        ve = (ve_on + ve_off).squeeze(1)

        lp = torch.stack([hi, he, vi, ve], dim=1)

        lplc2 = self._lplc2_from_lp(lp)
        return fh, fd_on, fd_off, lp, lplc2

    def _lplc2_from_lp(self, lp):
        threshold_1 = 1.5
        threshold_2 = -2.0

        # lp shape: (batch, 4, H, W)
        # 1. 拼接输入，形成 (batch, 4, H, W) 的张量
        diffs = torch.cat([
            lp[:, 0:1, ...] - lp[:, 1:2, ...],
            lp[:, 1:2, ...] - lp[:, 0:1, ...],
            lp[:, 2:3, ...] - lp[:, 3:4, ...],
            lp[:, 3:4, ...] - lp[:, 2:3, ...]
        ], dim=1)

        # 2. 统一 Padding
        pad_h = self.lplc2_kernel.shape[-2] // 2
        pad_w = self.lplc2_kernel.shape[-1] // 2
        padded_diffs = F.pad(diffs, (pad_w - 1, pad_w, pad_h - 1, pad_h))

        # 3. 核心优化：使用 groups=4 一次性完成 4 个方向的卷积！
        # self.lplc2_kernel 的 shape 正好是 (4, 1, 100, 100)，完美匹配 groups=4
        pre = F.conv2d(padded_diffs, self.lplc2_kernel, padding=0, groups=4)

        # pre: (batch,4,H,W)
        pre_lplc2_l = pre > threshold_1
        pre = pre * (pre > threshold_2)
        
        # combine directional responses per-pixel
        lplc2 = torch.abs(pre[:, 0, ...] * pre[:, 1, ...] * pre[:, 2, ...] * pre[:, 3, ...])
        mask = (pre_lplc2_l.sum(dim=1) >= 3)
        lplc2 = lplc2 * mask
        return lplc2

    def _lif_rhs(self, y, current):
        return (-y + self.lif_rest + self.lif_r * current) / self.lif_tau

    def _lif_rk4(self, v_prev, current):
        v_prev = torch.clamp(v_prev, min=-80.0)

        # 消除 new_tensor 造成的内存分配！直接和标量 h 运算即可
        k1 = self._lif_rhs(v_prev, current)
        k2 = self._lif_rhs(v_prev + k1 * self.h / 2.0, current)
        k3 = self._lif_rhs(v_prev + k2 * self.h / 2.0, current)
        k4 = self._lif_rhs(v_prev + k3 * self.h, current)
        
        return v_prev + (k1 + 2.0 * k2 + 2.0 * k3 + k4) * self.h / 6.0

    def forward(self, frame):
        # expect frame shape: (1,1,H,W)
        if self.normalize_input and frame.numel() > 0 and frame.max().item() > 1.0:
            frame = frame / 255.0

        if self.last_patt is None:
            self.last_frame = frame.detach().clone()
            self.last_patt = frame.detach().clone()
            self.fh_before = torch.zeros_like(frame)
            self.fd_on_before = torch.zeros_like(frame)
            self.fd_off_before = torch.zeros_like(frame)
            self.prev_count = None
            self.v_state = frame.new_tensor(self.v0)
            self.last_output = frame.new_tensor(0.5)
            self.last_nums = frame.new_tensor(0.0)
            self.last_vel = frame.new_tensor(0.0)
            self.last_current = frame.new_tensor(0.0)
            self.last_lp = None
            self.last_lplc2 = None
            self.frame_index += 1
            return self.last_output

        fh, fd_on, fd_off, lp, lplc2 = self._emd_step(
            self.last_patt, frame, self.fh_before, self.fd_on_before, self.fd_off_before
        )
        count = (lplc2 > 0).sum().to(torch.float32)
        i1 = count / frame.new_tensor(self.normal_i1)
        i2 = i1 if self.prev_count is None else self.prev_count / frame.new_tensor(self.normal_i1)
        vel = ((i1 - i2) / self.step) / frame.new_tensor(self.normal_vel)

        current = torch.where(
            i2 == 0.0, 
            torch.tensor(0.0, device=frame.device, dtype=frame.dtype), 
            self.current_gain * i1 * vel
        )

        # 优化 20 次的 RK4 循环
        vre_tensor = torch.tensor(self.vre, device=frame.device, dtype=frame.dtype)
        spikes_in_this_frame = 0  # 记录这 20 步内是否发生脉冲
        
        for _ in range(self.substeps):
            v_next = self._lif_rk4(self.v_state, current)
            # 判断是否激发 (得到一个 boolean Tensor)
            is_spike = v_next > self.vth
            # 累加激发次数 (全程保留在 GPU 上计算)
            spikes_in_this_frame += is_spike.int()
            # 使用 torch.where 重置电位，不打断 GPU 流水线
            self.v_state = torch.where(is_spike, vre_tensor, v_next)

        # 循环彻底结束后，只做 1 次 CPU-GPU 同步提取激发次数
        num_spikes = spikes_in_this_frame.item()
        if num_spikes > 0:
            # 如果在 substeps 中激发了多次，则记录多次帧索引
            self.last_spike_times.extend([self.valid_frame_index] * num_spikes)

        self.fh_before = fh.detach()
        self.fd_on_before = fd_on.detach()
        self.fd_off_before = fd_off.detach()
        self.last_frame = frame.detach()
        self.last_patt = frame.detach()
        self.prev_count = count.detach()
        self.valid_frame_index += 1
        self.frame_index += 1

        self.last_nums = count.detach()
        self.last_vel = vel.detach()
        self.last_current = current.detach()
        self.last_lp = lp.detach()
        self.last_lplc2 = lplc2.detach()

        v_normalized = torch.clamp(self.v_state - self.v0, min=0)
        self.last_output = torch.sigmoid(v_normalized)
        
        return self.last_output

