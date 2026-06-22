"""Loss functions used in patch generation."""

from typing import Tuple

import torch
import torch.nn as nn


class MaxProbExtractor(nn.Module):
    """MaxProbExtractor: extracts max class probability for class from YOLO output.

    Module providing the functionality necessary to extract the max class probability for one class from YOLO output.

    """

    def __init__(self, config):
        super(MaxProbExtractor, self).__init__()
        self.config = config

    def forward(self, output: torch.Tensor):
        """Output must be of the shape [batch, -1, 5 + num_cls]"""
        # get values necessary for transformation
        assert output.size(-1) == (5 + self.config.n_classes)

        class_confs = output[:, :, 5 : 5 + self.config.n_classes]  # [batch, -1, n_classes]
        objectness_score = output[:, :, 4]  # [batch, -1, 5 + num_cls] -> [batch, -1], no need to run sigmoid here

        if self.config.objective_class_id is not None:
            # norm probs for object classes to [0, 1]
            class_confs = torch.nn.Softmax(dim=2)(class_confs)
            # only select the conf score for the objective class
            class_confs = class_confs[:, :, self.config.objective_class_id]
        else:
            # get class with highest conf for each box if objective_class_id is None
            class_confs = torch.max(class_confs, dim=2)[0]  # [batch, -1, 4] -> [batch, -1]

        confs_if_object = self.config.loss_target(objectness_score, class_confs)
        max_conf, _ = torch.max(confs_if_object, dim=1)
        return max_conf


class SaliencyLoss(nn.Module):
    """
    Implementation of the colorfulness metric as the saliency loss.

    The smaller the value, the less colorful the image.
    Reference: https://infoscience.epfl.ch/record/33994/files/HaslerS03.pdf
    """

    def __init__(self):
        super(SaliencyLoss, self).__init__()

    def forward(self, adv_patch: torch.Tensor) -> torch.Tensor:
        """
        Args:
            adv_patch: Float Tensor of shape [C, H, W] where C=3 (R, G, B channels)
        """
        assert adv_patch.shape[0] == 3
        r, g, b = adv_patch
        rg = r - g
        yb = 0.5 * (r + g) - b

        mu_rg, sigma_rg = torch.mean(rg) + 1e-8, torch.std(rg) + 1e-8
        mu_yb, sigma_yb = torch.mean(yb) + 1e-8, torch.std(yb) + 1e-8
        sl = torch.sqrt(sigma_rg**2 + sigma_yb**2) + (0.3 * torch.sqrt(mu_rg**2 + mu_yb**2))
        return sl / torch.numel(adv_patch)


class TotalVariationLoss(nn.Module):
    """TotalVariationLoss: calculates the total variation of a patch.
    Module providing the functionality necessary to calculate the total vatiation (TV) of an adversarial patch.
    Reference: https://en.wikipedia.org/wiki/Total_variation
    """

    def __init__(self):
        super(TotalVariationLoss, self).__init__()

    def forward(self, adv_patch: torch.Tensor) -> torch.Tensor:
        """
        Args:
            adv_patch: Tensor of shape [C, H, W]
        """
        # calc diff in patch rows
        tvcomp_r = torch.sum(torch.abs(adv_patch[:, :, 1:] - adv_patch[:, :, :-1] + 0.000001), dim=0)
        tvcomp_r = torch.sum(torch.sum(tvcomp_r, dim=0), dim=0)
        # calc diff in patch columns
        tvcomp_c = torch.sum(torch.abs(adv_patch[:, 1:, :] - adv_patch[:, :-1, :] + 0.000001), dim=0)
        tvcomp_c = torch.sum(torch.sum(tvcomp_c, dim=0), dim=0)
        tv = tvcomp_r + tvcomp_c
        return tv / torch.numel(adv_patch)


# class TotalVariationLoss(nn.Module):
#     def __init__(self, alpha=0.7):
#         super(TotalVariationLoss, self).__init__()
#         self.alpha = alpha  # L1/L2混合比例

#     def forward(self, adv_patch: torch.Tensor) -> torch.Tensor:
#         # L1差异
#         diff_h = torch.abs(adv_patch[:, :, 1:] - adv_patch[:, :, :-1])
#         diff_v = torch.abs(adv_patch[:, 1:, :] - adv_patch[:, :-1, :])

#         # L2差异 
#         diff_h_sq = diff_h**2
#         diff_v_sq = diff_v**2

#         # 混合损失（保持原函数的归一化方式）
#         tv = (self.alpha*(torch.sum(diff_h_sq) + torch.sum(diff_v_sq)) + 
#              (1-self.alpha)*(torch.sum(diff_h) + torch.sum(diff_v))) 

#         return tv / torch.numel(adv_patch)  # 与原函数相同的归一化处理




class NPSLoss(nn.Module):
    """NMSLoss: calculates the non-printability-score loss of a patch.
    Module providing the functionality necessary to calculate the non-printability score (NMS) of an adversarial patch.
    However, a summation of the differences is used instead of the total product to calc the NPSLoss
    Reference: https://users.ece.cmu.edu/~lbauer/papers/2016/ccs2016-face-recognition.pdf
        Args:
            triplet_scores_fpath: str, path to csv file with RGB triplets sep by commas in newlines
            size: Tuple[int, int], Tuple with height, width of the patch
    """

    def __init__(self, triplet_scores_fpath: str, size: Tuple[int, int]):
        super(NPSLoss, self).__init__()
        self.printability_array = nn.Parameter(
            self.get_printability_array(triplet_scores_fpath, size), requires_grad=False
        )

    def forward(self, adv_patch):
        # calculate euclidean distance between colors in patch and colors in printability_array
        # square root of sum of squared difference
        color_dist = adv_patch - self.printability_array + 0.000001
        color_dist = color_dist**2
        color_dist = torch.sum(color_dist, 1) + 0.000001
        color_dist = torch.sqrt(color_dist)
        # use the min distance
        color_dist_prod = torch.min(color_dist, 0)[0]
        # calculate the nps by summing over all pixels
        nps_score = torch.sum(color_dist_prod, 0)
        nps_score = torch.sum(nps_score, 0)
        return nps_score / torch.numel(adv_patch)

    def get_printability_array(self, triplet_scores_fpath: str, size: Tuple[int, int]) -> torch.Tensor:
        """
        Get printability tensor array holding the rgb triplets (range [0,1]) loaded from triplet_scores_fpath
        Args:
            triplet_scores_fpath: str, path to csv file with RGB triplets sep by commas in newlines
            size: Tuple[int, int], Tuple with height, width of the patch
        """
        ref_triplet_list = []
        # read in reference printability triplets into a list
        with open(triplet_scores_fpath, "r", encoding="utf-8") as f:
            for line in f:
                ref_triplet_list.append(line.strip().split(","))

        p_h, p_w = size
        printability_array = []
        for ref_triplet in ref_triplet_list:
            r, g, b = map(float, ref_triplet)
            ref_tensor_img = torch.stack(
                [torch.full((p_h, p_w), r), torch.full((p_h, p_w), g), torch.full((p_h, p_w), b)]
            )
            printability_array.append(ref_tensor_img.float())
        return torch.stack(printability_array)


class FrequencyLoss(nn.Module):
    """频域损失，抑制高频成分生成"""
    
    def __init__(self, cutoff_ratio=0.3):
        super(FrequencyLoss, self).__init__()
        self.cutoff_ratio = cutoff_ratio

    def forward(self, adv_patch: torch.Tensor) -> torch.Tensor:
        """
        Args:
            adv_patch: Tensor of shape [C, H, W]
        """
        # 确保处理三维张量（单通道或多通道）
        if adv_patch.dim() == 3:
            patch = adv_patch
        else:
            raise ValueError("Input tensor must be 3D [C, H, W]")
            
        # 傅里叶变换
        freq = torch.fft.fft2(patch)
        freq_shifted = torch.fft.fftshift(freq)
        
        # 创建高通滤波器
        c, h, w = patch.shape[-3], patch.shape[-2], patch.shape[-1]
        cutoff = int(self.cutoff_ratio * min(h, w))
        mask = torch.ones_like(freq_shifted)
        center_h, center_w = h // 2, w // 2
        mask[..., 
             center_h - cutoff : center_h + cutoff, 
             center_w - cutoff : center_w + cutoff] = 0
        
        # 计算高频能量
        high_freq = freq_shifted * mask
        loss = torch.norm(high_freq)**2
        
        # 归一化处理
        return loss / (h * w * c)



class DistributionLoss(nn.Module):
    """分散化损失函数，鼓励补丁的有效信息分散化
    
    基于梯度方差的正则化方法，通过惩罚梯度分布的集中性，
    强制优化过程将补丁的有效攻击特征均匀分布在整个补丁区域。
    这有助于提高对抗局部反光（光斑）的鲁棒性。
    """
    
    def __init__(self, method='activation_entropy', lambda_dist=1.0):
        """
        Args:
            method: 分散化度量方法，可选 'gradient_variance' 或 'activation_entropy'
            lambda_dist: 分散化损失的权重系数
        """
        super(DistributionLoss, self).__init__()
        self.method = method
        self.lambda_dist = lambda_dist
        
    def forward(self, adv_patch: torch.Tensor, det_loss: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            adv_patch: 对抗补丁张量，形状 [C, H, W]
            det_loss: 检测损失（可选，用于梯度方法）
        """
        # 默认使用激活熵方法，避免梯度计算问题
        if self.method == 'activation_entropy':
            return self._activation_entropy_loss(adv_patch)
        elif self.method == 'spatial_distribution':
            return self._spatial_distribution_loss(adv_patch)
        # elif self.method == 'gradient_variance':
        #     # 警告：梯度方差方法可能导致梯度计算问题
        #     print("警告：梯度方差方法可能导致训练不稳定，建议使用activation_entropy方法")
        #     if det_loss is not None:
        #         return self._gradient_variance_loss(adv_patch, det_loss)
        #     else:
        #         # 如果det_loss为None，回退到激活熵方法
        #         return self._activation_entropy_loss(adv_patch)
        else:
            raise ValueError(f"Unsupported method: {self.method}")

    def _activation_entropy_loss(self, adv_patch: torch.Tensor) -> torch.Tensor:
        
        """基于激活熵的分散化损失
        
        计算补丁像素值的熵，鼓励像素值分布更均匀。
        高熵表示信息分布更分散。
        """
        # 将补丁展平为概率分布
        patch_flat = adv_patch.view(-1)
        
        # 添加小值避免log(0)
        patch_flat = patch_flat + 1e-8
        
        # 归一化为概率分布
        probabilities = patch_flat / torch.sum(patch_flat)
        
        # 计算熵
        entropy = -torch.sum(probabilities * torch.log(probabilities))
        
        # 最大化熵 = 最小化负熵
        entropy_loss = -entropy
        
        return self.lambda_dist * entropy_loss

    def _spatial_distribution_loss(self, adv_patch: torch.Tensor) -> torch.Tensor:
        """空间分布损失 - 鼓励补丁在不同区域都有有效特征"""
        c, h, w = adv_patch.shape
        
        # 将补丁分割为多个区域
        grid_size = 4  # 4x4网格
        region_h, region_w = h // grid_size, w // grid_size
        
        region_energies = []
        for i in range(grid_size):
            for j in range(grid_size):
                # 提取区域
                region = adv_patch[:, 
                                 i*region_h:(i+1)*region_h, 
                                 j*region_w:(j+1)*region_w]
                # 计算区域能量（L2范数）
                region_energy = torch.norm(region)
                region_energies.append(region_energy)
        
        # 计算区域能量的方差 - 惩罚能量分布不均
        region_energies_tensor = torch.stack(region_energies)
        variance = torch.var(region_energies_tensor)
        
        return self.lambda_dist * variance

    
    
    # def _gradient_variance_loss(self, adv_patch: torch.Tensor, det_loss: torch.Tensor) -> torch.Tensor:
    #     """基于梯度方差的分散化损失（可能不稳定）
        
    #     计算检测损失对补丁的梯度，然后计算梯度幅值的方差。
    #     高方差表示梯度集中在少数像素，低方差表示梯度分布均匀。
    #     """
    #     # 确保det_loss是标量
    #     if det_loss.dim() > 0:
    #         det_loss = det_loss.sum() if det_loss.numel() > 1 else det_loss
            
    #     # 确保adv_patch需要梯度
    #     if not adv_patch.requires_grad:
    #         adv_patch.requires_grad_(True)
            
    #     try:
    #         # 启用梯度保留以计算二阶导
    #         with torch.set_grad_enabled(True):
    #             # 计算det_loss对补丁的梯度
    #             grad = torch.autograd.grad(
    #                 det_loss, adv_patch, 
    #                 create_graph=True, 
    #                 retain_graph=True,
    #                 allow_unused=True
    #             )[0]
                
    #             # 如果梯度为None（可能由于detach操作），返回0损失
    #             if grad is None:
    #                 return torch.tensor(0.0, device=adv_patch.device)
                
    #             # grad形状: [C, H, W]
    #             # 计算梯度幅值图 (L2 norm for RGB)
    #             grad_magnitude = torch.sqrt(torch.sum(grad ** 2, dim=0))  # 形状: [H, W]
                
    #             # 计算梯度幅值的方差 - 惩罚高方差（集中）
    #             variance = torch.var(grad_magnitude)
                
    #             return self.lambda_dist * variance
    #     except Exception as e:
    #         # 如果梯度计算失败，回退到激活熵方法
    #         print(f"梯度方差损失计算失败，使用激活熵方法替代: {e}")
    #         return self._activation_entropy_loss(adv_patch)
    
    
    
    def _compute_entropy(self, tensor: torch.Tensor) -> torch.Tensor:
        """计算张量的熵"""
        # 展平并归一化
        flat_tensor = tensor.view(-1)
        flat_tensor = flat_tensor + 1e-8  # 避免log(0)
        probabilities = flat_tensor / torch.sum(flat_tensor)
        
        # 计算熵
        entropy = -torch.sum(probabilities * torch.log(probabilities))
        return entropy
    

