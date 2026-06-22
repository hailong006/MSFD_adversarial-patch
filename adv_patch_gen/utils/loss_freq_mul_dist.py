"""Loss functions used in patch generation."""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F 


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
    """Frequency domain loss that suppresses high-frequency component generation"""
    
    def __init__(self, cutoff_ratio=0.3):
        super(FrequencyLoss, self).__init__()
        self.cutoff_ratio = cutoff_ratio

    def forward(self, adv_patch: torch.Tensor) -> torch.Tensor:
        """
        Args:
            adv_patch: Tensor of shape [C, H, W]
        """
        # Ensure processing 3D tensor (single or multi-channel)
        if adv_patch.dim() == 3:
            patch = adv_patch
        else:
            raise ValueError("Input tensor must be 3D [C, H, W]")
            
        # Fourier transform
        freq = torch.fft.fft2(patch)
        freq_shifted = torch.fft.fftshift(freq)
        
        # Create high-pass filter
        c, h, w = patch.shape[-3], patch.shape[-2], patch.shape[-1]
        cutoff = int(self.cutoff_ratio * min(h, w))
        mask = torch.ones_like(freq_shifted)
        center_h, center_w = h // 2, w // 2
        mask[..., 
             center_h - cutoff : center_h + cutoff, 
             center_w - cutoff : center_w + cutoff] = 0
        
        # Calculate high-frequency energy
        high_freq = freq_shifted * mask
        loss = torch.norm(high_freq)**2
        
        # Normalization
        return loss / (h * w * c)



class DistributionLoss(nn.Module):
    """Distribution loss function that encourages dispersion of effective information in the patch.
    
    Based on Moran's I spatial autocorrelation measurement, 
    by penalizing spatial clustering of patch pixel values, forces the optimization process to 
    uniformly distribute attack features across the entire patch area. This helps improve robustness 
    against local reflections (light spots) and occlusions.
    
    Moran's I range: [-1, 1]
    - Moran's I > 0: Positive spatial autocorrelation (pixel value clustering)
    - Moran's I < 0: Negative spatial autocorrelation (pixel value dispersion)
    - Moran's I ≈ 0: Random distribution
    
    Loss design objective: Minimize Moran's I to make patch pixel value distribution tend to be 
    dispersed or random.
    """
    
    def __init__(self, method='morans_i', lambda_dist=1.0, 
                 spatial_weights='queen', distance_decay=False):
        """
        Args:
            method: Dispersion measurement method. Options: 'morans_i', 'activation_entropy', 
                    'spatial_distribution', or 'combined'
            lambda_dist: Weight coefficient for distribution loss
            spatial_weights: Spatial weight matrix type. Options: 'queen' (4-neighborhood + diagonal) 
                            or 'rook' (4-neighborhood only)
            distance_decay: Whether to use distance-decay weights (inverse distance squared)
        """
        super(DistributionLoss, self).__init__()
        self.method = method
        self.lambda_dist = lambda_dist
        self.spatial_weights = spatial_weights
        self.distance_decay = distance_decay
        
        # Precomputed spatial weight matrix (lazy initialization)
        self.register_buffer('weights_matrix', None)
        self.last_size = None
        
    def _create_spatial_weights(self, height: int, width: int, 
                                device: torch.device) -> torch.Tensor:
        """Create spatial weight matrix
        
        Args:
            height: Patch height
            width: Patch width
            device: Computing device
            
        Returns:
            weights_matrix: Spatial weight matrix with shape [N, N], where N = height * width
        """
        n_pixels = height * width
        weights = torch.zeros(n_pixels, n_pixels, device=device)
        
        # Assign index for each pixel
        idx_map = torch.arange(n_pixels).view(height, width)
        
        for i in range(height):
            for j in range(width):
                current_idx = idx_map[i, j]
                
                # Define neighborhood range
                if self.spatial_weights == 'queen':
                    # 4-neighborhood + diagonal neighbors (8 directions total)
                    neighbors = [
                        (i-1, j-1), (i-1, j), (i-1, j+1),
                        (i, j-1),              (i, j+1),
                        (i+1, j-1), (i+1, j), (i+1, j+1)
                    ]
                else:  # rook
                    # 4-neighborhood only
                    neighbors = [
                        (i-1, j), (i, j-1), (i, j+1), (i+1, j)
                    ]
                
                for ni, nj in neighbors:
                    if 0 <= ni < height and 0 <= nj < width:
                        neighbor_idx = idx_map[ni, nj]
                        
                        if self.distance_decay:
                            # Inverse distance squared weights
                            distance = torch.sqrt(torch.tensor(
                                (i - ni)**2 + (j - nj)**2, dtype=torch.float32
                            ))
                            weights[current_idx, neighbor_idx] = 1.0 / (distance**2 + 1e-8)
                        else:
                            # Binary weights
                            weights[current_idx, neighbor_idx] = 1.0
        
        return weights
    
    def _compute_morans_i(self, adv_patch: torch.Tensor) -> torch.Tensor:
        """Compute Global Moran's I
        
        Formula:
        I = (N / W) * [ΣiΣj wij(xi - x̄)(xj - x̄) / Σi(xi - x̄)²]
        
        Where:
        - N: Total number of pixels
        - W: Sum of spatial weights
        - wij: Spatial weight between pixels i and j
        - xi, xj: Values of pixels i and j
        - x̄: Mean value
        
        Args:
            adv_patch: Adversarial patch tensor with shape [C, H, W]
            
        Returns:
            morans_i: Moran's I value
        """
        c, h, w = adv_patch.shape
        n_pixels = h * w
        device = adv_patch.device
        
        # Check if weight matrix needs to be recomputed
        if self.weights_matrix is None or self.last_size != (h, w):
            self.weights_matrix = self._create_spatial_weights(h, w, device)
            self.last_size = (h, w)
        
        weights = self.weights_matrix
        W = torch.sum(weights) + 1e-8
        
        # Compute Moran's I for each channel and average
        morans_i_list = []
        for channel in range(c):
            # Extract single channel and flatten
            channel_data = adv_patch[channel, :, :].view(n_pixels)
            
            # Compute mean
            mean_val = torch.mean(channel_data)
            
            # Compute deviations (xi - x̄)
            deviations = channel_data - mean_val  # [N]
            
            # Compute numerator: ΣiΣj wij(xi - x̄)(xj - x̄)
            numerator = deviations @ weights @ deviations  # scalar
            
            # Compute denominator: Σi(xi - x̄)²
            denominator = torch.sum(deviations ** 2) + 1e-8
            
            # Compute Moran's I
            if denominator > 0:
                morans_i = (n_pixels / W) * (numerator / denominator)
            else:
                morans_i = torch.tensor(0.0, device=device)
            
            morans_i_list.append(morans_i)
        
        # Average across channels
        avg_morans_i = torch.mean(torch.stack(morans_i_list))
        return avg_morans_i
    
    def _morans_i_loss(self, adv_patch: torch.Tensor) -> torch.Tensor:
        """Dispersion loss based on Moran's I
        
        By minimizing Moran's I (making it approach 0 or negative), encourages 
        patch pixel values to be distributed dispersively, improving robustness 
        against local occlusions.
        
        Args:
            adv_patch: Adversarial patch tensor with shape [C, H, W]
            
        Returns:
            loss: Dispersion loss value
        """
        morans_i = self._compute_morans_i(adv_patch)
        
        # Loss design: Penalize positive Moran's I when Moran's I > 0 (clustering)
        positive_part = torch.max(morans_i, torch.tensor(0.0, device=adv_patch.device))
        loss = positive_part
        
        return self.lambda_dist * loss

    def forward(self, adv_patch: torch.Tensor, det_loss: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            adv_patch: Adversarial patch tensor with shape [C, H, W]
            det_loss: Detection loss (optional, for gradient method)
        """
        if self.method == 'morans_i':
            return self._morans_i_loss(adv_patch)
        elif self.method == 'activation_entropy':
            return self._activation_entropy_loss(adv_patch)
        elif self.method == 'spatial_distribution':
            return self._spatial_distribution_loss(adv_patch)
        elif self.method == 'combined':
            # Combine Moran's I and entropy loss
            morans_loss = self._morans_i_loss(adv_patch)
            entropy_loss = self._activation_entropy_loss(adv_patch)
            return self.lambda_dist * (0.5 * morans_loss + 0.5 * entropy_loss)
        else:
            raise ValueError(f"Unsupported method: {self.method}")

    def _activation_entropy_loss(self, adv_patch: torch.Tensor) -> torch.Tensor:
        
        """Dispersion loss based on activation entropy
        
        Computes the entropy of patch pixel values to encourage more uniform distribution.
        Higher entropy indicates more dispersed information.
        """
        # Flatten patch into probability distribution
        patch_flat = adv_patch.view(-1)
        
        # Add small value to avoid log(0)
        patch_flat = patch_flat + 1e-8
        
        # Normalize to probability distribution
        probabilities = patch_flat / torch.sum(patch_flat)
        
        # Compute entropy
        entropy = -torch.sum(probabilities * torch.log(probabilities))
        
        # Maximize entropy = minimize negative entropy
        entropy_loss = -entropy
        
        return self.lambda_dist * entropy_loss

    def _spatial_distribution_loss(self, adv_patch: torch.Tensor) -> torch.Tensor:
        """Spatial distribution loss - encourages effective features in different regions of the patch"""
        c, h, w = adv_patch.shape
        
        # Split patch into multiple regions
        grid_size = 4  # 4x4 grid
        region_h, region_w = h // grid_size, w // grid_size
        
        region_energies = []
        for i in range(grid_size):
            for j in range(grid_size):
                # Extract region
                region = adv_patch[:, 
                                 i*region_h:(i+1)*region_h, 
                                 j*region_w:(j+1)*region_w]
                # Compute region energy (L2 norm)
                region_energy = torch.norm(region)
                region_energies.append(region_energy)
        
        # Compute variance of region energies - penalize uneven energy distribution
        region_energies_tensor = torch.stack(region_energies)
        variance = torch.var(region_energies_tensor)
        
        return self.lambda_dist * variance

    
   
    
    
    def _compute_entropy(self, tensor: torch.Tensor) -> torch.Tensor:
        """Compute entropy of tensor"""
        # Flatten and normalize
        flat_tensor = tensor.view(-1)
        flat_tensor = flat_tensor + 1e-8  # Avoid log(0)
        probabilities = flat_tensor / torch.sum(flat_tensor)
        
        # Compute entropy
        entropy = -torch.sum(probabilities * torch.log(probabilities))
        return entropy
    
    def get_morans_i(self, adv_patch: torch.Tensor) -> torch.Tensor:
        """Get Moran's I value of current patch (for analysis)"""
        return self._compute_morans_i(adv_patch)


class MultiScaleEntropyConsistencyLoss(nn.Module):
    """
    Multi-Scale Entropy Consistency Loss with Moran's I (MSEC-Moran Loss)
       
    Moran's I range: [-1, 1]
    - Moran's I > 0: Positive spatial autocorrelation (pixel value clustering) → needs penalty
    - Moran's I < 0: Negative spatial autocorrelation (pixel value dispersion) → ideal state
    - Moran's I ≈ 0: Random distribution → ideal state
    """
    
    def __init__(self, scale_factors=[0.9, 0.7, 0.5], alpha=0.5, beta=1.0, gamma=1.0,
                 spatial_weights='queen', distance_decay=False):
        """
        Args:
            scale_factors: List of downsampling scale factors
            alpha: Weight for entropy maximization term
            beta: Weight for multi-scale entropy difference minimization term
            gamma: Weight for Moran's I spatial dispersion constraint
            spatial_weights: Spatial weight matrix type. Options: 'queen' (4-neighborhood + diagonal) 
                            or 'rook' (4-neighborhood only)
            distance_decay: Whether to use distance-decay weights (inverse distance squared)
        """
        super(MultiScaleEntropyConsistencyLoss, self).__init__()
        self.scale_factors = scale_factors
        self.alpha = alpha  # Weight for entropy maximization term
        self.beta = beta    # Weight for multi-scale entropy consistency term
        self.gamma = gamma  # Weight for Moran's I constraint
        self.spatial_weights = spatial_weights
        self.distance_decay = distance_decay
        
        # Precomputed spatial weight matrix cache (keyed by size)
        self.weights_cache = {}

    def _create_spatial_weights(self, height: int, width: int, 
                                device: torch.device) -> torch.Tensor:
        """Create spatial weight matrix
        
        Args:
            height: Patch height
            width: Patch width
            device: Computing device
            
        Returns:
            weights_matrix: Spatial weight matrix with shape [N, N], where N = height * width
        """
        # Check cache
        cache_key = (height, width)
        if cache_key in self.weights_cache:
            return self.weights_cache[cache_key].to(device)
        
        n_pixels = height * width
        weights = torch.zeros(n_pixels, n_pixels)
        
        # Assign index for each pixel
        idx_map = torch.arange(n_pixels).view(height, width)
        
        for i in range(height):
            for j in range(width):
                current_idx = idx_map[i, j]
                
                # Define neighborhood range
                if self.spatial_weights == 'queen':
                    # 4-neighborhood + diagonal neighbors (8 directions total)
                    neighbors = [
                        (i-1, j-1), (i-1, j), (i-1, j+1),
                        (i, j-1),              (i, j+1),
                        (i+1, j-1), (i+1, j), (i+1, j+1)
                    ]
                else:  # rook
                    # 4-neighborhood only
                    neighbors = [
                        (i-1, j), (i, j-1), (i, j+1), (i+1, j)
                    ]
                
                for ni, nj in neighbors:
                    if 0 <= ni < height and 0 <= nj < width:
                        neighbor_idx = idx_map[ni, nj]
                        
                        if self.distance_decay:
                            # Inverse distance squared weights
                            distance = torch.sqrt(torch.tensor(
                                (i - ni)**2 + (j - nj)**2, dtype=torch.float32
                            ))
                            weights[current_idx, neighbor_idx] = 1.0 / (distance**2 + 1e-8)
                        else:
                            # Binary weights
                            weights[current_idx, neighbor_idx] = 1.0
        
        # Cache weight matrix
        self.weights_cache[cache_key] = weights
        return weights.to(device)
    
    def _compute_morans_i(self, adv_patch: torch.Tensor) -> torch.Tensor:
        """Compute Global Moran's I
        
        Formula:
        I = (N / W) * [ΣiΣj wij(xi - x̄)(xj - x̄) / Σi(xi - x̄)²]
        
        Where:
        - N: Total number of pixels
        - W: Sum of spatial weights
        - wij: Spatial weight between pixels i and j
        - xi, xj: Values of pixels i and j
        - x̄: Mean value
        
        Args:
            adv_patch: Adversarial patch tensor with shape [C, H, W]
            
        Returns:
            morans_i: Moran's I value
        """
        c, h, w = adv_patch.shape
        n_pixels = h * w
        device = adv_patch.device
        
        # Get spatial weight matrix
        weights = self._create_spatial_weights(h, w, device)
        W = torch.sum(weights) + 1e-8
        
        # Compute Moran's I for each channel and average
        morans_i_list = []
        for channel in range(c):
            # Extract single channel and flatten
            channel_data = adv_patch[channel, :, :].view(n_pixels)
            
            # Compute mean
            mean_val = torch.mean(channel_data)
            
            # Compute deviations (xi - x̄)
            deviations = channel_data - mean_val  # [N]
            
            # Compute numerator: ΣiΣj wij(xi - x̄)(xj - x̄)
            numerator = deviations @ weights @ deviations  # scalar
            
            # Compute denominator: Σi(xi - x̄)²
            denominator = torch.sum(deviations ** 2) + 1e-8
            
            # Compute Moran's I
            if denominator > 0:
                morans_i = (n_pixels / W) * (numerator / denominator)
            else:
                morans_i = torch.tensor(0.0, device=device)
            
            morans_i_list.append(morans_i)
        
        # Average across channels
        avg_morans_i = torch.mean(torch.stack(morans_i_list))
        return avg_morans_i

    def _compute_entropy(self, x: torch.Tensor) -> torch.Tensor:
        """Compute entropy of tensor (consistent with DistributionLoss)"""
        # Flatten entire tensor to 1D
        flat_x = x.view(-1)
        # Add small value to avoid log(0)
        flat_x = flat_x + 1e-8
        # Directly normalize to probability distribution (without softmax)
        probabilities = flat_x / torch.sum(flat_x)
        # Compute standard information entropy
        entropy = -torch.sum(probabilities * torch.log(probabilities))
        return entropy

    def forward(self, adv_patch: torch.Tensor) -> torch.Tensor:
        """
        Args:
            adv_patch: Adversarial patch tensor with shape [C, H, W]
            
        Returns:
            total_loss: Total loss = alpha*entropy_loss + beta*multi_scale_consistency_loss + gamma*morans_loss
        """
        entropy_loss = 0
        divergence_loss = 0
        morans_loss = 0

        # Entropy at original scale
        entropy_P0 = self._compute_entropy(adv_patch)

        # Entropy maximization term: maximize entropy = minimize negative entropy
        entropy_loss = -entropy_P0

        # Multi-scale processing
        entropies = [entropy_P0]
        morans_i_values = []
        
        # Compute Moran's I at original scale
        morans_i_0 = self._compute_morans_i(adv_patch)
        morans_i_values.append(morans_i_0)
        
        for scale_factor in self.scale_factors:
            # Downsample patch
            downsampled_patch = F.interpolate(adv_patch.unsqueeze(0),
                                            scale_factor=scale_factor,
                                            mode='bilinear',
                                            align_corners=False).squeeze(0)
            # Compute entropy after downsampling
            entropy_s = self._compute_entropy(downsampled_patch)
            entropies.append(entropy_s)

            # Difference term: minimize squared difference between original and downsampled entropy
            divergence_loss += F.mse_loss(entropy_P0, entropy_s)
            
            # Compute Moran's I at downsampled scale
            morans_i_s = self._compute_morans_i(downsampled_patch)
            morans_i_values.append(morans_i_s)

        # Average difference across all scales
        divergence_loss /= len(self.scale_factors)
        
        # Moran's I loss:
        # 1. Penalize positive Moran's I (clustering)
        # 2. Encourage Moran's I consistency across scales (scale invariance)
        morans_i_tensor = torch.stack(morans_i_values)
        
        # Penalize positive Moran's I (clustering)
        morans_positive_penalty = torch.mean(torch.clamp(morans_i_tensor, min=0))
        
        # Multi-scale Moran's I consistency: minimize variance of Moran's I across scales
        morans_consistency_loss = torch.var(morans_i_tensor)
        
        # Combine Moran's I loss
        morans_loss = morans_positive_penalty + morans_consistency_loss

        # Combine three loss terms
        total_loss = self.alpha * entropy_loss + self.beta * divergence_loss + self.gamma * morans_loss
        return total_loss
    
    def get_metrics(self, adv_patch: torch.Tensor) -> dict:
        """Get various metrics (for analysis)
        
        Returns:
            metrics: Dictionary containing entropy, Moran's I and other metrics
        """
        metrics = {}
        
        # Entropy at original scale
        entropy_P0 = self._compute_entropy(adv_patch)
        metrics['entropy'] = entropy_P0.item()
        
        # Moran's I at original scale
        morans_i_0 = self._compute_morans_i(adv_patch)
        metrics['morans_i'] = morans_i_0.item()
        
        # Multi-scale entropy
        entropies = [entropy_P0]
        for scale_factor in self.scale_factors:
            downsampled_patch = F.interpolate(adv_patch.unsqueeze(0),
                                            scale_factor=scale_factor,
                                            mode='bilinear',
                                            align_corners=False).squeeze(0)
            entropy_s = self._compute_entropy(downsampled_patch)
            entropies.append(entropy_s.item())
        metrics['multiscale_entropies'] = entropies
        
        # Multi-scale Moran's I
        morans_i_values = [morans_i_0.item()]
        for scale_factor in self.scale_factors:
            downsampled_patch = F.interpolate(adv_patch.unsqueeze(0),
                                            scale_factor=scale_factor,
                                            mode='bilinear',
                                            align_corners=False).squeeze(0)
            morans_i_s = self._compute_morans_i(downsampled_patch)
            morans_i_values.append(morans_i_s.item())
        metrics['multiscale_morans_i'] = morans_i_values
        
        return metrics

