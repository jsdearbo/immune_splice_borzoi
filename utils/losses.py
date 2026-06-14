import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for class imbalance (or sparse events).
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    This dynamically down-weights easy examples (where p_t is high),
    focusing training on hard examples (rare events) without the
    risk of overfitting caused by massive static weights.
    """

    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets, mask=None, confidence=None):
        # inputs are logits. convert to probs
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        pt = torch.exp(-bce_loss)  # pt is the probability of the true class

        focal_loss = ((1 - pt) ** self.gamma) * bce_loss

        if self.alpha is not None:
            # If alpha is 0.25 (for class 1), then (1-alpha) is for class 0
            # weight = alpha if target=1 else (1-alpha)
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_loss = alpha_t * focal_loss

        # Apply mask if provided
        if mask is not None:
            # Mask shape (B, 1, L) or (B, T, L)
            focal_loss = focal_loss * mask

        if self.reduction == "mean":
            if mask is not None:
                return focal_loss.sum() / (mask.sum() + 1e-8)
            return focal_loss.mean()
        if self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class WeightedPSILoss(nn.Module):
    """
    Custom Loss for Splicing PSI (Percent Spliced In).

    Standard pos_weight in BCEWithLogitsLoss distorts probabilities for regression tasks
    (i.e., it shifts the optimal prediction for target=0.5 towards 1.0).

    This loss uses SAMPLE weights (element-wise), which preserves probability calibration
    while forcing the model to prioritize specific value ranges (like rare AS events).
    """

    def __init__(self, zero_weight=1.0, one_weight=10.0, intermediate_weight=100.0, epsilon=1e-4):
        super(WeightedPSILoss, self).__init__()
        self.zero_weight = zero_weight
        self.one_weight = one_weight
        self.intermediate_weight = intermediate_weight
        self.epsilon = epsilon

    def forward(self, logits, targets, mask=None, confidence=None):
        # Calculate base BCE Loss (element-wise)
        loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        # Calculate Weights based on Targets
        # Targets are in [0, 1]

        # Intermediate: (0 + eps) < target < (1 - eps)
        is_intermediate = (targets > self.epsilon) & (targets < (1.0 - self.epsilon))

        # Ones: target >= (1 - eps)
        is_one = targets >= (1.0 - self.epsilon)

        # Zeros: target <= eps (Default case, weight=1.0 usually)
        # We start with base weight
        weights = torch.full_like(loss, self.zero_weight)

        # Apply overrides
        weights[is_one] = self.one_weight
        weights[is_intermediate] = self.intermediate_weight

        # Apply Mask if provided
        if mask is not None:
            weights = weights * mask

        # Weighted Loss
        weighted_loss = loss * weights

        # For mean reduction, normalize by the sum of weights (or mask area)
        # Normalizing by sum of weights keeps the scale consistent regardless of sparsity
        return weighted_loss.sum() / (weights.sum() + 1e-8)


class PSILoss(nn.Module):
    """
    BCEWithLogitsLoss for PSI-style soft targets in [0,1],
    with smooth emphasis on intermediate PSI values.

    Supports:
    - multitask outputs (3 cell types)
    - bin-level masks
    """

    def __init__(self, lambda_mid=25.0, eps=1e-6, use_mask=True):
        super().__init__()
        self.lambda_mid = lambda_mid
        self.eps = eps
        self.use_mask = use_mask

    def forward(self, logits, targets, mask=None, confidence=None):
        """
        logits:  [B, L, T]  (raw model outputs)
        targets: [B, L, T]  (PSI labels in [0,1])
        mask:    [B, L, T]  (1=valid, 0=ignore)
        confidence: [B, L, T] (optional weights)
        """

        # elementwise BCE
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        # smooth mid-PSI emphasis
        mid_factor = 4.0 * targets * (1.0 - targets)
        weights = 1.0 + self.lambda_mid * mid_factor

        # Scale by external confidence if provided
        # We add 1.0 to confidence to ensure background (conf=0) retains a weight of 1.0.
        # Otherwise, 0-confidence regions would have 0 loss and the model would hallucinate events there.
        if confidence is not None:
            weights = weights * (1.0 + confidence)

        loss = weights * bce

        # apply supervision mask
        if self.use_mask and mask is not None:
            loss = loss * mask
            denom = mask.sum().clamp_min(self.eps)
        else:
            denom = loss.numel()

        return loss.sum() / denom


class BhattacharyyaLoss(nn.Module):
    """
    Bhattacharyya distance loss for comparing predicted and target signal profiles.
    Treats the signal across the genomic window as a probability distribution and
    penalizes divergence in distributional shape.

    This captures global profile structure (e.g., a retained intron creating a
    broad plateau) that pointwise BCE can miss.
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # Convert logits to PSI probabilities [0, 1]
        probs = torch.sigmoid(logits)

        # Normalize over the length dimension to get distributional profiles.
        # If a mask is present, normalize only over unmasked bins so that
        # different mask patterns don't distort the comparison.
        if mask is not None:
            # Zero out masked positions, then normalize over remaining mass
            p = probs * mask
            q = targets * mask
            p_sum = p.sum(dim=-1, keepdim=True).clamp(min=self.eps)
            q_sum = q.sum(dim=-1, keepdim=True).clamp(min=self.eps)
            p = p / p_sum
            q = q / q_sum
            # Re-zero masked positions after normalization so they contribute
            # nothing to the Bhattacharyya coefficient
            p = p * mask
            q = q * mask
        else:
            p_sum = probs.sum(dim=-1, keepdim=True).clamp(min=self.eps)
            q_sum = targets.sum(dim=-1, keepdim=True).clamp(min=self.eps)
            p = probs / p_sum
            q = targets / q_sum

        # Bhattacharyya coefficient: BC = sum(sqrt(p * q))
        # Use clamp(min=) instead of add(eps) to avoid inflating BC above 1.0
        # when thousands of near-zero bins each contribute sqrt(eps).
        bc = (p * q).clamp(min=1e-12).sqrt().sum(dim=-1)

        # Distance = -log(BC). Perfect overlap -> BC=1 -> loss=0
        # Clamp BC to [eps, 1.0] so loss is always non-negative.
        loss = -torch.log(bc.clamp(min=self.eps, max=1.0))
        return loss.mean()


class CompoundPSILoss(nn.Module):
    """
    Hybrid Loss: Pointwise BCE (with mid-PSI emphasis) + Profile Shape Penalty.

    Combines the existing PSILoss logic (elementwise BCE with smooth mid-PSI
    weighting and confidence scaling) with a Bhattacharyya distributional
    shape penalty that models the overall signal profile across the window.

    Args:
        lambda_mid: Weight multiplier for intermediate PSI values (~0.5).
        alpha_shape: Weight for the Bhattacharyya shape loss component.
        eps: Numerical stability epsilon.
        use_mask: Whether to apply the coverage mask.
    """

    def __init__(self, lambda_mid=25.0, alpha_shape=0.1, eps=1e-6, use_mask=True):
        super().__init__()
        self.lambda_mid = lambda_mid
        self.alpha_shape = alpha_shape
        self.eps = eps
        self.use_mask = use_mask
        self.shape_fn = BhattacharyyaLoss(eps=eps)

    def forward(self, logits, targets, mask=None, confidence=None):
        # 1. Pointwise BCE
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        # 2. Smooth mid-PSI emphasis: 4*t*(1-t) peaks at 1.0 when t=0.5
        mid_factor = 4.0 * targets * (1.0 - targets)
        weights = 1.0 + self.lambda_mid * mid_factor

        # Scale by external confidence (add 1.0 so conf=0 retains weight of 1.0)
        if confidence is not None:
            weights = weights * (1.0 + confidence)

        loss = weights * bce

        # Apply supervision mask
        if self.use_mask and mask is not None:
            loss = loss * mask
            denom = mask.sum().clamp_min(self.eps)
        else:
            denom = loss.numel()

        pointwise_loss = loss.sum() / denom

        # 3. Shape penalty (Bhattacharyya)
        shape_mask = mask if (self.use_mask and mask is not None) else None
        shape_loss = self.shape_fn(logits, targets, mask=shape_mask)

        return pointwise_loss + self.alpha_shape * shape_loss


class TaskDiversityLoss(nn.Module):
    """
    Encourages predictions for different tasks/cell types to be DIFFERENT.
    Prevents output collapse by penalizing high correlation between task outputs.

    This mitigates the collapse seen in PyTorch where all cell types predict similar values.
    The TensorFlow version uses strand_pair mapping for this; we use explicit regularization.

    Implementation:
    - Computes pairwise cosine similarity between task output channels
    - Penalizes off-diagonal similarities (we want tasks to be different)
    - Loss = mean(|cosine_similarity[i,j]| for i!=j)
    """

    def __init__(self, weight=0.01, use_mean_centering=True):
        super(TaskDiversityLoss, self).__init__()
        self.weight = weight
        self.use_mean_centering = use_mean_centering

    def forward(self, logits, reduction="mean", dim_tasks=1):
        """
        Args:
            logits: (B, n_tasks, L) - model outputs for each task
            reduction: 'mean' or 'sum'
            dim_tasks: which dimension contains task outputs (usually 1)
        Returns:
            scalar loss value (already weighted)
        """
        B, n_tasks, L = logits.shape

        # Handle case where n_tasks is 1 (no diversity to enforce)
        if n_tasks <= 1:
            return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

        # Flatten spatial dimension: (B, n_tasks, L) -> (B*L, n_tasks)
        logits_flat = logits.reshape(B * L, n_tasks)

        # Optionally center each task representation (mean-subtract)
        if self.use_mean_centering:
            task_means = logits_flat.mean(dim=0, keepdim=True)
            logits_centered = logits_flat - task_means
        else:
            logits_centered = logits_flat

        # Normalize each task's predictions (across samples)
        # Shape: (B*L, n_tasks)
        logits_norm = F.normalize(logits_centered, p=2, dim=0)  # Normalize within each task

        # Compute pairwise cosine similarity: (n_tasks, n_tasks)
        # sim[i,j] = cos_sim between task_i and task_j across all positions
        similarity = torch.abs(torch.mm(logits_norm.t(), logits_norm))  # (n_tasks, n_tasks)

        # Zero out diagonal (we don't care about self-similarity)
        mask_off_diag = ~torch.eye(n_tasks, dtype=torch.bool, device=logits.device)
        off_diag_sim = similarity[mask_off_diag]

        # Penalize off-diagonal similarities
        if reduction == "mean":
            loss = off_diag_sim.mean()
        else:
            loss = off_diag_sim.sum()

        return loss * self.weight

# latest attempt loss: combines pointwise BCE with mid-PSI emphasis, cell-type specialization penalty, and profile shape penalty.
# compatible with Logit Huber as an alternative to BCE
class CellTypeBhattacharyyaLoss(nn.Module):
    """
    Bhattacharyya distance across cell types at each genomic bin.
    Forces the model to learn task specialization (prevents output collapse).
    """
    def __init__(self, eps: float = 1e-8, signal_threshold: float = 0.01):
        super().__init__()
        self.eps = eps
        self.signal_threshold = signal_threshold

    def forward(self, logits, targets, mask=None):
        # Shape expectations: [B, T, L]
        # We need T (tasks) to be dimension 1.
        if logits.shape[1] == 1:
            # Cannot compute cell-type distribution with only 1 output channel
            return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

        probs = torch.sigmoid(logits)

        # Create a boolean signal mask where the total PSI across cell types is above the threshold
        # This prevents amplifying noise in completely flat background regions
        signal_mask = (targets.sum(dim=1, keepdim=True) > self.signal_threshold).float()

        if mask is not None:
            # Combine the geographical confidence mask with the intensity signal mask
            combined_mask = mask * signal_mask
            probs = probs * combined_mask
            targets = targets * combined_mask
        else:
            combined_mask = signal_mask
            probs = probs * combined_mask
            targets = targets * combined_mask

        # normalize across tasks (dim=1)
        p_sum = probs.sum(dim=1, keepdim=True).clamp(min=self.eps)
        q_sum = targets.sum(dim=1, keepdim=True).clamp(min=self.eps)

        p = probs / p_sum
        q = targets / q_sum

        # Bhattacharyya coefficient along task dimension
        bc = torch.sqrt((p * q).clamp(min=self.eps)).sum(dim=1)
        dist = -torch.log(bc.clamp(min=self.eps))

        # If any task in a bin is valid under our combined mask, keep the bin
        valid_bins = combined_mask.any(dim=1).float() 
        denom = valid_bins.sum().clamp(min=1.0)
        return (dist * valid_bins).sum() / denom

class StrongPSILoss(nn.Module):
    """
    Robust compound loss for multitask PSI prediction.
    Returns: total_loss, pointwise_loss, cell_loss, profile_loss (for detailed logging)
    """
    def __init__(self, lambda_mid=25.0, alpha_cell=0.05, beta_profile=0.02, gamma_pointwise=1.0, eps=1e-6, use_mask=True, pointwise_mode="logit", cell_signal_threshold=0.01):
        super().__init__()
        self.lambda_mid = lambda_mid
        self.alpha_cell = alpha_cell
        self.beta_profile = beta_profile
        self.gamma_pointwise = gamma_pointwise
        self.eps = eps
        self.use_mask = use_mask
        self.pointwise_mode = pointwise_mode
        self.cell_loss_fn = CellTypeBhattacharyyaLoss(eps=eps, signal_threshold=cell_signal_threshold)

    def profile_bhattacharyya(self, probs, targets, mask):
        if mask is not None:
            p = probs * mask
            q = targets * mask
            p_sum = p.sum(dim=-1, keepdim=True).clamp(min=self.eps)
            q_sum = q.sum(dim=-1, keepdim=True).clamp(min=self.eps)
            p = p / p_sum
            q = q / q_sum
            p = p * mask
            q = q * mask
        else:
            p_sum = probs.sum(dim=-1, keepdim=True).clamp(min=self.eps)
            q_sum = targets.sum(dim=-1, keepdim=True).clamp(min=self.eps)
            p = probs / p_sum
            q = targets / q_sum

        bc = (p * q).clamp(min=1e-12).sqrt().sum(dim=-1)
        dist = -torch.log(bc.clamp(min=self.eps, max=1.0))
        return dist.mean()

    def forward(self, logits, targets, mask=None, confidence=None):
        probs = torch.sigmoid(logits)

        # ==========================================
        # MASK DECOUPLING LOGIC
        # ==========================================
        broad_mask = mask if self.use_mask else None
        
        narrow_mask = None
        if confidence is not None:
            # Create a binary mask strictly where rMATS events exist
            # Shape: [B, T, L]
            base_narrow = (confidence > 0).float()
            
            # Dilate the narrow mask by 2 bins on each side (kernel=5, padding=2)
            # This allows the Profile loss to see the splice site transitions!
            # MaxPool1d expects [Batch, Channels, Length] which perfectly matches [B, T, L]
            dilated_narrow = F.max_pool1d(base_narrow, kernel_size=5, stride=1, padding=2)
            
            # Intersect with the broad mask (so we don't accidentally dilate into invalid/N-masked bins)
            if broad_mask is not None:
                narrow_mask = dilated_narrow * broad_mask
            else:
                narrow_mask = dilated_narrow
        else:
            # Fallback if confidence isn't loaded
            narrow_mask = broad_mask

        # ==========================================
        # 1. Pointwise Loss (Uses BROAD Mask)
        # ==========================================
        if self.pointwise_mode == "logit":
            targets_clamped = targets.clamp(self.eps, 1.0 - self.eps)
            target_logits = torch.log(targets_clamped / (1.0 - targets_clamped))
            pointwise_base = F.smooth_l1_loss(logits, target_logits, reduction="none")
        else:
            pointwise_base = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        mid_factor = 4.0 * targets * (1.0 - targets)
        weights = 1.0 + self.lambda_mid * mid_factor

        if confidence is not None:
            weights = weights * (1.0 + confidence)

        loss = weights * pointwise_base

        if broad_mask is not None:
            loss = loss * broad_mask
            denom = broad_mask.sum().clamp_min(self.eps)
        else:
            denom = loss.numel()

        pointwise_loss = loss.sum() / denom

        # ==========================================
        # 2. Shape / Cell Losses (Uses NARROW DILATED Mask)
        # ==========================================
        cell_loss = self.cell_loss_fn(logits, targets, narrow_mask) if self.alpha_cell > 0 else torch.tensor(0.0, device=logits.device)
        profile_loss = self.profile_bhattacharyya(probs, targets, narrow_mask) if self.beta_profile > 0 else torch.tensor(0.0, device=logits.device)

        total_loss = (self.gamma_pointwise * pointwise_loss) + (self.alpha_cell * cell_loss) + (self.beta_profile * profile_loss)
        return total_loss, pointwise_loss, cell_loss, profile_loss