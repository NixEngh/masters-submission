import torch
import torch.nn as nn
import torch.nn.functional as F
from cldice import soft_cldice_loss

class WeightedMSE(nn.Module):
    def __init__(self, non_zero_weight):
        super(WeightedMSE, self).__init__()
        self.non_zero_weight = non_zero_weight
    
    def forward(self, predictions, targets):
        squared_error = (predictions - targets) ** 2
        weight_mask = torch.where(targets != 0, self.non_zero_weight, 1.0)
        weighted_squared_error = squared_error * weight_mask
        weighted_mse = torch.mean(weighted_squared_error)
        return weighted_mse

class WeightedBCE(nn.Module):
    def __init__(self, pos_weight=1.0):
        super().__init__()
        self.pos_weight = pos_weight
        
    def forward(self, predictions, targets):
        weight = torch.tensor([self.pos_weight], device=predictions.device)
        return F.binary_cross_entropy_with_logits(predictions, targets, pos_weight=weight, reduction='mean')

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, predictions, targets):
        # predictions are logits; apply sigmoid
        predictions = torch.sigmoid(predictions)
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        intersection = (predictions * targets).sum()
        dice_coeff = (2. * intersection + self.smooth) / (predictions.sum() + targets.sum() + self.smooth)
        return 1 - dice_coeff

class SoftDiceLoss(nn.Module):
    def __init__(self, eps: float = 1e-6, treat_empty_as_perfect: bool = True):
        super().__init__()
        self.eps = eps
        self.treat_empty_as_perfect = treat_empty_as_perfect

    def forward(self, logits, targets):
        # logits: (B,1,H,W); targets: (B,1,H,W) in {0,1}
        probs = torch.sigmoid(logits)
        dims = (1,2,3)
        inter = (probs * targets).sum(dims)
        denom = probs.sum(dims) + targets.sum(dims)
        dice = (2*inter + self.eps) / (denom + self.eps)  # per-image

        if self.treat_empty_as_perfect:
            tgt_empty = (targets.sum(dims) == 0)
            pred_empty = (probs.sum(dims) <= self.eps)
            dice = torch.where(tgt_empty & pred_empty, torch.ones_like(dice), dice)

        return 1.0 - dice.mean()


class DiceBCELoss(nn.Module):
    """
    Weighted sum of Dice Loss and BCE-with-Logits.
    """
    def __init__(self, bce_pos_weight=1.0, bce_weight=0.5, dice_weight=0.5):
        super(DiceBCELoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce_loss = WeightedBCE(pos_weight=bce_pos_weight)
        self.dice_loss = SoftDiceLoss()

    def forward(self, predictions, targets):
        bce = self.bce_loss(predictions, targets)
        dice = self.dice_loss(predictions, targets)
        return self.bce_weight * bce + self.dice_weight * dice

class ClDiceLoss(nn.Module):
    """
    Soft clDice loss wrapper.
    Expects logits; applies sigmoid before soft skeletonization.
    Optionally accepts precomputed target skeleton (same shape) as target_skeleton kwarg.
    """
    def __init__(self, detach_target_skeleton=False):
        super().__init__()
        self.detach_target_skeleton = detach_target_skeleton

    def forward(self, logits, targets, target_skeleton=None):
        # 1. Clamp probabilities to a safe range to avoid log(0) or other issues inside cldice.
        probs = torch.sigmoid(logits).clamp(1e-6, 1.0 - 1e-6)
        
        tgt = targets
        if target_skeleton is not None and self.detach_target_skeleton:
            target_skeleton = target_skeleton.detach()

        raw = soft_cldice_loss(probs, tgt, target_skeleton=target_skeleton)
        
        # 2. CRITICAL: Replace any NaN or Inf that might come from the library with a zero.
        # This prevents the NaN from poisoning your entire model.
        loss = torch.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)

        # soft_cldice_loss currently returns -score in [-1, 0]; convert to loss in [0, 1]
        return (1.0 + loss).clamp(0.0, 1.0)

class WeightedDiceClDiceBCELoss(nn.Module):
    """
    Combined weighted loss:
      total = wd * Dice + wc * clDice + wb * BCE(logits)
    Set any weight to 0 to disable that component.
    Optionally supply pos_weight (scalar) for BCE.
    forward(..., target_skeleton=None)
    """
    def __init__(self, w_dice=0.6, w_cldice=0.2, w_bce=0.2, pos_weight=None,
                 cldice_detach_skeleton=False):
        super().__init__()
        self.wd, self.wc, self.wb = w_dice, w_cldice, w_bce
        self.weight_history = []

        self.dice = SoftDiceLoss()
        if pos_weight is None:
            self.bce_fn = nn.BCEWithLogitsLoss()
        else:
            self.register_buffer("pw", torch.tensor([pos_weight], dtype=torch.float32))
            self.bce_fn = nn.BCEWithLogitsLoss(pos_weight=self.pw)
        self.cldice = ClDiceLoss(detach_target_skeleton=cldice_detach_skeleton)
    
    def set_weights(self, w_dice=None, w_cldice=None, w_bce=None, normalize=True, epoch=None):
        wd = self.wd if w_dice is None else float(w_dice)
        wc = self.wc if w_cldice is None else float(w_cldice)
        wb = self.wb if w_bce   is None else float(w_bce)
        if normalize:
            s = max(wd + wc + wb, 1e-8)
            wd, wc, wb = wd/s, wc/s, wb/s
        self.wd, self.wc, self.wb = wd, wc, wb
        if epoch is not None:
            self.weight_history.append({"epoch": int(epoch), "w_dice": wd, "w_cldice": wc, "w_bce": wb})

    def get_weights(self):
        return {"w_dice": self.wd, "w_cldice": self.wc, "w_bce": self.wb}

    def forward(self, logits, targets, target_skeleton=None):
        total = torch.tensor(0.0, device=logits.device)
        comp = {
            "dice": torch.tensor(0.0, device=logits.device),
            "cldice": torch.tensor(0.0, device=logits.device),
            "bce": torch.tensor(0.0, device=logits.device),
        }

        if self.wd:
            d = self.dice(logits, targets)
            d = torch.nan_to_num(d, nan=0.0, posinf=1.0, neginf=1.0)
            total = total + self.wd * d
            comp["dice"] = d
        if self.wc:
            c = self.cldice(logits, targets, target_skeleton=target_skeleton)
            c = torch.nan_to_num(c, nan=0.0, posinf=1.0, neginf=1.0)
            total = total + self.wc * c
            comp["cldice"] = c
        if self.wb:
            b = self.bce_fn(logits, targets)
            b = torch.nan_to_num(b, nan=0.0, posinf=1.0, neginf=1.0)
            total = total + self.wb * b
            comp["bce"] = b

        total = torch.nan_to_num(total, nan=0.0, posinf=1.0, neginf=1.0)

        comp["total"] = total
        return total, comp



# ---------------- Metrics (use in validation) ----------------

@torch.no_grad()
def dice_coefficient(logits, targets, threshold=0.5, eps=1e-6):
    """
    Thresholded (hard) Dice coefficient.
    logits: (B,1,H,W) raw model outputs.
    targets: (B,1,H,W) binary {0,1}.
    """
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()
    dims = (1,2,3)
    inter = (preds * targets).sum(dims)
    denom = preds.sum(dims) + targets.sum(dims)
    dice = (2 * inter + eps) / (denom + eps)
    return dice.mean()

@torch.no_grad()
def soft_dice_coefficient(logits, targets, eps=1e-6):
    """
    Non-thresholded (probabilistic) Dice coefficient (1 - SoftDiceLoss without empty handling).
    """
    probs = torch.sigmoid(logits)
    dims = (1,2,3)
    inter = (probs * targets).sum(dims)
    denom = probs.sum(dims) + targets.sum(dims)
    dice = (2 * inter + eps) / (denom + eps)
    return dice.mean()

@torch.no_grad()
def soft_dice_per_sample(logits, targets, eps: float = 1e-6):
    """
    Per-sample probabilistic Dice (no threshold).
    Returns: tensor of shape [B].
    """
    probs = torch.sigmoid(logits)
    dims = (1, 2, 3)
    inter = (probs * targets).sum(dims)
    denom = probs.sum(dims) + targets.sum(dims)
    return (2 * inter + eps) / (denom + eps)

@torch.no_grad()
def dice_per_sample(logits, targets, threshold: float = 0.5, eps: float = 1e-6):
    """
    Per-sample hard Dice at a fixed threshold.
    Returns: tensor of shape [B].
    """
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()
    dims = (1, 2, 3)
    inter = (preds * targets).sum(dims)
    denom = preds.sum(dims) + targets.sum(dims)
    return (2 * inter + eps) / (denom + eps)


