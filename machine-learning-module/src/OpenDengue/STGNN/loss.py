import torch
import torch.nn.functional as F


def split_masks(y: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(mask passed to the model, mask for the loss) for one batch.

    One-month models (y: (B, N, 1)) keep the original behaviour: the model and the loss both
    use the batch's (B, N) target mask. Multi-month models (y, mask: (B, N, H)) score every
    observed lead, and the model treats a node as observed if any lead is.
    """
    if y.dim() == 3 and y.size(-1) > 1:
        return mask.any(dim=-1), mask
    return mask, (mask[:, -1, :] if mask.dim() == 3 else mask)


def masked_huber_loss(
    pred:   torch.Tensor,
    target: torch.Tensor,
    mask:   torch.Tensor,
    delta:  float = 1.0,
) -> torch.Tensor:
    """
    Huber loss computed only over masked (valid surveillance) nodes across a batch.

    Args:
        pred:   (B, N) or (B, N, 1) — model predictions.
        target: (B, N) or (B, N, 1) — ground truth incidence.
        mask:   (B, N)              — bool/float, True/1 = valid node.
        delta:  Huber threshold.

    Returns:
        Scalar loss averaged across valid elements.
    """
    # ── 1. Safe Dimension Alignment ──────────────────────────────────────────
    # Avoid general .squeeze(-1) which can destroy a batch dimension of size 1.
    if pred.dim() == 3 and pred.size(-1) == 1:
        pred = pred.squeeze(-1)
    if target.dim() == 3 and target.size(-1) == 1:
        target = target.squeeze(-1)
        
    mask = mask.bool()

    # ── 2. Differentiable Fallback ───────────────────────────────────────────
    if mask.sum() == 0:
        # Multiply a prediction element by 0.0 to keep the tensor attached 
        # to the autograd graph, preventing broken gradients.
        return pred.sum() * 0.0

    # ── 3. Masked Extracted Statistics ───────────────────────────────────────
    # Flat continuous indexing maps safely regardless of batch configurations
    return F.huber_loss(pred[mask], target[mask], delta=delta, reduction="mean")