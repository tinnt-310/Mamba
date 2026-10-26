"""Spatial ordering inspired by VMamba cross_scan_fwd (see docs/SOURCE_REVIEW.md)."""
import torch


def scan_image(x: torch.Tensor, directions: int = 1) -> torch.Tensor:
    """[B,C,H,W] -> [B,K,C,H*W]; K is 1 (raster) or 4."""
    if x.ndim != 4 or min(x.shape) <= 0:
        raise ValueError("Expected a nonempty [B,C,H,W] tensor")
    if directions not in (1, 4):
        raise ValueError("directions must be 1 or 4")
    row = x.flatten(2)
    if directions == 1:
        return row.unsqueeze(1)
    col = x.transpose(2, 3).flatten(2)
    return torch.stack((row, col, row.flip(-1), col.flip(-1)), dim=1)


def merge_image(sequences: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Undo each scan and average aligned maps (VMamba uses a sum)."""
    if sequences.ndim != 4 or height <= 0 or width <= 0:
        raise ValueError("Expected [B,K,C,L] and positive spatial dimensions")
    b, k, c, length = sequences.shape
    if k not in (1, 4) or length != height * width:
        raise ValueError("Invalid direction count or sequence length")
    row = sequences[:, 0].reshape(b, c, height, width)
    if k == 1:
        return row
    col = sequences[:, 1].reshape(b, c, width, height).transpose(2, 3)
    reverse_row = sequences[:, 2].flip(-1).reshape(b, c, height, width)
    reverse_col = sequences[:, 3].flip(-1).reshape(b, c, width, height).transpose(2, 3)
    return (row + col + reverse_row + reverse_col) / 4
