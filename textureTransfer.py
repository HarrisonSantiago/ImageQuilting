import torch
import math
from torchvision.transforms.functional import rgb_to_grayscale, gaussian_blur
from utils import L2OverlapDiff, minCutPatch
from typing import Optional


def bestCorrPatch(texture: torch.Tensor,
                  corrTexture: torch.Tensor,
                  patchLength: int,
                  corrTarget: torch.Tensor,
                  y: int,
                  x: int):
    """
    Find the best matching patch based on correlation with target image.

    Args:
        texture (Tensor): Source texture image of shape (H, W, C)
        corrTexture (Tensor): Grayscale version of source texture
        patchLength (int): Side length of patches to use
        corrTarget (Tensor): Grayscale version of target image
        y (int): Y-coordinate where patch will be placed
        x (int): X-coordinate where patch will be placed

    Returns:
        Tensor: Best matching patch from source texture
    """
    h, w, _ = texture.shape
    errors = torch.zeros((h - patchLength, w - patchLength), device=texture.device)

    corrTargetPatch = corrTarget[y:y+patchLength, x:x+patchLength]
    curPatchHeight, curPatchWidth = corrTargetPatch.shape

    for i in range(h - patchLength):
        for j in range(w - patchLength):
            corrTexturePatch = corrTexture[i:i+curPatchHeight, j:j+curPatchWidth]
            e = corrTexturePatch - corrTargetPatch
            errors[i, j] = torch.sum(e**2)

    # Get indices of minimum error
    min_idx = torch.argmin(errors)
    i, j = min_idx // errors.shape[1], min_idx % errors.shape[1]
    return texture[i:i+curPatchHeight, j:j+curPatchWidth]


def bestCorrOverlapPatch(texture: torch.Tensor,
                         corrTexture: torch.Tensor,
                         patchLength: int,
                         overlap: int,
                         corrTarget: torch.Tensor,
                         res: torch.Tensor,
                         y: int,
                         x: int,
                         alpha: float = 0.1,
                         level: int = 0) -> torch.Tensor:
    """
    Find the best matching patch considering both correlation and overlap.

    Args:
        texture (Tensor): Source texture image
        corrTexture (Tensor): Grayscale version of source texture
        patchLength (int): Side length of patches
        overlap (int): Width of overlapping region
        corrTarget (Tensor): Grayscale version of target image
        res (Tensor): Current result image being synthesized
        y (int): Y-coordinate where patch will be placed
        x (int): X-coordinate where patch will be placed
        alpha (float, optional): Weight between overlap and correlation errors.
        level (int, optional): Current iteration level for hierarchical
                                synthesis.

    Returns:
        Tensor: Best matching patch considering both correlation and
                overlap constraints
    """
    h, w, _ = texture.shape
    errors = torch.zeros((h - patchLength, w - patchLength),
                         device=texture.device)

    corrTargetPatch = corrTarget[y:y+patchLength, x:x+patchLength]
    di, dj = corrTargetPatch.shape

    for i in range(h - patchLength):
        for j in range(w - patchLength):
            patch = texture[i:i+di, j:j+dj]
            l2error = L2OverlapDiff(patch, patchLength, overlap, res, y, x)
            overlapError = torch.sum(l2error)

            corrTexturePatch = corrTexture[i:i+di, j:j+dj]
            corrError = torch.sum((corrTexturePatch - corrTargetPatch)**2)

            prevError = torch.tensor(0.0, device=texture.device)
            if level > 0:
                prevError = patch[overlap:, overlap:] - res[y+overlap:y+patchLength, x+overlap:x+patchLength]
                prevError = torch.sum(prevError**2)

            errors[i, j] = alpha * \
                           (overlapError + prevError) + \
                           (1 - alpha) * \
                           corrError

    # Get indices of minimum error
    min_idx = torch.argmin(errors)
    i, j = min_idx // errors.shape[1], min_idx % errors.shape[1]
    return texture[i:i+di, j:j+dj]


def transfer(texture: torch.Tensor,
             target: torch.Tensor,
             patchLength: int,
             overlap: int,
             mode: str = "cut",
             algorithm: str = "dijkstra",
             alpha: float = 0.1,
             level: int = 0,
             prior: Optional[torch.Tensor] = None,
             blur: Optional[bool] = False,
             device: Optional[str] = None) -> torch.Tensor:
    """
    Perform texture transfer from source texture to target image.

    Args:
        texture (Tensor): Source texture image
        target (Tensor): Target image to guide synthesis
        patchLength (int): Side length of patches
        overlap (int): Width of overlapping region
        mode (str, optional): Transfer mode - "best", "overlap", or "cut".
        alpha (float, optional): Weight between overlap and correlation errors.
        level (int, optional): Current iteration level for hierarchical
                                synthesis.
        prior (Tensor, optional): Result from previous iteration level.
        blur (bool, optional): Whether to blur correlation images.
        device (str, optional): Device to place tensors on

    Returns:
        Tensor: Synthesized image combining texture appearance with
                target structure
    """
    if device is None:
        device = texture.device

    # Convert to grayscale for correlation
    corrTexture = rgb_to_grayscale(texture.permute(2, 0, 1)).squeeze(0)
    corrTarget = rgb_to_grayscale(target.permute(2, 0, 1)).squeeze(0)

    if blur:
        corrTexture = gaussian_blur(corrTexture.unsqueeze(0), kernel_size=7, sigma=3).squeeze(0)
        corrTarget = gaussian_blur(corrTarget.unsqueeze(0), kernel_size=7, sigma=3).squeeze(0)

    h, w, _ = target.shape

    numPatchesHigh = math.ceil((h - patchLength) / (patchLength - overlap)) + 1 or 1
    numPatchesWide = math.ceil((w - patchLength) / (patchLength - overlap)) + 1 or 1

    if level == 0:
        res = torch.zeros_like(target, device=device)
    else:
        res = prior

    for i in range(numPatchesHigh):
        for j in range(numPatchesWide):
            y = i * (patchLength - overlap)
            x = j * (patchLength - overlap)

            if i == 0 and j == 0 or mode == "best":
                patch = bestCorrPatch(texture,
                                      corrTexture,
                                      patchLength,
                                      corrTarget,
                                      y,
                                      x)
            elif mode == "overlap":
                patch = bestCorrOverlapPatch(texture,
                                             corrTexture,
                                             patchLength,
                                             overlap,
                                             corrTarget,
                                             res,
                                             y,
                                             x)
            elif mode == "cut":
                patch = bestCorrOverlapPatch(texture,
                                             corrTexture,
                                             patchLength,
                                             overlap,
                                             corrTarget,
                                             res,
                                             y,
                                             x,
                                             alpha,
                                             level)
                patch = minCutPatch(patch,
                                    overlap,
                                    res,
                                    y,
                                    x,
                                    algorithm)

            res[y:y+patchLength, x:x+patchLength] = patch

    return res


def transfer_iter(texture: torch.Tensor,
                  target: torch.Tensor,
                  patchLength: int,
                  n: int,
                  device: Optional[str] = None) -> torch.Tensor:
    """
    Perform hierarchical texture transfer with multiple iterations.

    Args:
        texture (Tensor): Source texture image
        target (Tensor): Target image to guide synthesis
        patchLength (int): Initial side length of patches
        n (int): Number of iterations
        device (str, optional): Device to place tensors on

    Returns:
        Tensor: Final synthesized image after n iterations
    """
    if device is None:
        device = texture.device

    overlap = patchLength // 6
    res = transfer(texture, target, patchLength, overlap, device=device)

    for i in range(1, n):
        alpha = 0.1 + 0.8 * i / (n - 1)
        patchLength = patchLength * 2**i // 3**i
        print(f"Iteration {i}: alpha={alpha:.2f}, patchLength={patchLength}")

        overlap = patchLength // 6
        if overlap == 0:
            break

        res = transfer(texture,
                       target,
                       patchLength,
                       overlap,
                       alpha=alpha,
                       level=i,
                       prior=res,
                       device=device)

    return res
