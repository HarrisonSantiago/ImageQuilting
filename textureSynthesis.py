import torch
import math
from utils import L2OverlapDiff, minCutPatch


def randomPatch(texture, patchLength):
    """
    Extract a random square patch from the input texture.
    
    Args:
        texture (Tensor): Input texture image of shape (H, W, C)
        patchLength (int): Side length of the square patch to extract
        device (str, optional): Device to place tensors on
        
    Returns:
        Tensor: Random patch of shape (patchLength, patchLength, C)
    """
    h, w, _ = texture.shape
    i = torch.randint(h - patchLength, (1,), device=texture.device).item()
    j = torch.randint(w - patchLength, (1,), device=texture.device).item()

    return texture[i:i+patchLength, j:j+patchLength]


def randomBestPatch(texture, patchLength, overlap, res, y, x):
    """
    Find the best matching patch from the texture by comparing overlapping
    regions. Evaluates all possible patches and returns the one with minimum
    L2 difference in overlapping regions with the existing result.
    
    Args:
        texture (Tensor): Input texture image
        patchLength (int): Side length of patches
        overlap (int): Width of overlapping region
        res (Tensor): Current result image being synthesized
        y (int): Y-coordinate where patch will be placed
        x (int): X-coordinate where patch will be placed
        
    Returns:
        Tensor: Best matching patch of shape (patchLength, patchLength, C)
    """
    h, w, _ = texture.shape
    errors = torch.zeros((h - patchLength, w - patchLength), device=texture.device)

    for i in range(h - patchLength):
        for j in range(w - patchLength):
            patch = texture[i:i+patchLength, j:j+patchLength]
            e = L2OverlapDiff(patch, patchLength, overlap, res, y, x)
            errors[i, j] = e

    # Get indices of minimum error
    min_idx = torch.argmin(errors)
    i, j = min_idx // errors.shape[1], min_idx % errors.shape[1]
    return texture[i:i+patchLength, j:j+patchLength]


def quilt(texture, patchLength, numPatches, overlap, mode="cut", algorithm='dijkstra', device=None):
    """
    Synthesize a larger texture by quilting together patches from an input texture.
    
    Args:
        texture (Tensor): Input texture image
        patchLength (int): Side length of patches to use
        numPatches (tuple): Number of patches in (height, width)
        overlap (int): Width of overlapping region
        mode (str, optional): Quilting mode - "random", "best", or "cut". Defaults to "cut"
        algorithm
        sequence (bool, optional): If True, shows intermediate results. Defaults to False
        device (str, optional): Device to place tensors on
        
    Returns:
        Tensor: Synthesized texture image

    Notes:
        Implements the Efros-Freeman image quilting algorithm with three modes:
        
        random: Takes random patches from the input texture and places
                them side by side. No attempt is made to match the patches
                with their neighbors.
    
        best: Uses randomBestPatch() to find patches that match well with
              existing neighbors. For each new patch position, it
                -Scans the entire input texture
                -Computes the L2 difference between the overlapping regions
                -Chooses the patch with the minimum error

        cut: Implements the full Efros-Freeman algorithm. First finds the
             best matching patch like "best" mode. Then uses minCutPatch()
             to compute an optimal boundary between the new patch and
             existing content. The optimal boundary is found using
             Dijkstra's algorithm to find the minimum cost path through the
             error surface
    """
    if device is None:
        device = texture.device
    
    # Ensure texture is float and normalized
    texture = texture.float()
    if texture.max() > 1.0:
        texture = texture / 255.0

    numPatchesHigh, numPatchesWide = numPatches
    h = (numPatchesHigh * patchLength) - (numPatchesHigh - 1) * overlap
    w = (numPatchesWide * patchLength) - (numPatchesWide - 1) * overlap

    res = torch.zeros((h, w, texture.shape[2]), device=device)

    for i in range(numPatchesHigh):
        for j in range(numPatchesWide):
            y = i * (patchLength - overlap)
            x = j * (patchLength - overlap)

            if i == 0 and j == 0 or mode == "random":
                patch = randomPatch(texture, patchLength)
            elif mode == "best":
                patch = randomBestPatch(texture, patchLength, overlap, res, y, x)
            elif mode == "cut":
                patch = randomBestPatch(texture, patchLength, overlap, res, y, x)
                patch = minCutPatch(patch, overlap, res, y, x, algorithm)
            
            res[y:y+patchLength, x:x+patchLength] = patch
      
    return res


def synthesize(texture, patchLength, overlap, shape, mode="cut", algorithm='dijkstra', device=None):
    """
    Synthesize a texture of specific dimensions using image quilting.
    
    Args:
        texture (Tensor): Input texture image
        patchLength (int): Side length of patches to use
        overlap (int): Width of overlapping region
        shape (tuple): Desired (height, width) of output image
        mode (str, optional): Quilting mode - "random", "best", or "cut". Defaults to "cut"
        algorithm
        device (str, optional): Device to place tensors on
        
    Returns:
        Tensor: Synthesized texture of requested size
    
    Notes:
        Wrapper around quilt() that calculates the required number of patches to
        achieve the desired output dimensions, then crops the result to exact size.
    """
    if device is None:
        device = texture.device
        
    h, w = shape

    numPatchesHigh = math.ceil((h - patchLength) / (patchLength - overlap)) + 1 or 1
    numPatchesWide = math.ceil((w - patchLength) / (patchLength - overlap)) + 1 or 1
    res = quilt(texture,
                patchLength,
                (numPatchesHigh, numPatchesWide),
                overlap,
                mode,
                algorithm,
                device=device)

    return res[:h, :w]