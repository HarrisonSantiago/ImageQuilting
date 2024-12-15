import numpy as np
import math
from skimage import io, util
from skimage.color import rgb2gray
from skimage.filters import gaussian
from misc import L2OverlapDiff, minCutPatch


def bestCorrPatch(texture, corrTexture, patchLength, corrTarget, y, x):
    """
    Find the best matching patch based on correlation with target image.
    
    Args:
        texture (ndarray): Source texture image of shape (H, W, C)
        corrTexture (ndarray): Grayscale version of source texture
        patchLength (int): Side length of patches to use
        corrTarget (ndarray): Grayscale version of target image
        y (int): Y-coordinate where patch will be placed
        x (int): X-coordinate where patch will be placed
        
    Returns:
        ndarray: Best matching patch from source texture
        
    Notes:
        Finds the patch in the source texture that best matches the corresponding
        region in the target image using L2 difference in grayscale space.
    """
    h, w, _ = texture.shape
    errors = np.zeros((h - patchLength, w - patchLength))

    corrTargetPatch = corrTarget[y:y+patchLength, x:x+patchLength]
    curPatchHeight, curPatchWidth = corrTargetPatch.shape

    for i in range(h - patchLength):
        for j in range(w - patchLength):
            corrTexturePatch = corrTexture[i:i+curPatchHeight, j:j+curPatchWidth]
            e = corrTexturePatch - corrTargetPatch
            errors[i, j] = np.sum(e**2)

    i, j = np.unravel_index(np.argmin(errors), errors.shape)
    return texture[i:i+curPatchHeight, j:j+curPatchWidth]


def bestCorrOverlapPatch(texture, corrTexture, patchLength, overlap, 
                         corrTarget, res, y, x, alpha=0.1, level=0):
    """
    Find the best matching patch considering both correlation and overlap.
    
    Args:
        texture (ndarray): Source texture image
        corrTexture (ndarray): Grayscale version of source texture
        patchLength (int): Side length of patches
        overlap (int): Width of overlapping region
        corrTarget (ndarray): Grayscale version of target image
        res (ndarray): Current result image being synthesized
        y (int): Y-coordinate where patch will be placed
        x (int): X-coordinate where patch will be placed
        alpha (float, optional): Weight between overlap and correlation errors. Defaults to 0.1
        level (int, optional): Current iteration level for hierarchical synthesis. Defaults to 0
        
    Returns:
        ndarray: Best matching patch considering both correlation and overlap constraints
        
    Notes:
        Combines three error terms:
        1. Overlap error with existing synthesis
        2. Correlation error with target image
        3. Previous level error (if level > 0)
        
        Alpha controls the balance between overlap/previous errors and correlation error.
    """
    h, w, _ = texture.shape
    errors = np.zeros((h - patchLength, w - patchLength))

    corrTargetPatch = corrTarget[y:y+patchLength, x:x+patchLength]
    di, dj = corrTargetPatch.shape

    for i in range(h - patchLength):
        for j in range(w - patchLength):
            patch = texture[i:i+di, j:j+dj]
            l2error = L2OverlapDiff(patch, patchLength, overlap, res, y, x)
            overlapError = np.sum(l2error)

            corrTexturePatch = corrTexture[i:i+di, j:j+dj]
            corrError = np.sum((corrTexturePatch - corrTargetPatch)**2)

            prevError = 0
            if level > 0:
                prevError = patch[overlap:, overlap:] - res[y+overlap:y+patchLength, x+overlap:x+patchLength]
                prevError = np.sum(prevError**2)

            errors[i, j] = alpha * (overlapError + prevError) + (1 - alpha) * corrError

    i, j = np.unravel_index(np.argmin(errors), errors.shape)
    return texture[i:i+di, j:j+dj]


def transfer(texture, target, patchLength, overlap, mode="cut", 
             alpha=0.1, level=0, prior=None, blur=False):
    """
    Perform texture transfer from source texture to target image.
    
    Args:
        texture (ndarray): Source texture image
        target (ndarray): Target image to guide synthesis
        patchLength (int): Side length of patches
        overlap
        mode (str, optional): Transfer mode - "best", "overlap", or "cut". Defaults to "cut"
        alpha (float, optional): Weight between overlap and correlation errors. Defaults to 0.1
        level (int, optional): Current iteration level for hierarchical synthesis. Defaults to 0
        prior (ndarray, optional): Result from previous iteration level. Defaults to None
        blur (bool, optional): Whether to blur correlation images. Defaults to False
        
    Returns:
        ndarray: Synthesized image combining texture appearance with target structure
        
    Notes:
        Implements texture transfer algorithm with three modes:
        - best: Only considers correlation with target
        - overlap: Considers both correlation and overlap
        - cut: Uses minimum cut optimization with correlation and overlap
        
        Can operate hierarchically when used with transferIter().
        Optional Gaussian blur can help with matching larger structures.
    """
    corrTexture = rgb2gray(texture)
    corrTarget = rgb2gray(target)

    if blur:
        corrTexture = gaussian(corrTexture, sigma=3)
        corrTarget = gaussian(corrTarget,  sigma=3)

    
    # remove alpha channel
    texture = util.img_as_float(texture)[:,:,:3]
    target = util.img_as_float(target)[:,:,:3]

    h, w, _ = target.shape

    numPatchesHigh = math.ceil((h - patchLength) / (patchLength - overlap)) + 1 or 1
    numPatchesWide = math.ceil((w - patchLength) / (patchLength - overlap)) + 1 or 1

    if level == 0:
        res = np.zeros_like(target)
    else:
        res = prior

    for i in range(numPatchesHigh):
        for j in range(numPatchesWide):
            y = i * (patchLength - overlap)
            x = j * (patchLength - overlap)

            if i == 0 and j == 0 or mode == "best":
                patch = bestCorrPatch(texture, corrTexture, patchLength, corrTarget, y, x)
            elif mode == "overlap":
                patch = bestCorrOverlapPatch(texture, corrTexture, patchLength, 
                                             overlap, corrTarget, res, y, x)
            elif mode == "cut":
                patch = bestCorrOverlapPatch(texture, corrTexture, patchLength, 
                                             overlap, corrTarget, res, y, x, 
                                             alpha, level)
                patch = minCutPatch(patch, overlap, res, y, x)

            res[y:y+patchLength, x:x+patchLength] = patch

    return res


def transfer_iter(texture, target, patchLength, n):
    """
    Perform hierarchical texture transfer with multiple iterations.
    
    Args:
        texture (ndarray): Source texture image
        target (ndarray): Target image to guide synthesis
        patchLength (int): Initial side length of patches
        n (int): Number of iterations
        
    Returns:
        ndarray: Final synthesized image after n iterations
        
    Notes:
        Implements coarse-to-fine texture transfer by:
        1. Starting with base synthesis
        2. Iteratively refining with:
           - Increasing alpha (more emphasis on structure)
           - Decreasing patch size (finer detail)
           
        Alpha increases linearly from 0.1 to 0.9
        Patch size decreases geometrically by factor of 2/3
    """
    overlap = patchLength // 6
    res = transfer(texture, target, patchLength, overlap)
    for i in range(1, n):
        alpha = 0.1 + 0.8 * i / (n - 1)
        patchLength = patchLength * 2**i // 3**i
        print((alpha, patchLength))
        overlap = patchLength // 6
        if overlap == 0:
            break
        res = transfer(texture, target, patchLength, overlap,
                       alpha=alpha, level=i, prior=res)

    return res