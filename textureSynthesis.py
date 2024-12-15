import numpy as np
import math
from skimage import io, util
from misc import L2OverlapDiff, minCutPatch


def randomPatch(texture, patchLength):
    """
    Extract a random square patch from the input texture.
    
    Args:
        texture (ndarray): Input texture image of shape (H, W, C)
        patchLength (int): Side length of the square patch to extract
        
    Returns:
        ndarray: Random patch of shape (patchLength, patchLength, C)
    """
    h, w, _ = texture.shape
    i = np.random.randint(h - patchLength)
    j = np.random.randint(w - patchLength)

    return texture[i:i+patchLength, j:j+patchLength]


def randomBestPatch(texture, patchLength, overlap, res, y, x):
    """
    Find the best matching patch from the texture by comparing overlapping regions.
    
    Args:
        texture (ndarray): Input texture image
        patchLength (int): Side length of patches
        overlap (int): Width of overlapping region
        res (ndarray): Current result image being synthesized
        y (int): Y-coordinate where patch will be placed
        x (int): X-coordinate where patch will be placed
        
    Returns:
        ndarray: Best matching patch of shape (patchLength, patchLength, C)
        
    Notes:
        Evaluates all possible patches and returns the one with minimum L2 difference
        in overlapping regions with the existing result.
    """
    h, w, _ = texture.shape
    errors = np.zeros((h - patchLength, w - patchLength))

    for i in range(h - patchLength):
        for j in range(w - patchLength):
            patch = texture[i:i+patchLength, j:j+patchLength]
            e = L2OverlapDiff(patch, patchLength, overlap, res, y, x)
            errors[i, j] = e

    i, j = np.unravel_index(np.argmin(errors), errors.shape)
    return texture[i:i+patchLength, j:j+patchLength]


def quilt(texture, patchLength, numPatches, overlap, mode="cut", sequence=False):
    """
    Synthesize a larger texture by quilting together patches from an input texture.
    
    Args:
        texture (ndarray): Input texture image
        patchLength (int): Side length of patches to use
        numPatches (tuple): Number of patches in (height, width)
        mode (str, optional): Quilting mode - "random", "best", or "cut". Defaults to "cut"
        sequence (bool, optional): If True, shows intermediate results. Defaults to False
        
    Returns:
        ndarray: Synthesized texture image
        
    Notes:
        Implements the Efros-Freeman image quilting algorithm with three modes:
       "random" mode:

Simply takes random patches from the input texture and places them side by side
No attempt is made to match the patches with their neighbors
Results in the most basic and usually lowest quality synthesis
Used for the first patch in all modes and throughout if "random" is selected


"best" mode:

Uses randomBestPatch() to find patches that match well with existing neighbors
For each new patch position, it:

Scans the entire input texture
Computes the L2 difference between the overlapping regions
Chooses the patch with the minimum error


Better than random but can still show visible seams between patches


"cut" mode (default):

Most sophisticated mode, implementing the full Efros-Freeman algorithm
First finds the best matching patch like "best" mode
Then uses minCutPatch() to compute an optimal boundary between the new patch and existing content
The optimal boundary is found using Dijkstra's algorithm to find the minimum cost path through the error surface
Results in the most seamless transitions between patches
    """
    texture = util.img_as_float(texture)

    numPatchesHigh, numPatchesWide = numPatches
    h = (numPatchesHigh * patchLength) - (numPatchesHigh - 1) * overlap
    w = (numPatchesWide * patchLength) - (numPatchesWide - 1) * overlap

    res = np.zeros((h, w, texture.shape[2]))

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
                patch = minCutPatch(patch, overlap, res, y, x)
            
            res[y:y+patchLength, x:x+patchLength] = patch

            if sequence:
                io.imshow(res)
                io.show()
      
    return res


def synthesize(texture, patchLength, overlap, shape, mode="cut"):
    """
    Synthesize a texture of specific dimensions using image quilting.
    
    Args:
        texture (ndarray): Input texture image
        patchLength (int): Side length of patches to use
        shape (tuple): Desired (height, width) of output image
        mode (str, optional): Quilting mode - "random", "best", or "cut". Defaults to "cut"
        
    Returns:
        ndarray: Synthesized texture of requested size
        
    Notes:
        Wrapper around quilt() that calculates the required number of patches to
        achieve the desired output dimensions, then crops the result to exact size.
    """
    h, w = shape

    numPatchesHigh = math.ceil((h - patchLength) / (patchLength - overlap)) + 1 or 1
    numPatchesWide = math.ceil((w - patchLength) / (patchLength - overlap)) + 1 or 1
    res = quilt(texture, patchLength, (numPatchesHigh, numPatchesWide), overlap, mode)

    return res[:h, :w]
