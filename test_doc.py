import numpy as np
import math
from skimage import io, util
import heapq


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


def L2OverlapDiff(patch, patchLength, overlap, res, y, x):
    """
    Calculate the L2 difference between a patch and existing overlapping regions.
    
    Args:
        patch (ndarray): Candidate patch to compare
        patchLength (int): Side length of the patch
        overlap (int): Width of the overlapping region
        res (ndarray): Current result image being synthesized
        y (int): Y-coordinate where patch would be placed
        x (int): X-coordinate where patch would be placed
        
    Returns:
        float: Sum of squared differences in overlapping regions
    
    Notes:
        Computes error in left and top overlapping regions, correcting for 
        double-counted corner region when both overlaps exist.
    """
    error = 0

    if x > 0:
        left = patch[:, :overlap] - res[y:y+patchLength, x:x+overlap]
        error += np.sum(left**2)

    if y > 0:
        up = patch[:overlap, :] - res[y:y+overlap, x:x+patchLength]
        error += np.sum(up**2)

    if x > 0 and y > 0:
        corner = patch[:overlap, :overlap] - res[y:y+overlap, x:x+overlap]
        error -= np.sum(corner**2)

    return error
 

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


def minCutPath(errors):
    """
    Find the minimum cost path through an error matrix using Dijkstra's algorithm.
    
    Args:
        errors (ndarray): 2D array of error values
        
    Returns:
        list: Sequence of indices defining the minimum cut path
        
    Notes:
        Implements Dijkstra's algorithm to find the minimum cost vertical path
        through the error matrix, allowing diagonal movements of ±1 pixel.
    """
    pq = [(error, [i]) for i, error in enumerate(errors[0])]
    heapq.heapify(pq)

    h, w = errors.shape
    seen = set()

    while pq:
        error, path = heapq.heappop(pq)
        curDepth = len(path)
        curIndex = path[-1]

        if curDepth == h:
            return path

        for delta in -1, 0, 1:
            nextIndex = curIndex + delta

            if 0 <= nextIndex < w:
                if (curDepth, nextIndex) not in seen:
                    cumError = error + errors[curDepth, nextIndex]
                    heapq.heappush(pq, (cumError, path + [nextIndex]))
                    seen.add((curDepth, nextIndex))


def minCutPath2(errors):
    """
    Find the minimum cost path through an error matrix using dynamic programming.
    
    Args:
        errors (ndarray): 2D array of error values of shape (H, W)
    
    Returns:
        iterator: Sequence of indices defining the minimum cut path from top to bottom
    
    Notes:
        Implements a dynamic programming solution to find the minimum cost vertical
        path through the error matrix. For each pixel, considers three possible
        moves (diagonal left, straight down, diagonal right) to find optimal path.
        
        Algorithm steps:
        1. Pads the error matrix with infinity values to handle edge cases
        2. Builds cumulative error matrix from top to bottom
        3. Tracks optimal moves using a paths matrix
        4. Reconstructs optimal path from bottom to top
        
        Compared to Dijkstra-based minCutPath():
        + Faster due to vectorized NumPy operations
        + More memory efficient for grid graphs
        - Less flexible for modified constraints
        - Must process entire grid
        
        Time complexity: O(H * W) where H, W are height and width of error matrix
        Space complexity: O(H * W) for paths matrix
    """
    errors = np.pad(errors, [(0, 0), (1, 1)], 
                    mode='constant', 
                    constant_values=np.inf)

    cumError = errors[0].copy()
    paths = np.zeros_like(errors, dtype=int)    

    for i in range(1, len(errors)):
        M = cumError
        L = np.roll(M, 1)  # shifted right, giving left neighbor
        R = np.roll(M, -1) # shifted left, giving right neighbor

        # For each position, find minimum of left diagonal, straight down, right diagonal
        cumError = np.min((L, M, R), axis=0) + errors[i]
        paths[i] = np.argmin((L, M, R), axis=0)  # 0=left, 1=straight, 2=right
    
    paths -= 1  # Convert to -1, 0, 1 for left/straight/right moves
    
    # Reconstruct path from bottom up
    minCutPath = [np.argmin(cumError)]  # Start from minimum value in bottom row
    for i in reversed(range(1, len(errors))):
        minCutPath.append(minCutPath[-1] + paths[i][minCutPath[-1]])
    
    # Return iterator of path indices, corrected for padding
    return map(lambda x: x - 1, reversed(minCutPath))


def minCutPatch(patch, overlap, res, y, x):
    """
    Apply minimum cut optimization to blend a patch with existing content.
    
    Args:
        patch (ndarray): Patch to be blended
        overlap (int): Width of overlapping region
        res (ndarray): Current result image being synthesized
        y (int): Y-coordinate where patch will be placed
        x (int): X-coordinate where patch will be placed
        
    Returns:
        ndarray: Modified patch with minimum cut applied in overlap regions
        
    Notes:
        Computes and applies minimum cost paths through overlap regions to create
        seamless transitions between patches.
    """
    patch = patch.copy()
    dy, dx, _ = patch.shape
    minCut = np.zeros_like(patch, dtype=bool)

    if x > 0:
        left = patch[:, :overlap] - res[y:y+dy, x:x+overlap]
        leftL2 = np.sum(left**2, axis=2)
        for i, j in enumerate(minCutPath(leftL2)):
            minCut[i, :j] = True

    if y > 0:
        up = patch[:overlap, :] - res[y:y+overlap, x:x+dx]
        upL2 = np.sum(up**2, axis=2)
        for j, i in enumerate(minCutPath(upL2.T)):
            minCut[:i, j] = True

    np.copyto(patch, res[y:y+dy, x:x+dx], where=minCut)

    return patch


def quilt(texture, patchLength, numPatches, mode="cut", sequence=False):
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

    overlap = patchLength // 6
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


def synthesize(texture, patchLength, shape, mode="cut"):
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
    overlap = patchLength // 6
    h, w = shape

    numPatchesHigh = math.ceil((h - patchLength) / (patchLength - overlap)) + 1 or 1
    numPatchesWide = math.ceil((w - patchLength) / (patchLength - overlap)) + 1 or 1
    res = quilt(texture, patchLength, (numPatchesHigh, numPatchesWide), mode)

    return res[:h, :w]

texture = io.imread("wallpaper.jpeg")
io.imshow(texture)
io.show()

io.imshow(quilt(texture, 25, (6, 6), "random"))
io.show()

#io.imshow(quilt(texture, 5, (6, 6), "best"))
#io.show()

io.imshow(quilt(texture, 10, (25, 25), "cut"))
io.show()