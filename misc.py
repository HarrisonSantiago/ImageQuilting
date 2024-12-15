import numpy as np
import heapq


def L2OverlapDiff(patch, patchLength, overlap, res, y, x):
    """
    Calculate the L2 difference between a patch and existing overlapping
    regions.
    
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


def minCutPath(errors):
    """
    Find the minimum cost path through an error matrix using
    Dijkstra's algorithm.
    
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
    Find the minimum cost path through an error matrix using
    dynamic programming.
    
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