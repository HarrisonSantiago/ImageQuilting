import torch
import heapq


def L2OverlapDiff(patch, patchLength, overlap, res, y, x):
    """
    Calculate the L2 difference between a patch and existing overlapping
    regions using PyTorch tensors.

    Args:
        patch (Tensor): Candidate patch to compare
        patchLength (int): Side length of the patch
        overlap (int): Width of the overlapping region
        res (Tensor): Current result image being synthesized
        y (int): Y-coordinate where patch would be placed
        x (int): X-coordinate where patch would be placed

    Returns:
        float: Sum of squared differences in overlapping regions

    Notes:
        Computes error in left and top overlapping regions, correcting for
        double-counted corner region when both overlaps exist.
    """
    error = torch.tensor(0.0, device=patch.device)

    if x > 0:
        left = patch[:, :overlap] - res[y:y+patchLength, x:x+overlap]
        error += torch.sum(left**2)

    if y > 0:
        up = patch[:overlap, :] - res[y:y+overlap, x:x+patchLength]
        error += torch.sum(up**2)

    if x > 0 and y > 0:
        corner = patch[:overlap, :overlap] - res[y:y+overlap, x:x+overlap]
        error -= torch.sum(corner**2)

    return error


def minCutPath(errors, algorithm='dijkstra'):

    if algorithm == 'dijkstra':
        return minCutPath_dijkstra(errors)
    elif algorithm == 'dynamic':
        return minCutPath_dynamic(errors)
    else:
        raise ValueError('algorithm must be one of [dijkstra, dynamic]')


def minCutPath_dijkstra(errors):
    """
    Find the minimum cost path through an error matrix using
    Dijkstra's algorithm.

    Args:
        errors (Tensor): 2D tensor of error values

    Returns:
        list: Sequence of indices defining the minimum cut path

    Notes:
        Converts tensor to CPU for heap operations, then processes using
        Dijkstra's algorithm.
    """
    # Convert to CPU numpy for heap operations
    errors_cpu = errors.cpu().numpy()

    pq = [(float(error), [i]) for i, error in enumerate(errors_cpu[0])]
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
                    cumError = error + float(errors_cpu[curDepth, nextIndex])
                    heapq.heappush(pq, (cumError, path + [nextIndex]))
                    seen.add((curDepth, nextIndex))


def minCutPath_dynamic(errors):
    """
    Find the minimum cost path through an error matrix using
    dynamic programming with PyTorch operations.

    Args:
        errors (Tensor): 2D tensor of error values of shape (H, W)

    Returns:
        iterator: Sequence of indices defining the minimum cut path
        from top to bottom
    """
    # Pad the error tensor
    errors = torch.nn.functional.pad(
        errors,
        (1, 1, 0, 0),
        mode='constant',
        value=float('inf')
    )

    cumError = errors[0].clone()
    paths = torch.zeros_like(errors, dtype=torch.long, device=errors.device)

    for i in range(1, len(errors)):
        M = cumError
        L = torch.roll(M, 1)  # shifted right, giving left neighbor
        R = torch.roll(M, -1)  # shifted left, giving right neighbor

        # Stack for min operation
        stacked = torch.stack((L, M, R))

        # Find minimum values and indices
        cumError = torch.min(stacked, dim=0).values + errors[i]
        paths[i] = torch.argmin(stacked, dim=0)

    paths = paths - 1  # Convert to -1, 0, 1 for left/straight/right moves

    # Reconstruct path from bottom up
    minCutPath = [torch.argmin(cumError).item()]
    for i in reversed(range(1, len(errors))):
        minCutPath.append(minCutPath[-1] + paths[i][minCutPath[-1]].item())

    # Return iterator of path indices, corrected for padding
    return map(lambda x: x - 1, reversed(minCutPath))


def minCutPatch(patch, overlap, res, y, x, algorithm):
    """
    Apply minimum cut optimization to blend a patch with existing content
    using PyTorch operations.

    Args:
        patch (Tensor): Patch to be blended, shape (H, W, C)
        overlap (int): Width of overlapping region
        res (Tensor): Current result image being synthesized
        y (int): Y-coordinate where patch will be placed
        x (int): X-coordinate where patch will be placed
        algorithm

    Returns:
        Tensor: Modified patch with minimum cut applied in overlap regions
    """
    patch = patch.clone()
    dy, dx, _ = patch.shape
    minCut = torch.zeros_like(patch, dtype=torch.bool, device=patch.device)

    if x > 0:
        left = patch[:, :overlap] - res[y:y+dy, x:x+overlap]
        leftL2 = torch.sum(left**2, dim=2)
        for i, j in enumerate(minCutPath(leftL2, algorithm)):
            minCut[i, :j] = True

    if y > 0:
        up = patch[:overlap, :] - res[y:y+overlap, x:x+dx]
        upL2 = torch.sum(up**2, dim=2)
        for j, i in enumerate(minCutPath(upL2.T, algorithm)):
            minCut[:i, j] = True

    patch = torch.where(minCut, res[y:y+dy, x:x+dx], patch)

    return patch
