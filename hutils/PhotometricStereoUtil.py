import numpy as np
from sklearn import preprocessing

def evalsurfaceNormal(N_gt, N_est, mask):
    """
    N_gt and N_est are both [H, W, 3] surface normal
    mask: [H, W] bool matrix to indicate forward mask
    """

    gt = N_gt[mask, :]
    est = N_est[mask, :]
    ae = evaluate_angular_error(gt, est)
    error_map = np.zeros([N_gt.shape[0], N_gt.shape[1]], dtype=np.float)
    error_map[mask] = ae
    return [error_map, np.mean(ae), np.median(ae)]


def evaldepth(z_gt, z_est, mask):
    """
    N_gt and N_est are both [H, W, 3] surface normal
    mask: [H, W] bool matrix to indicate forward mask
    """

    gt = z_gt[mask,]
    est = z_est[mask]
    absE = np.abs(gt - est)
    error_map = np.zeros_like(z_gt)
    error_map[mask] = absE
    return [error_map, np.mean(absE), np.median(absE)]


def evaluate_angular_error(gtnormal=None, normal=None, background=None):
    """
    gtnormal: [N, 3]
    normal: estimated normal [N, 3]
    background: bool index with length N
    return angular error with size N
    """
    if gtnormal is None or normal is None:
        raise ValueError("surface normal is not given")
    if background is not None:
        gtnormal[background] = 0
        normal[background] = 0
    gtnormal = preprocessing.normalize(gtnormal)
    normal = preprocessing.normalize(normal)
    ae = np.multiply(gtnormal, normal)
    aesum = np.sum(ae, axis=1)
    coord = np.where(aesum > 1.0)
    aesum[coord] = 1.0
    coord = np.where(aesum < -1.0)
    aesum[coord] = -1.0
    ae = np.arccos(aesum) * 180.0 / np.pi
    if background is not None:
        ae[background] = 0
    return ae


def fillHole(mask_valid, mask_full, input_with_hole, require_normalize=False):
    """
    Fill the hole in the input data
    Parameters
    ----------
    mask_valid: The mask where input with value
    mask: The full mask we want to recover [M, N]
    input_with_hole: The input with the hole [M, N, C]

    Returns input with hole filled
    -------

    """

    assert mask_full is not None
    assert input_with_hole is not None
    from scipy.interpolate import griddata
    height, width, channel = input_with_hole.shape
    coll, roww = np.meshgrid(np.arange(width), np.arange(height))
    valid_positions = np.dstack([roww[mask_valid], coll[mask_valid]]).squeeze()

    input_full = np.zeros_like(input_with_hole)
    for i in range(channel):
        value_c = griddata(valid_positions, input_with_hole[mask_valid, i], (roww, coll), method='nearest')
        input_full[:,:,i] = value_c
        input_full[~mask_full, i] = 0

    if require_normalize:
        mask_valid = np.linalg.norm(input_full, axis=2) > 1e-6
        input_full = input_full / np.linalg.norm(input_full, axis=2, keepdims=True)

    return input_full, mask_valid

