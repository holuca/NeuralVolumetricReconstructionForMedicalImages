import cv2
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import torch
from scipy.ndimage import rotate
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast
import numpy as np
import torch
from scipy.ndimage import rotate
from torch.utils.checkpoint import checkpoint
from src.loss import compute_tv_regularization

def compute_tv_regularization(pts):
    """
    Compute total variation (TV) regularization on points sampled along rays.

    Args:
        pts: [num_rays, n_samples, 3] Points sampled along the rays.

    Returns:
        tv_loss: Total variation loss.
    """
    diff = pts[:, 1:, :] - pts[:, :-1, :]  # Differences between consecutive points
    tv_loss = torch.sum(torch.abs(diff))  # Sum of L1 norms
    return tv_loss


def render(rays, net, net_fine, n_samples, n_fine, perturb, netchunk, raw_noise_std, chunk_size=None):

    """
    Perform rendering with optional chunking for memory efficiency.

    Args:
        rays: Input rays [num_rays, 8].
        net: Coarse network.
        net_fine: Fine network (if any).
        n_samples: Number of coarse samples.
        n_fine: Number of fine samples.
        perturb: If True, applies perturbation to sampling.
        netchunk: Maximum size for network chunks.
        raw_noise_std: Standard deviation of noise to add to raw predictions.
        chunk_size: Optional. Number of rays to process in one chunk.
    Returns:
        Dictionary containing the accumulated results.
    """
    
    n_rays = rays.shape[0]
    if chunk_size is None or chunk_size >= n_rays:
        # Process all rays at once if no chunking is needed
        return render_chunk(rays, net, net_fine, n_samples, n_fine, perturb, netchunk, raw_noise_std)
    
    # Initialize outputs
    acc_all, pts_all = [], []
    acc0_all, weights0_all, pts0_all = [], [], []

    for i in range(0, n_rays, chunk_size):
        rays_chunk = rays[i:i+chunk_size]
        ret = render_chunk(rays_chunk, net, net_fine, n_samples, n_fine, perturb, netchunk, raw_noise_std)
        acc_all.append(ret["acc"])
        pts_all.append(ret["pts"])
        if "acc0" in ret:
            acc0_all.append(ret["acc0"])
            weights0_all.append(ret["weights0"])
            pts0_all.append(ret["pts0"])
    
    # Combine all chunks into final outputs
    ret = {
        "acc": torch.cat(acc_all, dim=0),
        "pts": torch.cat(pts_all, dim=0),
    }
    if acc0_all:
        ret["acc0"] = torch.cat(acc0_all, dim=0)
        ret["weights0"] = torch.cat(weights0_all, dim=0)
        ret["pts0"] = torch.cat(pts0_all, dim=0)

    return ret


def render_chunk(rays, net, net_fine, n_samples, n_fine, perturb, netchunk, raw_noise_std):
    """
    Core rendering logic for a single chunk of rays.
    Args are similar to render.
    """
    n_rays = rays.shape[0]
    rays_o, rays_d, near, far = rays[...,:3], rays[...,3:6], rays[...,6:7], rays[...,7:]

    # Sample depth along rays
    t_vals = torch.linspace(0., 1., steps=n_samples, device=near.device)
    z_vals = near * (1. - t_vals) + far * (t_vals)
    z_vals = z_vals.expand([n_rays, n_samples])

    if perturb:
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        lower = torch.cat([z_vals[..., :1], mids], -1)
        t_rand = torch.rand(z_vals.shape, device=lower.device)
        z_vals = lower + (upper - lower) * t_rand

    # Compute points along rays
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
    bound = net.bound - 1e-6
    pts = pts.clamp(-bound, bound)

    # Query the network for coarse samples
    #######
    
    raw = run_network(pts, net, netchunk)
    acc, weights = raw2outputs(raw, z_vals, rays_d, raw_noise_std)

    if net_fine is not None and n_fine > 0:
        acc_0 = acc
        weights_0 = weights
        pts_0 = pts

        z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        z_samples = sample_pdf(z_vals_mid, weights[..., 1:-1], n_fine, det=(perturb == 0.))
        z_samples = z_samples.detach()

        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
        pts = pts.clamp(-bound, bound)
        raw = run_network(pts, net_fine, netchunk)
        acc, _ = raw2outputs(raw, z_vals, rays_d, raw_noise_std)
    

    # Calculate total variation loss on points (rays)
    lambda_tv = 0.1
    tv_loss = compute_tv_regularization(pts) * lambda_tv



    ret = {"acc": acc, "pts": pts, "tv_loss":tv_loss}
    if net_fine is not None and n_fine > 0:
        ret["acc0"] = acc_0
        ret["weights0"] = weights_0
        ret["pts0"] = pts_0

    # Check for NaNs or infinities
    for k in ret:
        if torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any():
            print(f"! [Numerical Error] {k} contains nan or inf.")

    return ret

def run_network(inputs, fn, netchunk):
    uvt_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    out_flat = []
    for i in range(0, uvt_flat.shape[0], netchunk):
        chunk = uvt_flat[i:i+netchunk]
        out_flat.append(fn(chunk))
    out_flat = torch.cat(out_flat, 0)
    out = out_flat.reshape(list(inputs.shape[:-1]) + [out_flat.shape[-1]])
    return out





def forward_with_checkpoint(net, x):
    return checkpoint(net, x)



def process_voxel_chunks(voxels, net, chunk_size, netchunk):
        results = []
        for chunk in torch.split(voxels, chunk_size, dim=0):  # Split along the first dimension
            chunk_result = run_network(chunk, net, netchunk)
            results.append(chunk_result)
        return torch.cat(results, dim=0)  # Concatenate results along the first dimension





def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0.):
    """Transforms model"s predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """

    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([dists, torch.Tensor([1e-10]).expand(dists[..., :1].shape).to(dists.device)], -1)
    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[..., 0].shape) * raw_noise_std
        noise = noise.to(raw.device)

    acc = torch.sum((raw[..., 0] + noise) * dists, dim=-1)

    if raw.shape[-1] == 1:
        eps = torch.ones_like(raw[:, :1, -1]) * 1e-10
        weights = torch.cat([eps, torch.abs(raw[:, 1:, -1] - raw[:, :-1, -1])], dim=-1)
        weights = weights / torch.max(weights)
    elif raw.shape[-1] == 2: # with jac
        weights = raw[..., 1] / torch.max(raw[..., 1])
    else:
        raise NotImplementedError("Wrong raw shape")

    return acc, weights


def sample_pdf(bins, weights, N_samples, det=False):
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    # Invert CDF
    u = u.contiguous().to(cdf.device)
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples