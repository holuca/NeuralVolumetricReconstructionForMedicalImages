import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as functional
import torchvision
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import pickle

from skimage.metrics import structural_similarity

#get_mse = lambda x, y: torch.mean((x - y) ** 2)
#get_mse = lambda x, y: torch.mean((torch.as_tensor(x) - torch.as_tensor(y)) ** 2)
#get_mse = lambda x, y: torch.mean((torch.as_tensor(x).float() - torch.as_tensor(y).float()) ** 2)
#get_mse = lambda x, y: torch.mean((x.real - y.real) ** 2 + (x.imag - y.imag) ** 2)
def get_mse(x, y):
    """
    Compute MSE between two tensors x and y.
    Handles both complex and real inputs.
    """
    if torch.is_complex(x) and torch.is_complex(y):
        return torch.mean((x.real - y.real) ** 2 + (x.imag - y.imag) ** 2)
    else:
        return torch.mean((x - y) ** 2)


def get_psnr(x, y):
    """
    Compute the PSNR between two tensors x and y.
    Assumes x and y are complex tensors and uses magnitude for computation.
    """
    # Convert complex tensors to magnitude
    x = torch.abs(x)
    y = torch.abs(y)

    # Handle edge cases
    if torch.max(x) == 0 or torch.max(y) == 0:
        return torch.zeros(1, device=x.device)

    # Normalize tensors
    x_norm = (x - torch.min(x)) / (torch.max(x) - torch.min(x))
    y_norm = (y - torch.min(y)) / (torch.max(y) - torch.min(y))

    # Compute MSE
    mse = get_mse(x_norm, y_norm)

    # Compute PSNR
    psnr = -10. * torch.log10(mse)
    return psnr



def get_psnr_3d(arr1, arr2, size_average=True, PIXEL_MAX=1.0):
    """
    :param arr1:
        Format-[NDHW], OriImage [0,1]
    :param arr2:
        Format-[NDHW], ComparedImage [0,1]
    :return:
        Format-None if size_average else [N]
    """
    if torch.is_tensor(arr1):
        arr1 = arr1.cpu().detach().numpy()
    if torch.is_tensor(arr2):
        arr2 = arr2.cpu().detach().numpy()
    arr1 = arr1[np.newaxis, ...]
    arr2 = arr2[np.newaxis, ...]
    arr1 = arr1.astype(np.float64)
    arr2 = arr2.astype(np.float64)
    eps = 1e-10
    se = np.power(arr1 - arr2, 2)
    mse = se.mean(axis=1).mean(axis=1).mean(axis=1)
    zero_mse = np.where(mse == 0)
    mse[zero_mse] = eps
    psnr = 20 * np.log10(PIXEL_MAX / np.sqrt(mse))
    # #zero mse, return 100
    psnr[zero_mse] = 100

    if size_average:
        return psnr.mean()
    else:
        return psnr


def get_ssim_3d(arr1, arr2, size_average=True, PIXEL_MAX=1.0):
    """
    :param arr1:
        Format-[NDHW], OriImage [0,1]
    :param arr2:
        Format-[NDHW], ComparedImage [0,1]
    :return:
        Format-None if size_average else [N]
    """
    if torch.is_tensor(arr1):
        arr1 = arr1.cpu().detach().numpy()
    if torch.is_tensor(arr2):
        arr2 = arr2.cpu().detach().numpy()
    arr1 = arr1[np.newaxis, ...]
    arr2 = arr2[np.newaxis, ...]
    assert (arr1.ndim == 4) and (arr2.ndim == 4)
    arr1 = arr1.astype(np.float64)
    arr2 = arr2.astype(np.float64)

    N = arr1.shape[0]
    # Depth
    arr1_d = np.transpose(arr1, (0, 2, 3, 1))
    arr2_d = np.transpose(arr2, (0, 2, 3, 1))
    ssim_d = []
    for i in range(N):
        ssim = structural_similarity(arr1_d[i], arr2_d[i])
        ssim_d.append(ssim)
    ssim_d = np.asarray(ssim_d, dtype=np.float64)

    # Height
    arr1_h = np.transpose(arr1, (0, 1, 3, 2))
    arr2_h = np.transpose(arr2, (0, 1, 3, 2))
    ssim_h = []
    for i in range(N):
        ssim = structural_similarity(arr1_h[i], arr2_h[i])
        ssim_h.append(ssim)
    ssim_h = np.asarray(ssim_h, dtype=np.float64)

    # Width
    # arr1_w = np.transpose(arr1, (0, 1, 2, 3))
    # arr2_w = np.transpose(arr2, (0, 1, 2, 3))
    ssim_w = []
    for i in range(N):
        ssim = structural_similarity(arr1[i], arr2[i])
        ssim_w.append(ssim)
    ssim_w = np.asarray(ssim_w, dtype=np.float64)

    ssim_avg = (ssim_d + ssim_h + ssim_w) / 3

    if size_average:
        return ssim_avg.mean()
    else:
        return ssim_avg


def cast_to_image_orig(tensor, normalize=True):
    # tensor range: [0, 1]
    # Conver to PIL Image and then np.array (output shape: (H, W))
    if torch.is_tensor(tensor):
        img = tensor.cpu().detach().numpy()
    else:
        img = tensor
    if normalize:
        img = cv2.normalize(img, None, 0, 1, cv2.NORM_MINMAX)
    return img[..., np.newaxis]


# Updated cast_to_image to handle complex inputs
def cast_to_image(tensor, normalize=True):
    # tensor range: [0, 1]
    # Convert to numpy array
    if torch.is_tensor(tensor):
        img = tensor.cpu().detach().numpy()
    else:
        img = tensor

    # If input is complex, use only the magnitude
    if np.iscomplexobj(img):
        img = np.abs(img)

    if normalize:
        img = cv2.normalize(img, None, 0, 1, cv2.NORM_MINMAX)

    return img[..., np.newaxis]


def get_ptycho_mask_1d(projs, threshold=0.007):
    """
    Compute a binary mask for projections to exclude values below a threshold.

    Args:
        projs (torch.Tensor): The input projections, either 1D (sampled rays) or 2D (full projection).
        threshold (float): The magnitude threshold for masking.

    Returns:
        torch.Tensor: A binary mask with the same shape as the input.
    """
    with torch.no_grad():
        if projs.ndim == 1:  # 1D input (sampled rays)
            mask = torch.abs(projs) > threshold
        elif projs.ndim == 2:  # 2D input (full projection)
            mask = torch.abs(projs) > threshold
            mask[1:] &= mask[1:] == mask[:-1]
            mask[:, 1:] &= mask[:, 1:] == mask[:, :-1]
        else:
            raise ValueError(f"Unsupported input dimension {projs.ndim}. Expected 1D or 2D.")
        return mask


def get_ptycho_mask(hr, threshold=0.007):
    """
    hr must be a complex projection
    """
    with torch.no_grad():
        mask = torch.abs(hr) < threshold
        mask &= torch.abs(hr) > -threshold
        mask[1:] &= mask[1:] == mask[:-1]
        mask[:, 1:] &= mask[:, 1:] == mask[:, :-1]
        return ~mask
def manual_vmap(func, inputs, *args, **kwargs):
    """
    Applies a function `func` to each batch element of `inputs` and stacks the results.

    Args:
        func: Function to be applied.
        inputs: Batch of inputs, first dimension is the batch size.
        *args, **kwargs: Additional arguments to `func`.

    Returns:
        Stacked results of applying `func` to each element in `inputs`.
    """
    return torch.stack([func(inp, *args, **kwargs) for inp in inputs])




def visualize_sampled_points(full_mask, sampled_coords, mask_sampled, global_step):
    """
    Visualize sampled points and their masking on the full mask.
    
    Args:
        full_mask (torch.Tensor): The 2D full mask.
        sampled_coords (torch.Tensor): Coordinates of the sampled points.
        mask_sampled (torch.Tensor): Mask values at the sampled coordinates.
        global_step (int): Current global training step, for saving the plot.
    """
    # Convert tensors to CPU and numpy for visualization
    full_mask_np = full_mask.cpu().numpy()
    sampled_coords_np = sampled_coords.cpu().numpy()
    mask_sampled_np = mask_sampled.cpu().numpy()

    # Separate the points based on their mask values
    valid_points = sampled_coords_np[mask_sampled_np > 0]
    invalid_points = sampled_coords_np[mask_sampled_np == 0]

    # Create the plot
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    
    # First subplot: Full mask with all sampled points
    ax[0].imshow(full_mask_np, cmap='gray', origin='upper')
    ax[0].scatter(sampled_coords_np[:, 1], sampled_coords_np[:, 0], c='yellow', s=2, label='Sampled Points')
    ax[0].set_title("Full Mask with Sampled Points")
    ax[0].legend(loc='upper right')

    # Second subplot: Full mask with valid and invalid sampled points
    ax[1].imshow(full_mask_np, cmap='gray', origin='upper')
    if len(valid_points) > 0:
        ax[1].scatter(valid_points[:, 1], valid_points[:, 0], c='red', s=2, label='Valid Points')
    if len(invalid_points) > 0:
        ax[1].scatter(invalid_points[:, 1], invalid_points[:, 0], c='blue', s=2, label='Invalid Points')
    ax[1].set_title("Full Mask with Valid (Red) and Invalid (Blue) Points")
    ax[1].legend(loc='upper right')

    # Save the plot
    plt.tight_layout()
    plt.savefig(f'sampled_points_visualization_step_{global_step}.png')
    plt.close(fig)

    print(f"Visualization saved as 'sampled_points_visualization_step_{global_step}.png'")

def visualize_after_mask(full_mask, sampled_coords, projs_values, global_step, title_suffix=""):
    """
    Visualize sampled points after applying the mask.
    
    Args:
        full_mask (torch.Tensor): The 2D full mask.
        sampled_coords (torch.Tensor): Coordinates of the sampled points.
        projs_values (torch.Tensor): Values after applying the mask (e.g., projs_phase_normalized or projs_pred_chunk).
        global_step (int): Current global training step, for saving the plot.
        title_suffix (str): Suffix for plot titles to distinguish visualizations.
    """
    # Convert tensors to CPU and numpy for visualization
    full_mask_np = full_mask.cpu().numpy()
    sampled_coords_np = sampled_coords.cpu().numpy()
    projs_values_np = projs_values.detach().cpu().numpy()

    # Separate points based on their values (0 = invalid, non-zero = valid)
    valid_points = sampled_coords_np[projs_values_np != 0]
    invalid_points = sampled_coords_np[projs_values_np == 0]

    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    # Full mask as background
    ax.imshow(full_mask_np, cmap='gray', origin='upper')
    if len(valid_points) > 0:
        ax.scatter(valid_points[:, 1], valid_points[:, 0], c='green', s=2, label='Valid Points')
    if len(invalid_points) > 0:
        ax.scatter(invalid_points[:, 1], invalid_points[:, 0], c='purple', s=2, label='Invalid Points')
    ax.set_title(f"Full Mask with Points after Mask Application {title_suffix}")
    ax.legend(loc='upper right')

    # Save the plot
    plt.tight_layout()
    plt.savefig(f'points_after_mask_step_{global_step}{title_suffix}.png')
    plt.close(fig)

    print(f"Visualization saved as 'points_after_mask_step_{global_step}{title_suffix}.png'")
