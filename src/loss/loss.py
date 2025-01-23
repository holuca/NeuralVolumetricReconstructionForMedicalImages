import torch
import torch.fft
import math

def fourier_transform(x):
    return torch.fft.fft2(x)

def inverse_fourier_transform(x):
    return torch.fft.ifft2(x)
def compute_tv_regularization(loss, values, weight):
    """
    Compute total variation along rays.
    
    Args:
        values: Tensor of shape [num_rays, num_samples, feature_dim] (e.g., densities, colors).
    
    Returns:
        tv_loss: Total variation regularization term.
    """
    # Compute differences along ray samples (axis=1)
    diffs = values[:, 1:, :] - values[:, :-1, :]
    tv_loss = torch.sum(torch.abs(diffs))  # L1 norm of differences
    loss["loss"] = loss["loss"] + tv_loss * weight
    return loss

def calc_mse_loss(loss, x, y, tv_loss=None):
    """
    Calculate MSE loss and optionally include total variation (TV) loss.

    Args:
        loss: Dictionary to store loss components.
        x: Ground truth values.
        y: Predicted values.
        tv_loss: Total variation loss (optional). Default is None.
    """
    # Compute MSE loss
    loss_mse = torch.mean((x - y) ** 2)
    loss["loss"] += loss_mse
    loss["loss_mse"] = loss_mse

    # If TV loss is provided, add it to the total loss
    if tv_loss is not None:
        loss["loss"] += tv_loss
        loss["tv_loss"] = tv_loss

    return loss


def calc_phase_only_loss(loss, x, y):
    """
    Calculate MSE loss based only on the phase.
    """
    x_phase = torch.angle(x)
    y_phase = torch.angle(y)

    # Normalize phase values to [0, 1]
    x_phase_normalized = (x_phase + torch.pi) / (2 * torch.pi)
    y_phase_normalized = (y_phase + torch.pi) / (2 * torch.pi)

    # Compute MSE for phase
    loss_mse_phase = torch.mean((x_phase_normalized - y_phase_normalized)**2)

    # Accumulate into the loss dictionary
    loss["loss"] += loss_mse_phase
    loss["phase_loss"] = loss_mse_phase
    return loss

def calc_mse_loss_mask(loss, x, y, mask=None):
    """
    Calculate masked MSE loss.
    Args:
        x: Ground truth projections.
        y: Predicted projections.
        mask: Binary mask (same shape as x and y) where True indicates valid data points.
    """
    if mask is not None:
        x = x[mask]
        y = y[mask]

    # Compute loss
    loss_mse = torch.mean((x - y) ** 2)
    loss["loss"] += loss_mse
    loss["loss_mse"] = loss_mse
    return loss


def calc_hinge_loss(loss, x, y):
    """
    Calculate hinge loss.
    """
    # Compute hinge loss
    hinge_loss = torch.mean(torch.clamp(1 - x * y, min=0))  # hinge loss formula
    loss["loss"] += hinge_loss  # Accumulate the loss
    loss["loss_hinge"] = hinge_loss  # Store the specific hinge loss
    return loss

def calc_mse_loss_with_gradient(loss, x, y, mask=None, lambda_grad=0.1):
    """
    Calculate MSE loss with a gradient-based regularizer.
    
    Args:
        loss (dict): Dictionary to store different components of the loss.
        x (torch.Tensor): Predicted tensor (2D, [H, W]).
        y (torch.Tensor): Ground truth tensor (2D, [H, W]).
        mask (torch.Tensor): Binary mask to focus the loss on certain regions (optional).
        lambda_grad (float): Weight for the gradient regularizer.
    
    Returns:
        dict: Updated loss dictionary.
    """
    if mask is not None:
        # Apply the mask without flattening
        x = x * mask
        y = y * mask

    # Compute MSE loss
    loss_mse = torch.mean((x - y) ** 2)
    loss["loss_mse"] = loss_mse

    # Compute gradient loss
    def compute_gradients(tensor):
        grad_x = tensor[:, 1:] - tensor[:, :-1]  # Horizontal gradients
        grad_y = tensor[1:, :] - tensor[:-1, :]  # Vertical gradients
        return grad_x, grad_y

    x_grad_x, x_grad_y = compute_gradients(x)
    y_grad_x, y_grad_y = compute_gradients(y)

    grad_loss_x = torch.mean((x_grad_x - y_grad_x) ** 2)
    grad_loss_y = torch.mean((x_grad_y - y_grad_y) ** 2)
    loss_grad = grad_loss_x + grad_loss_y
    loss["loss_grad"] = loss_grad

    # Combine losses
    loss["loss"] += loss_mse + lambda_grad * loss_grad

    return loss


def calc_huber_loss(loss, x, y, delta=1.0):
    """
    Calculate Huber loss.
    Args:
        loss (dict): Loss dictionary to update.
        x (Tensor): Predicted tensor.
        y (Tensor): Ground truth tensor.
        delta (float): Threshold for Huber loss transition.
    """
    diff = x - y
    abs_diff = torch.abs(diff)

    # Compute Huber loss
    loss_huber = torch.where(
        abs_diff <= delta,
        0.5 * diff**2,
        delta * (abs_diff - 0.5 * delta)
    ).mean()

    # Update loss dictionary
    loss["loss"] += loss_huber
    loss["loss_huber"] = loss_huber

    return loss

#loss term that penalizes predictions that deviate from zero where the real data is near-zero
def calc_zero_loss(loss, pred, real_data, threshold=1e-5, weight=1.0):
    """
    Penalize non-zero predictions in regions of real data close to zero.
    Args:
        loss: Dictionary to store the loss values.
        pred: Predicted tensor.
        real_data: Real data tensor.
        threshold: Threshold to define near-zero regions.
        weight: Weight for the zero loss.
    """
    zero_region = (real_data.abs() <= threshold).float()
    zero_loss = weight * torch.mean(zero_region * pred ** 2)  # Penalize non-zero predictions
    loss["loss"] += zero_loss
    loss["loss_zero"] = zero_loss
    return loss

#Encourage small predictions by adding an overall regularization loss: bias the network toward predicitng black in ambiguous regions
def calc_small_loss(loss, pred, weight=1.0):
    """
    Penalize large predictions across the entire output.
    Args:
        loss: Dictionary to store the loss values.
        pred: Predicted tensor.
        weight: Weight for the small prediction loss.
    """
    small_loss = weight * torch.mean(pred ** 2)  # Penalize large predictions
    loss["loss"] += small_loss
    loss["loss_small"] = small_loss
    return loss

def calc_tv_loss_3d_orig(loss, x, k):
    """
    Calculate total variation loss.
    Args:
        x (n1, n2, n3): 3d density field.
        k: relative weight
    """
    # Ensure x has 3 dimensions
    if x.ndim != 3:
        raise ValueError(f"Expected x to have 3 dimensions (n1, n2, n3), but got {x.ndim}.")

    n1, n2, n3 = x.shape
    tv_1 = torch.abs(x[1:, :, :] - x[:-1, :, :]).sum()  # TV along the first dimension
    tv_2 = torch.abs(x[:, 1:, :] - x[:, :-1, :]).sum()  # TV along the second dimension
    tv_3 = torch.abs(x[:, :, 1:] - x[:, :, :-1]).sum()  # TV along the third dimension
    tv = (tv_1 + tv_2 + tv_3) / (n1 * n2 * n3)
    loss["loss"] += tv * k
    loss["loss_tv"] = tv * k
    return loss



def calc_tv_loss(loss, image, weight):
    """
    Calculate total variation loss.
    Args:
        loss: Dictionary to store the loss values.
        image: Tensor of shape [B, H, W] or [H, W].
        weight: Weight for the TV loss.
    Returns:
        Updated loss dictionary with TV loss.
    """
    tv_h = torch.mean((image[..., :-1, :] - image[..., 1:, :]) ** 2)
    tv_w = torch.mean((image[..., :, :-1] - image[..., :, 1:]) ** 2)
    tv_loss = weight * (tv_h + tv_w)
    loss["loss"] += tv_loss
    loss["loss_tv"] = tv_loss
    return loss



def total_variation_loss(densities):
    """
    Compute the Total Variation (TV) loss on volumetric densities along rays.
    Args:
        densities (torch.Tensor): [n_rays, n_samples] or similar. Volumetric densities predicted.
        weights (torch.Tensor): [n_rays, n_samples] or similar. Weights along rays.
    Returns:
        tv_loss (torch.Tensor): Scalar TV loss value.
    """
    # Calculate differences between consecutive density samples
    tv_loss = torch.mean(torch.abs(densities[:, 1:] - densities[:, :-1]))
    return tv_loss

#predicted x and target y 
def calc_fourier_loss(loss, x, y, lambda_sparsity=0.01, lambda_smoothness=0.01):
    """
    Calculate Fourier loss with sparsity and smoothness regularization.
    Args:
        loss (dict): Dictionary to store loss components.
        x (Tensor): Predicted tensor (2D or reshaped to 2D).
        y (Tensor): Ground truth tensor (2D or reshaped to 2D).
        lambda_sparsity (float): Weight for sparsity loss.
        lambda_smoothness (float): Weight for smoothness loss.
    """
    # Ensure x and y are 2D
    if x.ndim < 2 or y.ndim < 2:
        raise ValueError("Inputs x and y must have at least 2 dimensions.")

    # Compute Fourier transforms
    x_fft = torch.fft.fft2(x)
    y_fft = torch.fft.fft2(y)

    # Calculate magnitude of the Fourier coefficients
    x_fft_abs = torch.abs(x_fft)
    y_fft_abs = torch.abs(y_fft)

    # Sparsity loss
    loss_sparsity = lambda_sparsity * torch.sum(x_fft_abs)

    # Smoothness loss (finite differences method)
    if x_fft_abs.shape[-2] > 1 and x_fft_abs.shape[-1] > 1:
        dx = x_fft_abs[:, 1:, :] - x_fft_abs[:, :-1, :]  # Vertical differences
        dy = x_fft_abs[:, :, 1:] - x_fft_abs[:, :, :-1]  # Horizontal differences
        loss_smoothness = lambda_smoothness * (dx.abs().mean() + dy.abs().mean())
    else:
        loss_smoothness = torch.tensor(0.0, device=x.device)  # No smoothness penalty if dimensions are too small

    # Reconstruction loss
    loss_fourier_reconstruction = torch.mean((x_fft_abs - y_fft_abs) ** 2)

    loss_fourier = loss_fourier_reconstruction + loss_sparsity + loss_smoothness
    
    # Update the provided loss dictionary
    loss["loss"] += loss_fourier
    # Update loss dictionary
    loss["loss_fourier_reconstruction"] = loss_fourier_reconstruction
    loss["loss_sparsity"] = loss_sparsity
    loss["loss_smoothness"] = loss_smoothness
    loss["loss"] += loss_fourier
    return loss


def calc_fourier_sparsity_loss(loss, y, weight):
    """
    Calculate Fourier space sparsity loss using the L1 norm.
    Args:
        loss (dict): Dictionary to store loss values.
        y (tensor): Predicted tensor (image or reconstruction).
        weight (float): Weight for the Fourier sparsity loss.
    """
    # Ensure the input has at least two dimensions (H, W)
    if y.dim() < 2:
        raise ValueError(f"Input tensor y must have at least 2 dimensions for Fourier transform, but got shape {y.shape}")

    # If y has more than 2 dimensions, take the Fourier transform over the last two dimensions
    fft_y = torch.fft.fftshift(torch.fft.fft2(y, dim=(-2, -1)))  # Fourier transform

    # Compute the L1 norm of the Fourier coefficients (sparsity)
    sparsity_loss = torch.mean(torch.abs(fft_y))

    # Add weighted sparsity loss to the loss dictionary
    loss["loss"] += sparsity_loss * weight
    loss["loss_fourier_sparsity"] = sparsity_loss * weight
    return loss




def calc_l1_loss(loss, x, y):
    """
    Calculate L1 loss.
    Args:
        loss (dict): Dictionary to accumulate loss values.
        x (torch.Tensor): Ground truth tensor.
        y (torch.Tensor): Predicted tensor.
    Returns:
        dict: Updated loss dictionary.
    """
    # Compute L1 loss
    loss_l1 = torch.mean(torch.abs(x - y))
    loss["loss"] += loss_l1
    loss["loss_l1"] = loss_l1
    return loss