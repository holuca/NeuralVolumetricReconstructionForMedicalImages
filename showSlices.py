import numpy as np
import imageio.v2 as iio

# Load ground truth and predicted images
image_gt = np.load("./logs/laminography_360/eval/epoch_00250/image_gt.npy")
image_pred = np.load("./logs/tomography_new/eval/epoch_00250/image_pred.npy")


num_slices = 5

# Choose the slicing axis
axis = 0  # Change this to 0, 1, or -1 to select different viewing angles

# Select 5 equally spaced slice indices along the chosen axis
slice_indices = np.linspace(0, image_gt.shape[axis] - 1, num_slices, dtype=int)

# Extract and concatenate slices based on the chosen axis
if axis == 0:
    gt_slices = [image_gt[idx, ...] for idx in slice_indices]
    pred_slices = [image_pred[idx, ...] for idx in slice_indices]
elif axis == 1:
    gt_slices = [image_gt[:, idx, :] for idx in slice_indices]
    pred_slices = [image_pred[:, idx, :] for idx in slice_indices]
else:  # Default is axis=-1 (top-down view)
    gt_slices = [image_gt[..., idx] for idx in slice_indices]
    pred_slices = [image_pred[..., idx] for idx in slice_indices]

# Concatenate slices to create rows
gt_row = np.concatenate(gt_slices, axis=1)
pred_row = np.concatenate(pred_slices, axis=1)

# Stack the rows to form a single image
combined_image = np.concatenate([gt_row, pred_row], axis=0)

# Normalize and save the image
combined_image = (combined_image - combined_image.min()) / (combined_image.max() - combined_image.min())
combined_image = (combined_image * 255).astype(np.uint8)
iio.imwrite('comparison_slices_side_view.png', combined_image)
