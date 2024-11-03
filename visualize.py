import napari
import numpy as np
import matplotlib.pyplot as plt


tomography_projections_gt = np.load("./data_npy/ground_truth.npy")
tomography_ORIG = tomography_projections_gt
tomography_projections_gt = np.rot90(tomography_projections_gt, k=3, axes=(0, 2))
tomography_ROTATED = tomography_projections_gt
tomography_projections_gt = np.rot90(tomography_projections_gt, k=1, axes=(1, 2))



pred_laminography = np.load("./logs/laminography_4096/eval/epoch_01250/image_pred.npy")
viewer = napari.Viewer()

viewer.add_image(tomography_ORIG, name='ORIG', colormap='magma', opacity=0.6)
viewer.add_image(pred_laminography, name='predicted_lami', colormap='blue', opacity=0.6)
viewer.add_image(tomography_ROTATED, name='rotated', colormap='gray', opacity=0.6)
#viewer.add_image(laminography_pred, name='lami_pred', colormap='gray', opacity=0.6)
napari.run()


###look at images
#num_images = 5
## Display a few images
## Calculate the step size to evenly distribute the selected images
#step_size = max(1, 360 // num_images)
#selected_indices = range(0, 360, step_size)[:num_images]
#
#fig, axes = plt.subplots(1, num_images, figsize=(15, 15))
#
#
#for i, idx in enumerate(selected_indices):
#    ax = axes[i]
#    ax.imshow(image_gt[i], cmap='gray' if image_gt.shape[-1] == 1 else None)
#    ax.axis('off')
#
#plt.show()
