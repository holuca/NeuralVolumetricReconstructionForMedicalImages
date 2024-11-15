import napari
import numpy as np
import matplotlib.pyplot as plt


tomography_projections_gt = np.load("./data_npy/ground_truth.npy")
tomography_ORIG = tomography_projections_gt
#tomography_projections_gt = np.rot90(tomography_projections_gt, k=3, axes=(0, 2))
#tomography_ROTATED = tomography_projections_gt
#tomography_projections_gt = np.rot90(tomography_projections_gt, k=1, axes=(1, 2))

#tomo_noRot = np.load("./logs/tomography_noRot/eval/epoch_00250/image_pred.npy")
#tomo = np.load("./logs/tomography_3000rays/eval/epoch_00250/image_pred.npy")
k212 = np.load("./logs/laminography_k212/eval/epoch_00250/image_pred.npy")
lami_180 = np.load("./logs/laminography_180/eval/epoch_00250/image_pred.npy")
lami_180_noRot = np.load(("./logs/laminography_180_noRot/eval/epoch_00250/image_pred.npy"))
lami_360 = np.load("./logs/laminography_360/eval/epoch_00250/image_pred.npy")
lami_360_noRot = np.load(("./logs/laminography_360_noRot/eval/epoch_00250/image_pred.npy"))


tomo_now =  np.load(("./logs/tomography_new/eval/epoch_00250/image_pred.npy"))
#current = np.load("./logs/laminography/eval/epoch_00250/image_pred.npy")
viewer = napari.Viewer()


#viewer.add_image(tomo, name='tomo_ManyRays', colormap='magma', opacity=0.6)
#viewer.add_image(tomo_noRot, name='tomo_noRot', colormap='blue', opacity=0.6)

viewer.add_image(tomography_ORIG, name='ORIG', colormap='magma', opacity=0.6)
viewer.add_image(lami_180, name='lami_180', colormap='blue', opacity=0.6)
viewer.add_image(k212, name='k212', colormap='gray', opacity=0.6)
viewer.add_image(lami_360, name='lami_360', colormap='magma', opacity=0.6)
viewer.add_image(lami_180_noRot, name='lami_180_noRot', colormap='gray', opacity=0.6)
viewer.add_image(lami_360_noRot, name='lami_360_noRot', colormap='magma', opacity=0.6)

viewer.add_image(tomo_now, name='tomo_NOW', colormap='magma', opacity=0.6)
#viewer.add_image(current, name='CURRENT', colormap='magma', opacity=0.6)
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
