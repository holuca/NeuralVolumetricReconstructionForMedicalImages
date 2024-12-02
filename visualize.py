import napari
import numpy as np
import matplotlib.pyplot as plt


tomo_gt = np.load("./data_npy/ground_truth_2.npy")
tomo_real = np.load("./logs/laminography_1000samples_3000rays/eval/epoch_00250/image_pred.npy")

#tomo_noRot = np.load("./logs/tomography_noRot/eval/epoch_00250/image_pred.npy")
#tomo = np.load("./logs/tomography_3000rays/eval/epoch_00250/image_pred.npy")
#lami_180 = np.load("./logs/laminography_180/eval/epoch_00250/image_pred.npy")
#lami_180_noRot = np.load(("./logs/laminography_180_noRot/eval/epoch_00250/image_pred.npy"))
#lami_360 = np.load("./logs/laminography_360/eval/epoch_00250/image_pred.npy")
#lami_360_noRot = np.load(("./logs/laminography_360_noRot/eval/epoch_00250/image_pred.npy"))


#tomo_now =  np.load(("./logs/tomography/eval/epoch_00220/image_pred.npy"))
viewer = napari.Viewer()


viewer.add_image(tomo_real, name='tomo_real', colormap='magma', opacity=0.6)
viewer.add_image(tomo_gt, name='GT', colormap='blue', opacity=0.6)


#iewer.add_image(CHEST, name="CHEST", colormap='blue', opacity=0.6)

#viewer.add_image(tomography_projections_gt, name='ORIG', colormap='magma', opacity=0.6)
#viewer.add_image(tomography_12, name='tomo_12', colormap='blue', opacity=0.6)
#viewer.add_image(tomography_02, name='tomo_02', colormap='magma', opacity=0.6)
#viewer.add_image(laminography_180_12, name='lami_180_12', colormap='gray', opacity=0.6)
#viewer.add_image(laminography_180_02, name='lami_180_02', colormap='magma', opacity=0.6)
#viewer.add_image(laminography_360_12, name='lami_360_12', colormap='gray', opacity=0.6)
#viewer.add_image(laminography_360_02, name='lami_360_02', colormap='magma', opacity=0.6)

#viewer.add_image(tomo_now, name='tomo_NOW', colormap='magma', opacity=0.6)
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
