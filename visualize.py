import napari
import numpy as np

image_gt = np.load('logs/lamino_chip_normalized_DSO2000/eval/epoch_01000/image_pred.npy')
image_pred_scaled = np.load('logs/lamino_chip_normalized/eval/epoch_01500/image_pred.npy')
viewer = napari.Viewer()

viewer.add_image(image_gt, name='Ground Truth Image', colormap='magma', opacity=0.6)
#viewer.add_image(image_pred_x, name='Image pred X-aixs0', colormap='gray', opacity=0.6)
viewer.add_image(image_pred_scaled, name='Image pred scaled', colormap='gray', opacity=0.6)



napari.run()