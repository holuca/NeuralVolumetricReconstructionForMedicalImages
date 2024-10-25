import napari
import numpy as np

image_gt = np.load('logs/lamino_chip/eval/epoch_01000/image_pred.npy')
image_pred_scaled = np.load('logs/lamino_chip_DSO2000/eval/epoch_01000/image_pred.npy')
image_pred_x = np.load('logs/lamino_chip_DSO1500/eval/epoch_01000/image_pred.npy')
viewer = napari.Viewer()

viewer.add_image(image_gt, name='orig', colormap='magma', opacity=0.6)
viewer.add_image(image_pred_x, name='DSO1500', colormap='blue', opacity=0.6)
viewer.add_image(image_pred_scaled, name='DSO2000', colormap='gray', opacity=0.6)



napari.run()