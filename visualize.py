import napari
import numpy as np

image_gt = np.load('logs/chest_50_correct/eval/epoch_03000/image_pred.npy')
image_pred_scaled = np.load('logs/chest_50_07Zoom/eval/epoch_01500/image_pred.npy')
image_pred_x = np.load('logs/chest_50_xayis/eval/epoch_01500/image_pred.npy')
viewer = napari.Viewer()

viewer.add_image(image_gt, name='Ground Truth Image', colormap='magma', opacity=0.6)
viewer.add_image(image_pred_x, name='Image pred X-aixs0', colormap='gray', opacity=0.6)
viewer.add_image(image_pred_scaled, name='Image pred scaled', colormap='blue', opacity=0.6)



napari.run()