


from math import log10, sqrt 
import numpy as np 
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt


ground_truth = np.load("./data_npy/ground_truth_2.npy")
tomography = np.load("./logs/tomography_2_180/eval/epoch_00250/image_pred.npy")
tomography = np.flip(tomography, axis=0)

laminography_45 = np.load("./logs/laminography_45/eval/epoch_00500/laminoggraphy45_image_pred.npy")
laminography_45 = np.flip(laminography_45, axis=0)

laminography_90 = np.load("./logs/laminography_90/eval/epoch_00500/laminography90_image_pred.npy")
laminography_90 = np.flip(laminography_90, axis=0)

laminography_180 = np.load("./logs/laminography_180/eval/epoch_00500/laminography180_image_pred.npy")
laminography_180 = np.flip(laminography_180, axis=0)

laminography_360 = np.load("./logs/laminography_360/eval/epoch_00500/laminography360_image_pred.npy")
laminography_360 = np.flip(laminography_360, axis=0)

laminography_360_250epochs = np.load("./logs/laminography_360/eval/epoch_00250/image_pred.npy")
laminography_360_250epochs = np.flip(laminography_360_250epochs, axis=0)

laminography_100samples_3000rays = np.load("./logs/laminography_1000samples_3000rays/eval/epoch_00500/laminography360_100samples_3000rays_image_pred.npy")
laminography_100samples_3000rays = np.flip(laminography_100samples_3000rays, axis=0)

laminography_3000rays = np.load("./logs/laminography_3000rays/eval/epoch_00500/image_pred.npy")
laminography_3000rays = np.flip(laminography_3000rays, axis=0)

laminography_1000samples = np.load("./logs/laminography_1000samples/eval/epoch_00500/image_pred.npy")
laminography_1000samples = np.flip(laminography_1000samples, axis=0)

laminography_100samples_3000rays_250epochs = np.load("./logs/laminography_3000rays/eval/epoch_00250/image_pred.npy")
laminography_100samples_3000rays_250epochs = np.flip(laminography_100samples_3000rays_250epochs, axis=0)



def PSNR(reconstructed, ground_truth, max_pixel=1.0):
    mse = np.mean((reconstructed - ground_truth) ** 2)
    if mse == 0:  # Perfect match
        return float('inf')
    psnr = 10 * np.log10(max_pixel**2 / mse)
    return psnr

def log_PSNR(reconstructed, ground_truth, max_pixel=1.0, log_base=10):
    
    mse = np.mean((reconstructed - ground_truth) ** 2)
    if mse == 0:  # Perfect match
        return float('inf')
    log_psnr = 20 * (np.log(max_pixel / np.sqrt(mse)) / np.log(log_base))
    return log_psnr

def volume_PSNR(reconstructed, ground_truth, max_pixel=1.0, log_base=10):
    mse = np.mean((reconstructed - ground_truth) ** 2)
    if mse == 0:  # Perfect match
        return float('inf')
    log_psnr = 20 * (np.log(max_pixel / np.sqrt(mse)) / np.log(log_base))
    return log_psnr

volume_log_psnr_lami45 =  volume_PSNR(laminography_45, ground_truth, max_pixel=1.0, log_base=2)
volume_log_psnr_lami90 =  volume_PSNR(laminography_90, ground_truth, max_pixel=1.0, log_base=2)
volume_log_psnr_lami180 = volume_PSNR(laminography_180, ground_truth, max_pixel=1.0, log_base=2)
volume_log_psnr_lami360 = volume_PSNR(laminography_360, ground_truth, max_pixel=1.0, log_base=2)
#print(f"Volume Log PSNR (Reconstruction Lami45): {volume_log_psnr_lami45:.2f}")
#print(f"Volume Log PSNR (Reconstruction Lami90): {volume_log_psnr_lami90:.2f}")
#print(f"Volume Log PSNR (Reconstruction Lami180): {volume_log_psnr_lami180:.2f}")
#print(f"Volume Log PSNR (Reconstruction Lami360): {volume_log_psnr_lami360:.2f}")


volume_log_psnr_1000samples =   volume_PSNR(laminography_1000samples, ground_truth, max_pixel=1.0, log_base=2)
volume_log_psnr_3000rays =      volume_PSNR(laminography_3000rays, ground_truth, max_pixel=1.0, log_base=2)
volume_log_psnr_both =          volume_PSNR(laminography_100samples_3000rays, ground_truth, max_pixel=1.0, log_base=2)
#print(f"Volume Log PSNR (Reconstruction Lami360): {volume_log_psnr_lami360:.2f}")
#print(f"Volume Log PSNR (Reconstruction 1000samples): {volume_log_psnr_1000samples:.2f}")
#print(f"Volume Log PSNR (Reconstruction 3000rays): {volume_log_psnr_3000rays:.2f}")
#print(f"Volume Log PSNR (Reconstruction both): {volume_log_psnr_both:.2f}")

volume_log_psnr_both_250epochs =    volume_PSNR(laminography_100samples_3000rays_250epochs, ground_truth, max_pixel=1.0, log_base=2)
volume_log_psnr_lami360_250epochs = volume_PSNR(laminography_360_250epochs, ground_truth, max_pixel=1.0, log_base=2)
#print(f"Volume Log PSNR (Reconstruction Lami360): {volume_log_psnr_lami360:.2f}")
#print(f"Volume Log PSNR (Reconstruction 250Epochs): {volume_log_psnr_lami360_250epochs:.2f}")
#print(f"Volume Log PSNR (Reconstruction both): {volume_log_psnr_both:.2f}")
#print(f"Volume Log PSNR (Reconstruction 250Epochs): {volume_log_psnr_both_250epochs:.2f}")



##SSIM

def ssim_2d(reconstructed, ground_truth):
    return ssim(reconstructed, ground_truth, data_range=ground_truth.max() - ground_truth.min())

def average_2d_ssim(reconstructed_volume, ground_truth_volume):
    ssim_values = []
    for i in range(reconstructed_volume.shape[2]):  # Iterate over z-axis slices
        ssim_value = ssim_2d(reconstructed_volume[:, :, i], ground_truth_volume[:, :, i])
        ssim_values.append(ssim_value)
    return np.mean(ssim_values)

# Example: Compute average SSIM for each reconstruction
average_ssim_lami45 =  average_2d_ssim(laminography_45, ground_truth)
average_ssim_lami90 =  average_2d_ssim(laminography_90, ground_truth)
average_ssim_lami180 = average_2d_ssim(laminography_180, ground_truth)
average_ssim_lami360 = average_2d_ssim(laminography_360, ground_truth)

#print(f"Average 2D SSIM (Reconstruction lami45): {average_ssim_lami45:.4f}")
#print(f"Average 2D SSIM (Reconstruction lami90): {average_ssim_lami90:.4f}")
#print(f"Average 2D SSIM (Reconstruction lami180): {average_ssim_lami180:.4f}")
#print(f"Average 2D SSIM (Reconstruction lami360): {average_ssim_lami360:.4f}")


average_ssim_1000samples =   average_2d_ssim(laminography_1000samples, ground_truth)
average_ssim_3000rays =      average_2d_ssim(laminography_3000rays, ground_truth)
average_ssim_both =          average_2d_ssim(laminography_100samples_3000rays, ground_truth)


#print(f"Average 2D SSIM (Reconstruction lami360): {average_ssim_lami360:.4f}")
#print(f"Average 2D SSIM (Reconstruction 1000samples): {average_ssim_1000samples:.4f}")
#print(f"Average 2D SSIM (Reconstruction 3000rays): {average_ssim_3000rays:.4f}")
#print(f"Average 2D SSIM (Reconstruction both): {average_ssim_both:.4f}")


average_ssim_both_250epochs =    average_2d_ssim(laminography_100samples_3000rays_250epochs, ground_truth)
average_ssim_lami360_250epochs = average_2d_ssim(laminography_360_250epochs, ground_truth)
print(f"Average 2D SSIM (Reconstruction lami360): {average_ssim_lami360:.4f}")
print(f"Average 2D SSIM(Reconstruction 250Epochs): {average_ssim_lami360_250epochs:.2f}")
print(f"Average 2D SSIM (Reconstruction both): {average_ssim_both:.4f}")
print(f"Average 2D SSIM (Reconstruction 250Epochs): {average_ssim_both_250epochs:.2f}")

# Calculate PSNR for each slice along the z-axis
psnr_values_tomo = []
psnr_values_lami45 = []
psnr_values_lami90 = []
psnr_values_lami180 = []
psnr_values_lami360 = []
psnr_values_1000samples = []
psnr_values_3000rays = []
psnr_values_1000samples3000rays = []
psnr_values_lami360_250epochs = []
psnr_values_1000samples3000rays_250epochs = []

for i in range(tomography.shape[2]):  # Iterate over z-axis slices
    psnr1 = log_PSNR(tomography[:, :, i], ground_truth[:, :, i], max_pixel=1.0)
    psnr2 = log_PSNR(laminography_45[:, :, i], ground_truth[:, :, i], max_pixel=1.0)
    psnr3 = log_PSNR(laminography_90[:, :, i], ground_truth[:, :, i], max_pixel=1.0)
    psnr4 = log_PSNR(laminography_180[:, :, i], ground_truth[:, :, i], max_pixel=1.0)
    psnr5 = log_PSNR(laminography_360[:, :, i], ground_truth[:, :, i], max_pixel=1.0)
    psnr6 = log_PSNR(laminography_1000samples[:, :, i], ground_truth[:, :, i], max_pixel=1.0)
    psnr7 = log_PSNR(laminography_3000rays[:, :, i], ground_truth[:, :, i], max_pixel=1.0)
    psnr8 = log_PSNR(laminography_100samples_3000rays[:, :, i], ground_truth[:, :, i], max_pixel=1.0)
    psnr9 = log_PSNR(laminography_360_250epochs[:, :, i], ground_truth[:, :, i], max_pixel=1.0)
    psnr10 = log_PSNR(laminography_100samples_3000rays_250epochs[:, :, i], ground_truth[:, :, i], max_pixel=1.0)


    psnr_values_tomo.append(psnr1)
    psnr_values_lami45.append(psnr2)
    psnr_values_lami90.append(psnr3)
    psnr_values_lami180.append(psnr4)
    psnr_values_lami360.append(psnr5)
    psnr_values_1000samples.append(psnr6)
    psnr_values_3000rays.append(psnr7)
    psnr_values_1000samples3000rays.append(psnr8)
    psnr_values_lami360_250epochs.append(psnr9)
    psnr_values_1000samples3000rays_250epochs.append(psnr10)

# Plot PSNR values
plt.figure(figsize=(10, 6))

##Comparison number of angles
#plt.plot(range(len(psnr_values_tomo)), psnr_values_tomo, label="tomo")
#plt.plot(range(len(psnr_values_lami360)), psnr_values_lami360, label="lami360")
#plt.plot(range(len(psnr_values_lami180)), psnr_values_lami180, label="lami180")
#plt.plot(range(len(psnr_values_lami90)), psnr_values_lami90, label="lami90")
#plt.plot(range(len(psnr_values_lami45)), psnr_values_lami45, label="lami45")

##comparison number of rays
#plt.plot(range(len(psnr_values_lami360)), psnr_values_lami360, label="lami360")
#plt.plot(range(len(psnr_values_1000samples)), psnr_values_1000samples, label="1000samples")
#plt.plot(range(len(psnr_values_3000rays)), psnr_values_3000rays, label="3000rays")
#plt.plot(range(len(psnr_values_1000samples3000rays)), psnr_values_1000samples3000rays, label="both")

##Comparison epochs
#plt.plot(range(len(psnr_values_lami360)), psnr_values_lami360, label="lami360_500epochs")
#plt.plot(range(len(psnr_values_lami360_250epochs)), psnr_values_lami360_250epochs, label="lami360_250epochs")

plt.plot(range(len(psnr_values_lami360)), psnr_values_1000samples3000rays_250epochs, label="bot_250epcohs")
plt.plot(range(len(psnr_values_lami360_250epochs)), psnr_values_1000samples3000rays, label="1000samples_3000rays")

plt.xlabel("Slice Index (Z-axis)")
plt.ylabel("PSNR (dB)")
plt.title("PSNR for Each Slice Along the Z-Axis")
plt.legend()
plt.grid()
plt.show()



