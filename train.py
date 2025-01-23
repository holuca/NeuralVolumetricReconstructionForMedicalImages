import os
import os.path as osp
import torch
import imageio.v2 as iio
import numpy as np
import argparse

import matplotlib
matplotlib.use("Agg")  # Use a non-interactive backend
import matplotlib.pyplot as plt

from tqdm import tqdm
from src.config.configloading import load_config
from src.render import render, run_network
from src.trainer import Trainer
from src.loss import calc_mse_loss, calc_phase_only_loss
from src.utils import get_psnr, get_mse, get_psnr_3d, get_ssim_3d, cast_to_image, get_ptycho_mask, visualize_after_mask, visualize_sampled_points

def config_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./config/abdomen_50.yaml",
                        help="configs file path")
    return parser

parser = config_parser()
args = parser.parse_args()

cfg = load_config(args.config)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




class BasicTrainer(Trainer):
    def __init__(self):
        """
        Basic network trainer.
        """
        self.device = device
        super().__init__(cfg, device)
        print(f"[Start] exp: {cfg['exp']['expname']}, net: Basic network")
    




    def compute_loss(self, data, global_step, idx_epoch):
        """
        Compute loss using phase or magnitude information dynamically, with mask integration.
        """
        rays = data["rays"].reshape(-1, 8).to(device)
        projs = data["projs"].reshape(-1).to(device)  # Complex projections
        full_proj = data.get("full_proj")  # Full projection (256x356) if available
    
        chunk_size = 200
        loss = {"loss": 0.0}  # Initialize loss dictionary
        # Generate the mask for the full projection
        if full_proj is not None:
            mask_full = get_ptycho_mask(full_proj, threshold=0.007).float().squeeze().to(projs.device)#move singleton dimensions
            

            # Visualize the full mask
            #plt.imshow(mask_full.cpu().numpy(), cmap='gray')
            #plt.title('Full Mask (mask_full)')
            #plt.colorbar()
            #plt.savefig(f'mask_full_step_{global_step}.png')
            #plt.close()
        for i in range(0, rays.shape[0], chunk_size):
            rays_chunk = rays[i:i + chunk_size]
            projs_chunk = projs[i:i + chunk_size]



            ret = render(
                rays_chunk,
                self.net,
                self.net_fine,
                n_samples=self.conf["render"]["n_samples"],
                n_fine=self.conf["render"]["n_fine"],
                perturb=self.conf["render"]["perturb"],
                netchunk=self.conf["render"]["netchunk"],
                raw_noise_std=self.conf["render"]["raw_noise_std"],
                chunk_size=chunk_size
            )

            # Predicted phase directly as real numbers from render
            projs_pred_chunk = ret["acc"].reshape(-1)

            # Apply the mask to the sampled projections and predictions
            if full_proj is not None:
                # Get the sampled mask corresponding to the sampled projections
                sampled_coords = data["coords"][i:i + chunk_size].squeeze()  # Remove extra dimensions

                mask_sampled = mask_full[sampled_coords[:, 0], sampled_coords[:, 1]]
                #visualize_sampled_points(
                #    full_mask=mask_full,
                #    sampled_coords=sampled_coords,
                #    mask_sampled=mask_sampled,
                #    global_step=global_step
                #)


                projs_pred_masked = projs_pred_chunk * mask_sampled
                projs_masked = projs_chunk * mask_sampled

                # Visualize after mask application
                #visualize_after_mask(
                #    full_mask=mask_full,
                #    sampled_coords=sampled_coords,
                #    projs_values=projs_chunk_masked,  # or projs_pred_chunk
                #    global_step=global_step,
                #    title_suffix="_phase"  # Suffix to indicate the type of values being visualized
                #)
            #
                #visualize_after_mask(
                #    full_mask=mask_full,
                #    sampled_coords=sampled_coords,
                #    projs_values=projs_pred_masked,  # Visualize predictions after masking
                #    global_step=global_step,
                #    title_suffix="_pred"
                #)


            # Compute MSE loss using masked phase information
            #projs_chunk, projs_pred_chunk = pad_to_match(projs_chunk.float(), projs_pred_chunk.float())
            calc_mse_loss(loss, projs_chunk[mask_sampled.bool()], projs_pred_chunk[mask_sampled.bool()])
        
            #calc_mse_loss(loss, projs_chunk, projs_pred_chunk)

        # Log loss
        for ls in loss.keys():
            self.writer.add_scalar(f"train/{ls}", loss[ls].item(), global_step)

        return loss["loss"]


    def eval_stepMASK(self, global_step, idx_epoch):
        """
        Evaluation step
        """
        print(f"Starting evaluation at epoch {idx_epoch}...")
        progress = tqdm(total=len(self.eval_dset), desc="Evaluating")
        select_ind = np.random.choice(len(self.eval_dset))
        
        # Ground truth projections and rays
        projs = self.eval_dset.projs[select_ind].to(self.device).to(torch.complex64)  # Move to device
        rays = self.eval_dset.rays[select_ind].reshape(-1, 8).to(self.device)  # Move to device
        H, W = projs.shape
    
        # Full projection and mask
        #mask = get_ptycho_mask(full_proj, threshold=0.007).float().to(self.device)
    
        # Predicted projections
        projs_pred = []
        for i in range(0, rays.shape[0], self.n_rays):
            projs_pred.append(
                render(rays[i:i + self.n_rays], self.net, self.net_fine, **self.conf["render"], chunk_size=1024)["acc"]
            )
            progress.update(1)
        projs_pred = torch.cat(projs_pred, 0).reshape(H, W).to(self.device).to(torch.complex64)  # Ensure complex dtype
        progress.close()
        print(f"Completed evaluation at epoch {idx_epoch}.")
    
        # Apply the mask to ground truth and predictions
        #projs_masked = projs * mask
        #projs_pred_masked = projs_pred * mask
    
        # Evaluate density
        image = self.eval_dset.image.to(self.device)  # Move to device
        image_pred = run_network(
            self.eval_dset.voxels.to(self.device),  # Move to device
            self.net_fine if self.net_fine is not None else self.net,
            self.netchunk
        )
        image_pred = image_pred.squeeze()
    
        # Compute metrics with masked projections
        loss = {
            "proj_mse": get_mse(projs_pred, projs),  # Masked MSE
            "proj_psnr": get_psnr(projs_pred, projs),  # Masked PSNR
            "psnr_3d": get_psnr_3d(image_pred, image),  # No mask needed for 3D metrics
            "ssim_3d": get_ssim_3d(image_pred, image),  # No mask needed for 3D metrics
        }
    
        # Logging
        show_slice = 5
        show_step = image.shape[-1] // show_slice
        show_image = image[..., ::show_step]
        show_image_pred = image_pred[..., ::show_step]
        show = []
        for i_show in range(show_slice):
            show.append(torch.concat([show_image[..., i_show], show_image_pred[..., i_show]], dim=0))
        show_density = torch.concat(show, dim=1)
        #show_proj = torch.concat([projs_masked, projs_pred_masked], dim=1)
        show_proj = torch.concat([projs, projs_pred], dim=1)
        self.writer.add_image("eval/density (row1: gt, row2: pred)", cast_to_image(show_density), global_step, dataformats="HWC")
        self.writer.add_image("eval/projection (left: gt, right: pred)", cast_to_image(show_proj), global_step, dataformats="HWC")
    
        for ls in loss.keys():
            self.writer.add_scalar(f"eval/{ls}", loss[ls], global_step)
            
        # Save
        eval_save_dir = osp.join(self.evaldir, f"epoch_{idx_epoch:05d}")
        os.makedirs(eval_save_dir, exist_ok=True)
        np.save(osp.join(eval_save_dir, "image_pred.npy"), image_pred.cpu().detach().numpy())
        np.save(osp.join(eval_save_dir, "image_gt.npy"), image.cpu().detach().numpy())
        iio.imwrite(osp.join(eval_save_dir, "slice_show_row1_gt_row2_pred.png"), (cast_to_image(show_density)*255).astype(np.uint8))
        iio.imwrite(osp.join(eval_save_dir, "proj_show_left_gt_right_pred.png"), (cast_to_image(show_proj)*255).astype(np.uint8))
        with open(osp.join(eval_save_dir, "stats.txt"), "w") as f: 
            for key, value in loss.items(): 
                f.write("%s: %f\n" % (key, value.item()))
    
        return loss





    def eval_step(self, global_step, idx_epoch):
        """
        Evaluation step
        """
        # Evaluate projection
        print(f"Starting evaluation at epoch {idx_epoch}...")
        progress = tqdm(total=len(self.eval_dset), desc="Evaluating")
        select_ind = np.random.choice(len(self.eval_dset))
        projs = self.eval_dset.projs[select_ind].to(self.device).to(torch.complex64)  # Move to device
        rays = self.eval_dset.rays[select_ind].reshape(-1, 8).to(self.device)  # Move to device
        H, W = projs.shape

        #full_proj = self.eval_dset.full_proj[select_ind].to(self.device)
        #mask = get_ptycho_mask(full_proj, threshold=0.007).float().to(device)
        projs_pred = []
        for i in range(0, rays.shape[0], self.n_rays):
            projs_pred.append(
                render(rays[i:i+self.n_rays], self.net, self.net_fine, **self.conf["render"], chunk_size=1024)["acc"]
            )
            progress.update(1)
        projs_pred = torch.cat(projs_pred, 0).reshape(H, W).to(self.device).to(torch.complex64)  # Ensure complex dtype
        progress.close()
        print(f"Completed evaluation at epoch {idx_epoch}.")

        # Evaluate density
        image = self.eval_dset.image.to(self.device)  # Move to device
        image_pred = run_network(
            self.eval_dset.voxels.to(self.device),  # Move to device
            self.net_fine if self.net_fine is not None else self.net,
            self.netchunk
        )
        image_pred = image_pred.squeeze()
        #apply mask to projections and predictions if needed
        loss = {
            "proj_mse": get_mse(projs_pred, projs),  # Ensure both are on the same device
            "proj_psnr": get_psnr(projs_pred, projs),  # Ensure both are on the same device
            "psnr_3d": get_psnr_3d(image_pred, image),  # Ensure both are on the same device
            "ssim_3d": get_ssim_3d(image_pred, image),  # Ensure both are on the same device
        }

        # Logging
        show_slice = 5
        show_step = image.shape[-1]//show_slice
        show_image = image[...,::show_step]
        show_image_pred = image_pred[...,::show_step]
        show = []
        for i_show in range(show_slice):
            show.append(torch.concat([show_image[..., i_show], show_image_pred[..., i_show]], dim=0))
        show_density = torch.concat(show, dim=1)
        show_proj = torch.concat([projs, projs_pred], dim=1)

        self.writer.add_image("eval/density (row1: gt, row2: pred)", cast_to_image(show_density), global_step, dataformats="HWC")
        self.writer.add_image("eval/projection (left: gt, right: pred)", cast_to_image(show_proj), global_step, dataformats="HWC")

        for ls in loss.keys():
            self.writer.add_scalar(f"eval/{ls}", loss[ls], global_step)
            
        # Save
        eval_save_dir = osp.join(self.evaldir, f"epoch_{idx_epoch:05d}")
        os.makedirs(eval_save_dir, exist_ok=True)
        np.save(osp.join(eval_save_dir, "image_pred.npy"), image_pred.cpu().detach().numpy())
        np.save(osp.join(eval_save_dir, "image_gt.npy"), image.cpu().detach().numpy())
        iio.imwrite(osp.join(eval_save_dir, "slice_show_row1_gt_row2_pred.png"), (cast_to_image(show_density)*255).astype(np.uint8))
        iio.imwrite(osp.join(eval_save_dir, "proj_show_left_gt_right_pred.png"), (cast_to_image(show_proj)*255).astype(np.uint8))
        with open(osp.join(eval_save_dir, "stats.txt"), "w") as f: 
            for key, value in loss.items(): 
                f.write("%s: %f\n" % (key, value.item()))

        return loss



trainer = BasicTrainer()
trainer.start()