import torch
import pickle
import os
import sys
import numpy as np
import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset


class ConeGeometry(object):
    """
    Cone beam CT geometry. Note that we convert to meter from millimeter.
    """
    def __init__(self, data):

        # VARIABLE                                          DESCRIPTION                    UNITS
        # -------------------------------------------------------------------------------------
        self.DSD = data["DSD"]/1000 # Distance Source Detector      (m) 
        self.DSO = data["DSO"]/1000  # Distance Source Origin        (m)  (to inf for parallel)
        # Detector parameters
        self.nDetector = np.array(data["nDetector"])  # number of pixels              (px)
        self.dDetector = np.array(data["dDetector"])/1000  # size of each pixel      (m)    --> sqrt 2 as for parallel beam diagonal needs to be included

        self.sDetector = self.nDetector * self.dDetector  # total size of the detector    (m)
        # Image parameters
        self.nVoxel = np.array(data["nVoxel"])  # number of voxels              (vx)
        #each voxel represents a cube of x mm
        self.dVoxel = np.array(data["dVoxel"])/1000  # size of each voxel            (m)
        self.sVoxel = self.nVoxel * self.dVoxel  # total size of the image       (m)

        # Offsets
        self.offOrigin = np.array(data["offOrigin"])/1000  # Offset of image from origin   (m)
        self.offDetector = np.array(data["offDetector"])/1000  # Offset of Detector            (m)

        # Auxiliary
        self.accuracy = data["accuracy"]  # Accuracy of FWD proj          (vx/sample)  # noqa: E501
        self.mode = data["mode"] # Mode parallel, cone    
        self.filter = data["filter"]

        self.magnification = 1
        self.tilt_angle = data.get("tilt_angle", 0) #default 0


class TIGREDataset(Dataset):
    """
    TIGRE dataset.
    """
    def __init__(self, path, n_rays=1024, type="train", device="cuda"):    
        super().__init__()

        with open(path, "rb") as handle:
            data = pickle.load(handle)
        
        self.geo = ConeGeometry(data)
        self.type = type
        self.n_rays = n_rays
        self.near, self.far = self.get_near_far(self.geo)

        

        if type == "train":
            self.projs = torch.tensor(data["train"]["projections"], dtype=torch.float32, device=device)
            print("Shape of the first projection of train function ", self.projs[0].shape)
            #if resizing of nDetector is done manually in ConeGemoetry, use this interpolation
            #print("Shape of the first projection of train function after rehsaping", self.projs[0].shape)
            self.projs = self.projs.unsqueeze(1)  # Shape now: (360, 1, 175, 128)
            # Resize using F.interpolate
            self.projs = F.interpolate(self.projs, size=(self.geo.nDetector[1], self.geo.nDetector[0]), mode='bilinear', align_corners=False)
            # Remove the channel dimension
            self.projs = self.projs.squeeze(1)  # Shape should now be (360, nDetector[0], nDetector[1])
            angles = data["train"]["angles"]
            rays = self.get_rays(angles, self.geo.tilt_angle, self.geo, device)
            self.rays = torch.cat([rays, torch.ones_like(rays[...,:1])*self.near, torch.ones_like(rays[...,:1])*self.far], dim=-1)
            self.n_samples = data["numTrain"]
            coords = torch.stack(torch.meshgrid(torch.linspace(0, self.geo.nDetector[1] - 1, self.geo.nDetector[1], device=device),
                                                torch.linspace(0, self.geo.nDetector[0] - 1, self.geo.nDetector[0], device=device), indexing="ij"), -1)
            self.coords = torch.reshape(coords, [-1, 2])
            self.image = torch.tensor(data["image"], dtype=torch.float32, device=device)
            self.voxels = torch.tensor(self.get_voxels(self.geo), dtype=torch.float32, device=device)

        elif type == "val":
            self.projs = torch.tensor(data["val"]["projections"], dtype=torch.float32, device=device)
            #if resizing of nDetector is done manually in ConeGemoetry, use this interpolation
            #print("Shape of the first projection of train function after rehsaping", self.projs[0].shape)
            self.projs = self.projs.unsqueeze(1)  # Shape now: (360, 1, 175, 128)
            # Resize using F.interpolate
            self.projs = F.interpolate(self.projs, size=(self.geo.nDetector[1], self.geo.nDetector[0]), mode='bilinear', align_corners=False)
            # Remove the channel dimension
            self.projs = self.projs.squeeze(1)  # Shape should now be (360, nDetector[0], nDetector[1])
            angles = data["val"]["angles"]
            rays = self.get_rays(angles, self.geo.tilt_angle, self.geo, device)
            self.rays = torch.cat([rays, torch.ones_like(rays[...,:1])*self.near, torch.ones_like(rays[...,:1])*self.far], dim=-1)
            self.n_samples = data["numVal"]
            self.image = torch.tensor(data["image"], dtype=torch.float32, device=device)
            self.voxels = torch.tensor(self.get_voxels(self.geo), dtype=torch.float32, device=device)

    

        
    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        if self.type == "train":
            projs_valid = (self.projs[index]>0).flatten()
            coords_valid = self.coords[projs_valid]
            select_inds = np.random.choice(coords_valid.shape[0], size=[self.n_rays], replace=False)
            select_coords = coords_valid[select_inds].long()
            rays = self.rays[index, select_coords[:, 0], select_coords[:, 1]]
            projs = self.projs[index, select_coords[:, 0], select_coords[:, 1]]
            out = {
                "projs":projs,
                "rays":rays,
            }
        elif self.type == "val":
            rays = self.rays[index]
            projs = self.projs[index]
            out = {
                "projs":projs,
                "rays":rays,
            }
        return out

    def get_voxels(self, geo: ConeGeometry):
        """
        Get the voxels.
        """
        n1, n2, n3 = geo.nVoxel 
        s1, s2, s3 = geo.sVoxel / 2 - geo.dVoxel / 2

        xyz = np.meshgrid(np.linspace(-s1, s1, n1),
                        np.linspace(-s2, s2, n2),
                        np.linspace(-s3, s3, n3), indexing="ij")
        voxel = np.asarray(xyz).transpose([1, 2, 3, 0])

        return voxel
    


    
    def get_rays(self, angles, tilt_angle, geo: ConeGeometry, device):
        """
        Get rays given one angle and x-ray machine geometry.
        """
        W, H = geo.nDetector
        DSD = geo.DSD
        rays = []
        
        for angle in angles:
            pose = torch.Tensor(self.angle2pose(geo.DSO, angle, tilt_angle)).to(device)
            rays_o, rays_d = None, None
            if geo.mode == "cone":
                i, j = torch.meshgrid(torch.linspace(0, W - 1, W, device=device),
                                    torch.linspace(0, H - 1, H, device=device), indexing="ij")  # pytorch"s meshgrid has indexing="ij"
                uu = (i.t() + 0.5 - W / 2) * geo.dDetector[0] + geo.offDetector[0]
                vv = (j.t() + 0.5 - H / 2) * geo.dDetector[1] + geo.offDetector[1]
                dirs = torch.stack([uu / DSD, vv / DSD, torch.ones_like(uu)], -1)
                rays_d = torch.sum(torch.matmul(pose[:3,:3], dirs[..., None]).to(device), -1) # pose[:3, :3] * 
                rays_o = pose[:3, -1].expand(rays_d.shape)

                #import open3d as o3d
                #cube1 = plot_cube(np.zeros((3,1)), geo.sVoxel[...,np.newaxis])
                #cube2 = plot_cube(np.zeros((3,1)), np.ones((3,1))*geo.DSO*2)
                #rays1 = plot_rays(rays_d.cpu().detach().numpy(), rays_o.cpu().detach().numpy(), 2)
                #poseray = plot_camera_pose(pose.cpu().detach().numpy())
                #o3d.visualization.draw_geometries([cube1, cube2, rays1, poseray])
            elif geo.mode == "parallel":

        
                i, j = torch.meshgrid(torch.linspace(0, W - 1, W, device=device),
                                        torch.linspace(0, H - 1, H, device=device), indexing="ij")  # pytorch"s meshgrid has indexing="ij"
                uu = (i.t() + 0.5 - W / 2) * geo.dDetector[0] + geo.offDetector[0]
                vv = (j.t() + 0.5 - H / 2) * geo.dDetector[1] + geo.offDetector[1]
                dirs = torch.stack([torch.zeros_like(uu), torch.zeros_like(uu), torch.ones_like(uu)], -1)
                rays_d = torch.sum(torch.matmul(pose[:3,:3], dirs[..., None]).to(device), -1) # pose[:3, :3] * 
                rays_o = torch.sum(torch.matmul(pose[:3,:3], torch.stack([uu,vv,torch.zeros_like(uu)],-1)[..., None]).to(device), -1) + pose[:3, -1].expand(rays_d.shape)


                ## Print the origin of the center ray and camera pose to compare

                #rays_o_np = rays_o.cpu().detach().numpy()  # Convert to NumPy for printing
                ## Access the central ray origin
                #center_ray_origin = rays_o_np[64, 64]  # Accessing the ray at (64, 64)
                #print(f"Central Ray Origin = (x: {center_ray_origin[0]:.3f}, y: {center_ray_origin[1]:.3f}, z: {center_ray_origin[2]:.3f})")
                #camera_position = pose[:3, -1]  # Extract camera position from the pose
                #print(f"Camera Position:     (x: {camera_position[0]:.3f}, y: {camera_position[1]:.3f}, z: {camera_position[2]:.3f})")
                #center_i, center_j = W // 2, H // 2  # Center pixel indices
                #central_ray_origin = rays_o[center_i, center_j]  # Get the origin of the central ray
                ## Update only the Z-axis of the camera pose
                #pose[2, -1] = central_ray_origin[2]  # Set Z position to the central ray origin's Z
                # Update pose for visualization
                #camera_position = pose[:3, -1] + torch.tensor([0, 0, translation], device=device)
                #camera_position = pose[:3, -1].to(device) + torch.tensor([0, 0, translation], device=device)  # Correctly apply translation

                # Visualization
                #import open3d as o3d
                #cube1 = plot_cube(np.zeros((3, 1)), geo.sVoxel[..., np.newaxis])
                #cube2 = plot_cube(np.zeros((3, 1)), np.ones((3, 1)) * geo.DSO * 2)
                #rays1 = plot_rays(rays_d.cpu().detach().numpy(), rays_o.cpu().detach().numpy(), 2)
                #poseray = plot_camera_pose(pose.cpu().detach().numpy())  # Use updated camera position
                #o3d.visualization.draw_geometries([cube1, cube2, rays1, poseray])

            else:
                raise NotImplementedError("Unknown CT scanner type!")
            rays.append(torch.concat([rays_o, rays_d], dim=-1))

        return torch.stack(rays, dim=0)

    def angle2pose(self, DSO, angle, tilt_angle):
        phi1 = -np.pi / 2
        #counterclockwise around x-axis (90degrees)
        #aligns detector and source palne with the x-ray system
        R1 = np.array([[1.0, 0.0, 0.0],
                    [0.0, np.cos(phi1), -np.sin(phi1)],
                    [0.0, np.sin(phi1), np.cos(phi1)]])
        
        phi2 = np.pi / 2
        #counterclockwise rotation around z-axis (90 degrees)
        # tomography system's defualt orientation
        R2 = np.array([[np.cos(phi2), -np.sin(phi2), 0.0],
                    [np.sin(phi2), np.cos(phi2), 0.0],
                    [0.0, 0.0, 1.0]])
        #counterclockwise rotation around z-axis by the current scanning angle
        R3 = np.array([[np.cos(angle), -np.sin(angle), 0.0],
                    [np.sin(angle), np.cos(angle), 0.0],
                    [0.0, 0.0, 1.0]])
        

        #Add tilt angle (y axis)
        tilt_angle = np.radians(tilt_angle)
    
        R4_x_clockwise = np.array([[1, 0, 0],
                           [0, np.cos(tilt_angle), np.sin(tilt_angle)],
                           [0, -np.sin(tilt_angle), np.cos(tilt_angle)]])

        #translation vector T places x-ray source at a distance DSO from the centerof the object
        #source is rotated around the object as the angle changes, simulation rotation during the tomography scan.
        #rot = np.dot(np.dot(np.dot(R4, R3), R2), R1)
        rot = np.dot(np.dot(R3, R2), R1)
        rot = rot @ R4_x_clockwise

        translation_z = DSO * np.tan(tilt_angle)
        # Translation vector to place the x-ray source at DSO distance, and account for tilt
        trans = np.array([DSO * np.cos(angle), DSO * np.sin(angle), translation_z])


        #trans = np.array([DSO * np.cos(angle), DSO * np.sin(angle), 0]) #when cosidering dome shape
        T = np.eye(4)
        T[:-1, :-1] = rot
        T[:-1, -1] = trans
        return T



    def get_near_far_1(self, geo: ConeGeometry, tolerance=0.005):
        """
        Compute the near and far threshold.
        """
        dist1 = np.linalg.norm([geo.offOrigin[0] - geo.sVoxel[0] / 2, geo.offOrigin[1] - geo.sVoxel[1] / 2])
        dist2 = np.linalg.norm([geo.offOrigin[0] - geo.sVoxel[0] / 2, geo.offOrigin[1] + geo.sVoxel[1] / 2])
        dist3 = np.linalg.norm([geo.offOrigin[0] + geo.sVoxel[0] / 2, geo.offOrigin[1] - geo.sVoxel[1] / 2])
        dist4 = np.linalg.norm([geo.offOrigin[0] + geo.sVoxel[0] / 2, geo.offOrigin[1] + geo.sVoxel[1] / 2])
        dist_max = np.max([dist1, dist2, dist3, dist4])
        near = np.max([0, geo.DSO - dist_max - tolerance])
        far = np.min([geo.DSO * 2, geo.DSO + dist_max + tolerance])
        print("NEAR: ", near)
        print("FAR: ", far)
        return near, far
    
    def get_near_far(self, geo, tolerance=0.005):

        tilt_radians = np.radians(geo.tilt_angle)
        z_tilt_offset = geo.sVoxel[2] * np.abs(np.sin(tilt_radians))

        # Define the four corners of the bounding box relative to `offOrigin`
        dist1 = np.abs(geo.offOrigin[1] - geo.sVoxel[1] / 2 - z_tilt_offset)
        dist2 = np.abs(geo.offOrigin[1] + geo.sVoxel[1] / 2 + z_tilt_offset)

        # Compute min and max distances along the Y-axis
        dist_min = np.min([dist1, dist2])
        dist_max = np.max([dist1, dist2])

        # The near and far planes now depend only on the bounding box extents in parallel-beam
        near = np.max([0, dist_min - tolerance])  # Ensure near is non-negative
        far = dist_max + tolerance

        # Normalize by a characteristic length, such as detector height (sDetector[1])
        near_normalized = near / geo.sDetector[1]
        far_normalized = far / geo.sDetector[1]
        print("NEAR_NJORMALIZED: ", near_normalized)
        print("FASDFASF   ", far_normalized)
        return near_normalized, far_normalized
    