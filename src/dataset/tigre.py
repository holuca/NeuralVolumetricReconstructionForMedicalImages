import torch
import pickle
import os
import sys
import numpy as np

from torch.utils.data import DataLoader, Dataset


class ConeGeometry(object):
    """
    Cone beam CT geometry. Note that we convert to meter from millimeter.
    """
    def __init__(self, data):

        # VARIABLE                                          DESCRIPTION                    UNITS
        # -------------------------------------------------------------------------------------
        #self.DSD = data["DSD"]/1000 # Distance Source Detector      (m) 
        self.DSD = sys.maxsize/1000 # Distance Source Detector      (m)                                        ----> Detector to Origin Distance same                           
        self.DSO = data["DSO"]/1000  # Distance Source Origin        (m)                                    ----> infinity/large value to simluate parallel X-rays=?
        # Detector parameters
        self.nDetector = np.array(data["nDetector"])  # number of pixels              (px)
        self.nDetector = np.array([512, 512])
        #self.dDetector = np.array(data["dDetector"])/1000  # size of each pixel            (m)
        self.dDetector = np.array([0.0007,0.0007])
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
        # Mode
        #self.mode = data["mode"]  # parallel, cone                ...
        self.mode = 'parallel'
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
        self.tilt_angle = data.get("tilt_angle", 0)
    
        if type == "train":
            self.projs = torch.tensor(data["train"]["projections"], dtype=torch.float32, device=device)
            angles = data["train"]["angles"]
            rays = self.get_rays(angles, self.tilt_angle, self.geo, device)
            self.rays = torch.cat([rays, torch.ones_like(rays[...,:1])*self.near, torch.ones_like(rays[...,:1])*self.far], dim=-1)
            self.n_samples = data["numTrain"]
            coords = torch.stack(torch.meshgrid(torch.linspace(0, self.geo.nDetector[1] - 1, self.geo.nDetector[1], device=device),
                                                torch.linspace(0, self.geo.nDetector[0] - 1, self.geo.nDetector[0], device=device), indexing="ij"), -1)
            self.coords = torch.reshape(coords, [-1, 2])
            self.image = torch.tensor(data["image"], dtype=torch.float32, device=device)
            self.voxels = torch.tensor(self.get_voxels(self.geo), dtype=torch.float32, device=device)
        elif type == "val":
            self.projs = torch.tensor(data["val"]["projections"], dtype=torch.float32, device=device)
            angles = data["val"]["angles"]
            rays = self.get_rays(angles, self.tilt_angle, self.geo, device)
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
    

    def get_rays1(self, angles, geo, device):
        """
        Custom method to calculate rays for laminography using TIGRE geometry but applying a tilt.
        """
        W, H = geo.nDetector  # Detector size (width and height)
        DSD = geo.DSD         # Distance source to detector
        DSO = geo.DSO         # Distance source to object
        
        tilt_angle = self.tilt_angle  # Tilt angle (specified during initialization)
        
        # Tilt matrix for laminography
        tilt_matrix = torch.tensor([[1, 0, 0],
                                    [0, np.cos(tilt_angle), -np.sin(tilt_angle)],
                                    [0, np.sin(tilt_angle), np.cos(tilt_angle)]], device=device)
        
        rays = []
        for angle in angles:
            pose = torch.tensor(self.angle2pose(DSO, angle)).to(device)  # Camera pose matrix based on angle
            
            rays_o, rays_d = None, None
            if geo.mode == "cone":
                # Generate rays for cone-beam setup (TIGRE-based)
                i, j = torch.meshgrid(torch.linspace(0, W - 1, W, device=device),
                                      torch.linspace(0, H - 1, H, device=device), indexing="ij")
                uu = (i.t() + 0.5 - W / 2) * geo.dDetector[0] + geo.offDetector[0]
                vv = (j.t() + 0.5 - H / 2) * geo.dDetector[1] + geo.offDetector[1]
                dirs = torch.stack([uu / DSD, vv / DSD, torch.ones_like(uu)], -1)
                
                # Apply tilt to the ray directions
                dirs = torch.matmul(dirs, tilt_matrix.T)
                
                rays_d = torch.sum(torch.matmul(pose[:3, :3], dirs[..., None]).to(device), -1)
                rays_o = pose[:3, -1].expand(rays_d.shape)  # Origin stays at source position

            elif geo.mode == "parallel":
                # Generate rays for parallel-beam setup (TIGRE-based)
                i, j = torch.meshgrid(torch.linspace(0, W - 1, W, device=device),
                                      torch.linspace(0, H - 1, H, device=device), indexing="ij")
                uu = (i.t() + 0.5 - W / 2) * geo.dDetector[0] + geo.offDetector[0]
                vv = (j.t() + 0.5 - H / 2) * geo.dDetector[1] + geo.offDetector[1]
                dirs = torch.stack([torch.zeros_like(uu), torch.zeros_like(uu), torch.ones_like(uu)], -1)
                
               

                # Apply tilt to the ray directions
                dirs = torch.matmul(dirs, tilt_matrix.T)
                rays_d = torch.sum(torch.matmul(pose[:3, :3], dirs[..., None]).to(device), -1)
                rays_o = torch.sum(torch.matmul(pose[:3, :3], torch.stack([uu, vv, torch.zeros_like(uu)], -1)[..., None]).to(device), -1) + pose[:3, -1].expand(rays_d.shape)
            
            else:
                raise NotImplementedError("Unknown scanner type!")
            
            # Append rays for each angle
            rays.append(torch.cat([rays_o, rays_d], dim=-1))
        
        return torch.stack(rays, dim=0)

    
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
            elif geo.mode == "parallel":
                i, j = torch.meshgrid(torch.linspace(0, W - 1, W, device=device),
                                        torch.linspace(0, H - 1, H, device=device), indexing="ij")  # pytorch"s meshgrid has indexing="ij"
                uu = (i.t() + 0.5 - W / 2) * geo.dDetector[0] + geo.offDetector[0]
                vv = (j.t() + 0.5 - H / 2) * geo.dDetector[1] + geo.offDetector[1]
                dirs = torch.stack([torch.zeros_like(uu), torch.zeros_like(uu), torch.ones_like(uu)], -1)
                rays_d = torch.sum(torch.matmul(pose[:3,:3], dirs[..., None]).to(device), -1) # pose[:3, :3] * 
                rays_o = torch.sum(torch.matmul(pose[:3,:3], torch.stack([uu,vv,torch.zeros_like(uu)],-1)[..., None]).to(device), -1) + pose[:3, -1].expand(rays_d.shape)


                # import open3d as o3d
                # from src.util.draw_util import plot_rays, plot_cube, plot_camera_pose
                # cube1 = plot_cube(np.zeros((3,1)), geo.sVoxel[...,np.newaxis])
                # cube2 = plot_cube(np.zeros((3,1)), np.ones((3,1))*geo.DSO*2)
                # rays1 = plot_rays(rays_d.cpu().detach().numpy(), rays_o.cpu().detach().numpy(), 2)
                # poseray = plot_camera_pose(pose.cpu().detach().numpy())
                # o3d.visualization.draw_geometries([cube1, cube2, rays1, poseray])
            
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
        R4_y = np.array([[np.cos(tilt_angle), 0.0, np.sin(tilt_angle)],
                    [0.0, 0.0, 1.0],
                    [-np.sin(tilt_angle), 0.0, np.cos(tilt_angle)]])

        R4 = np.array([[1, 0, 0],
                   [0, np.cos(tilt_angle), -np.sin(tilt_angle)],
                   [0, np.sin(tilt_angle), np.cos(tilt_angle)]])

        

        #translation vector T places x-ray source at a distance DSO from the centerof the object
        #source is rotated around the object asthe angle changes, simulation rotation during the tomography scan.
        rot = np.dot(np.dot(np.dot(R4, R3), R2), R1)
        trans = np.array([DSO * np.cos(angle), DSO * np.sin(angle), 0])
        T = np.eye(4)
        T[:-1, :-1] = rot
        T[:-1, -1] = trans
        return T


    def angle2pose1(self, DSO, angle, tilt_angle):
        phi1 = -np.pi / 2
        R1 = np.array([[1.0, 0.0, 0.0],
                    [0.0, np.cos(phi1), -np.sin(phi1)],
                    [0.0, np.sin(phi1), np.cos(phi1)]])
        phi2 = np.pi / 2
        R2 = np.array([[np.cos(phi2), -np.sin(phi2), 0.0],
                    [np.sin(phi2), np.cos(phi2), 0.0],
                    [0.0, 0.0, 1.0]])
        R3 = np.array([[np.cos(angle), -np.sin(angle), 0.0],
                    [np.sin(angle), np.cos(angle), 0.0],
                    [0.0, 0.0, 1.0]])
        rot = np.dot(np.dot(R3, R2), R1)
        trans = np.array([DSO * np.cos(angle), DSO * np.sin(angle), 0])
        T = np.eye(4)
        T[:-1, :-1] = rot
        T[:-1, -1] = trans
        return T

    def get_near_far(self, geo: ConeGeometry, tolerance=0.005):
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
        return near, far


''' DATA SHOULD LOOK SMH like this
data = {
    "DSD": 1500,  # Example distance source-detector
    "DSO": 1000,  # Example distance source-origin
    "nDetector": [512, 512],  # Example detector resolution
    "dDetector": [0.8, 0.8],  # Example detector pixel size
    "nVoxel": [256, 256, 256],  # Example voxel resolution
    "dVoxel": [1, 1, 1],  # Example voxel size
    "offOrigin": [0, 0, 0],  # Example image offset
    "offDetector": [0, 0, 0],  # Example detector offset
    "accuracy": 0.5,  # Example accuracy
    "mode": "cone",  # Cone-beam CT
    "filter": "ram-lak",  # Example filter
    "tilt_angle": 20,  # 20 degree tilt for laminography
}

'''