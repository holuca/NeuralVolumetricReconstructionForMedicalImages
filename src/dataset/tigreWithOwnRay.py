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
        self.dDetector = np.array(data["dDetector"])/1000  # size of each pixel            (m)
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

        #self.magnification = 1
        self.tilt_angle = data.get("tilt_angle", 0) #default 20


class CustomDataset(Dataset):
    """
    Custom dataset for laminography (non-TIGRE version).
    """
    def __init__(self, data, n_rays=1024, type="train", device="cuda"):
        super().__init__()

        # Set up the geometry using your custom data (you'll need to pass the geometry info)
        self.geo = self.create_geometry(data)  # Assume this method creates geometry similar to TIGRE
        self.type = type
        self.n_rays = n_rays
        self.near, self.far = self.get_near_far(self.geo)  # Function to compute near and far plane
    
        # Add tilt angle (which should be part of your custom data)
        self.tilt_angle = data.get("tilt_angle", 0)  # default to 0 if not provided

        if type == "train":
            self.projs = torch.tensor(data["train"]["projections"], dtype=torch.float32, device=device)
            angles = data["train"]["angles"]
            
            # Use custom get_rays method instead of TIGRE get_rays
            rays = self.get_rays(angles, self.geo, device)
            
            self.rays = torch.cat([rays, torch.ones_like(rays[...,:1]) * self.near, torch.ones_like(rays[...,:1]) * self.far], dim=-1)
            self.n_samples = data["numTrain"]
            
            # Coordinates of the detector
            coords = torch.stack(torch.meshgrid(
                torch.linspace(0, self.geo.nDetector[1] - 1, self.geo.nDetector[1], device=device),
                torch.linspace(0, self.geo.nDetector[0] - 1, self.geo.nDetector[0], device=device), 
                indexing="ij"),
                -1)
            self.coords = torch.reshape(coords, [-1, 2])
            
            self.image = torch.tensor(data["image"], dtype=torch.float32, device=device)
            self.voxels = torch.tensor(self.get_voxels(self.geo), dtype=torch.float32, device=device)
        elif type == "val":
            self.projs = torch.tensor(data["val"]["projections"], dtype=torch.float32, device=device)
            angles = data["val"]["angles"]
            
            # Use custom get_rays method instead of TIGRE get_rays
            rays = self.get_rays(angles, self.geo, device)
            
            self.rays = torch.cat([rays, torch.ones_like(rays[...,:1]) * self.near, torch.ones_like(rays[...,:1]) * self.far], dim=-1)
            self.n_samples = data["numVal"]
            self.image = torch.tensor(data["image"], dtype=torch.float32, device=device)
            self.voxels = torch.tensor(self.get_voxels(self.geo), dtype=torch.float32, device=device)
    
    def get_rays(self, angles, geo, device):
        """
        Custom method to calculate rays for laminography without TIGRE.
        """
        W, H = geo.nDetector
        DSD = geo.DSD
        tilt_angle = self.tilt_angle  # Include the tilt angle
        
        # Tilt matrix for laminography ray adjustments
        tilt_matrix = torch.Tensor([[1, 0, 0],
                                    [0, np.cos(tilt_angle), -np.sin(tilt_angle)],
                                    [0, np.sin(tilt_angle), np.cos(tilt_angle)]], device=device)
        
        rays = []
        
        for angle in angles:
            pose = torch.Tensor(self.angle2pose(geo.DSO, angle)).to(device)
            rays_o, rays_d = None, None
            if geo.mode == "cone":
                # Detector grid
                i, j = torch.meshgrid(torch.linspace(0, W - 1, W, device=device),
                                      torch.linspace(0, H - 1, H, device=device), indexing="ij")
                uu = (i.t() + 0.5 - W / 2) * geo.dDetector[0] + geo.offDetector[0]
                vv = (j.t() + 0.5 - H / 2) * geo.dDetector[1] + geo.offDetector[1]
                dirs = torch.stack([uu / DSD, vv / DSD, torch.ones_like(uu)], -1)
                
                # Apply tilt to the directions
                dirs = torch.matmul(dirs, tilt_matrix.T)
                
                rays_d = torch.sum(torch.matmul(pose[:3, :3], dirs[..., None]).to(device), -1)
                rays_o = pose[:3, -1].expand(rays_d.shape)
            elif geo.mode == "parallel":
                i, j = torch.meshgrid(torch.linspace(0, W - 1, W, device=device),
                                      torch.linspace(0, H - 1, H, device=device), indexing="ij")
                uu = (i.t() + 0.5 - W / 2) * geo.dDetector[0] + geo.offDetector[0]
                vv = (j.t() + 0.5 - H / 2) * geo.dDetector[1] + geo.offDetector[1]
                dirs = torch.stack([torch.zeros_like(uu), torch.zeros_like(uu), torch.ones_like(uu)], -1)
                
                # Apply tilt to the directions
                dirs = torch.matmul(dirs, tilt_matrix.T)
                
                rays_d = torch.sum(torch.matmul(pose[:3, :3], dirs[..., None]).to(device), -1)
                rays_o = torch.sum(torch.matmul(pose[:3, :3], torch.stack([uu,vv,torch.zeros_like(uu)],-1)[..., None]).to(device), -1) + pose[:3, -1].expand(rays_d.shape)
            else:
                raise NotImplementedError("Unknown CT scanner type!")
            rays.append(torch.cat([rays_o, rays_d], dim=-1))
        
        return torch.stack(rays, dim=0)






def get_voxels(nVoxel, dVoxel):
    """
    get the voxels.
        """
    n1, n2, n3 = nVoxel #number of voxels along xyz
    s1, s2, s3 = s1, s2, s3 = (n1 * dVoxel[0]) / 2, (n2 * dVoxel[1]) / 2, (n3 * dVoxel[2]) / 2

    xyz = np.meshgrid(np.linspace(-s1, s1, n1),
                        np.linspace(-s2, s2, n2),
                        np.linspace(-s3, s3, n3), indexing="ij")
    voxel = np.asarray(xyz).transpose([1, 2, 3, 0])
    #voxel = np.stack(xyz, axis=-1)  # Shape: (n1, n2, n3, 3)
    return voxel
    


def get_rays(angles, tilt_angle, mode, nDetector, dDetector, DSO, DSD):
    """
    Get rays given one angle and x-ray machine geometry.
    """
    W, H = nDetector
    #pixel position to units
    uu = (uu + 0.5 - W/2) * dDetector[0]
    vv = (vv + 0.5 - H/2) * dDetector[1]
    rays = []
    
    for angle in angles:
        R = angle2pose(angle, tilt_angle)

        if mode == "parallel":
            # Parallel rays: All directions are along z-axis
            dirs = np.array([np.zeros_like(uu), np.zeros_like(vv), np.ones_like(uu)]).T
            # Origins for the rays are on the detector plane (no divergence)
            origins = np.dot(R, np.stack([uu, vv, np.zeros_like(uu)], axis=-1).T)
        

            # import open3d as o3d
            # from src.util.draw_util import plot_rays, plot_cube, plot_camera_pose
            # cube1 = plot_cube(np.zeros((3,1)), geo.sVoxel[...,np.newaxis])
            # cube2 = plot_cube(np.zeros((3,1)), np.ones((3,1))*geo.DSO*2)
            # rays1 = plot_rays(rays_d.cpu().detach().numpy(), rays_o.cpu().detach().numpy(), 2)
            # poseray = plot_camera_pose(pose.cpu().detach().numpy())
            # o3d.visualization.draw_geometries([cube1, cube2, rays1, poseray])
            
        else:
            raise NotImplementedError("Unknown CT scanner type!")
        rays.append(origins, dirs)

    return np.array(rays)


def angle2pose(angle, tilt_angle):
    # Rotation around z-axis for object rotation
    Rz = np.array([[np.cos(angle), -np.sin(angle), 0],
                   [np.sin(angle), np.cos(angle), 0],
                   [0, 0, 1]])

    # Rotation around x-axis for tilt (laminography)
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(tilt_angle), -np.sin(tilt_angle)],
                   [0, np.sin(tilt_angle), np.cos(tilt_angle)]])
    
    return np.dot(Rx, Rz)  # Combine rotations
    

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

def project_rays(voxel_grid, rays):
    """
    Compute the line integrals of the rays through the voxel grid.
    """
    projections = []
    for origins, directions in rays:
        # Implement the ray-voxel intersection algorithm (e.g., Siddon’s method)
        # For simplicity, let's assume a basic approach here.
        # Ray tracing through the grid to compute line integrals
        for ray_origin, ray_dir in zip(origins, directions):
            projection_value = ray_trace(voxel_grid, ray_origin, ray_dir)
            projections.append(projection_value)
    
    return np.array(projections)

def ray_trace(voxel_grid, ray_origin, ray_dir):
    """
    Basic ray tracing function through the voxel grid. Replace with more
    advanced algorithms like Siddon's method for efficiency.
    """
    # Placeholder for ray tracing algorithm
    # For each voxel, compute intersection length and sum up the contributions
    return np.random.random()  # Random value as a placeholder

def tomography_simulation(angles, tilt_angle, mode, nVoxel, dVoxel, nDetector, dDetector, DSO, DSD):
    # Generate the voxel grid
    voxel_grid = get_voxels(nVoxel, dVoxel)

    # Get the rays (origins and directions) based on the current angles and geometry
    rays = get_rays(angles, tilt_angle, mode, nDetector, dDetector, DSO, DSD)

    # Perform the ray projection through the voxel grid (compute projections)
    projections = project_rays(voxel_grid, rays)

    return projections




#example usage
# Example parameters
angles = np.linspace(0, np.pi, 180)  # 180 projection angles
tilt_angle = np.radians(10)  # 10 degree tilt for laminography
mode = "parallel"  # or "cone"
nVoxel = [256, 256, 256]  # 3D grid resolution
dVoxel = [1, 1, 1]  # Voxel size
nDetector = [512, 512]  # Detector resolution
dDetector = [0.8, 0.8]  # Detector pixel size
DSO = 1000  # Source to object distance (large for parallel rays)
DSD = 1500  # Source to detector distance (if needed)

# Simulate tomography
projections = tomography_simulation(angles, tilt_angle, mode, nVoxel, dVoxel, nDetector, dDetector, DSO, DSD)
