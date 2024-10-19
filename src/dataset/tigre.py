import torch
import pickle
import os
import sys
import numpy as np
import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset



import numpy as np
import open3d as o3d

def plot_rays(ray_directions: np.array, ray_origins: np.array, ray_length: float):
    """
    Plot rays of a scanner (open3d).

    Args:
    ray_directions (np.array(W, H, 3)): ray directions.
    ray_origins (np.array(W, H, 3)): ray origins.
    ray_length (float): ray length.

    Returns:
    lines (o3d.geometry.LineSet): output lines.

    """

    W, H, _ = ray_directions.shape
    ori1 = ray_origins[0, 0, :]
    ori2 = ray_origins[W - 1, 0, :]
    ori3 = ray_origins[W - 1, H - 1, :]
    ori4 = ray_origins[0, H - 1, :]
    end1 = ray_origins[0, 0, :] + ray_directions[0, 0, :] * ray_length
    end2 = ray_origins[W - 1, 0, :] + ray_directions[W - 1, 0, :] * ray_length
    end3 = ray_origins[W - 1, H - 1, :] + ray_directions[W - 1, H - 1, :] * ray_length
    end4 = ray_origins[0, H - 1, :] + ray_directions[0, H - 1, :] * ray_length
    lines = [[0, 4], [1, 5], [2, 6], [3, 7], [4, 5], [5, 6], [6, 7], [7, 4]]
    pts = np.vstack([ori1, ori2, ori3, ori4, end1, end2, end3, end4])
    line_ray = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(pts),
        lines=o3d.utility.Vector2iVector(lines),
    )

    return line_ray

def plot_camera_pose(pose):
    """
    Plot camera pose (open3d).

    Args:
    pose (np.array(4, 4)): camera pose.

    Returns:
    lines (o3d.geometry.LineSet): output lines.

    """

    colorlines = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    origin = np.array([[0], [0], [0], [1]])
    axes = np.array([[1, 0, 0],
                     [0, 1, 0],
                     [0, 0, 1],
                     [1, 1, 1]])
    points = np.vstack([np.transpose(origin), np.transpose(axes)])[:, :-1]
    lines = [[0, 1], [0, 2], [0, 3]]
    worldframe = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    worldframe.colors = o3d.utility.Vector3dVector(colorlines)

    # bbox
    xyz_min = [-0.5, -0.5, -0.5]
    xyz_max = [0.5, 0.5, 0.5]
    points = [[xyz_min[0], xyz_min[1], xyz_min[2]],
              [xyz_max[0], xyz_min[1], xyz_min[2]],
              [xyz_min[0], xyz_max[1], xyz_min[2]],
              [xyz_max[0], xyz_max[1], xyz_min[2]],
              [xyz_min[0], xyz_min[1], xyz_max[2]],
              [xyz_max[0], xyz_min[1], xyz_max[2]],
              [xyz_min[0], xyz_max[1], xyz_max[2]],
              [xyz_max[0], xyz_max[1], xyz_max[2]]]
    lines = [
        [0, 1],
        [0, 2],
        [1, 3],
        [2, 3],
        [4, 5],
        [4, 6],
        [5, 7],
        [6, 7],
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
    ]
    colors = [[1, 0, 0] for i in range(len(lines))]
    line_set_bbox = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set_bbox.colors = o3d.utility.Vector3dVector(colors)

    origin = np.array([[0], [0], [0], [1]])
    unit = 0.3
    axes = np.array([[unit, 0, 0],
                     [0, unit, 0],
                     [0, 0, unit],
                     [1, 1, 1]])
    axes_trans = np.dot(pose, axes)
    origin_trans = np.dot(pose, origin)
    points = np.vstack([np.transpose(origin_trans), np.transpose(axes_trans)])[:, :-1]
    lines = [[0, 1], [0, 2], [0, 3]]
    colorlines = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colorlines)

    return line_set + worldframe

def plot_cube(cube_center: np.array, cube_size: np.array):
    """
    Plot a cube (open3d).

    Args:
    cube_center (np.array(3, 1)): cube center.
    cube_size (np.array(3, 1)): cube size.

    Returns:
    lines (o3d.geometry.LineSet): output lines.

    """

    # coordinate frame
    colorlines = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    origin = np.array([[0], [0], [0], [1]])
    unit = 0.3
    axes = np.array([[unit, 0, 0],
                     [0, unit, 0],
                     [0, 0, unit],
                     [1, 1, 1]]) * np.vstack([np.hstack([cube_size, cube_size, cube_size]), np.ones((1, 3))])
    points = np.vstack([np.transpose(origin), np.transpose(axes)])[:, :-1]
    points += cube_center.squeeze()
    lines = [[0, 1], [0, 2], [0, 3]]
    worldframe = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    worldframe.colors = o3d.utility.Vector3dVector(colorlines)
    # bbox
    xyz_min = cube_center.squeeze() + np.array([-0.5, -0.5, -0.5]) * cube_size.squeeze()
    xyz_max = cube_center.squeeze() + np.array([0.5, 0.5, 0.5]) * cube_size.squeeze()
    points = [[xyz_min[0], xyz_min[1], xyz_min[2]],
              [xyz_max[0], xyz_min[1], xyz_min[2]],
              [xyz_min[0], xyz_max[1], xyz_min[2]],
              [xyz_max[0], xyz_max[1], xyz_min[2]],
              [xyz_min[0], xyz_min[1], xyz_max[2]],
              [xyz_max[0], xyz_min[1], xyz_max[2]],
              [xyz_min[0], xyz_max[1], xyz_max[2]],
              [xyz_max[0], xyz_max[1], xyz_max[2]]]
    lines = [
        [0, 1],
        [0, 2],
        [1, 3],
        [2, 3],
        [4, 5],
        [4, 6],
        [5, 7],
        [6, 7],
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
    ]
    colors = [[1, 0, 0] for i in range(len(lines))]
    line_set_bbox = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set_bbox.colors = o3d.utility.Vector3dVector(colors)
    return line_set_bbox + worldframe



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
        #self.nDetector = np.array(data["nDetector"])  # number of pixels              (px)
        self.nDetector = np.array([128, 128]) 
        self.dDetector = np.array(data["dDetector"])/1000  * np.sqrt(2)# size of each pixel      (m)    --> sqrt 2 as for parallel beam diagonal needs to be included

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
        self.mode = 'parallel' # Mode parallel, cone    
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
            self.projs = F.interpolate(self.projs.unsqueeze(1), size=(self.geo.nDetector[0], self.geo.nDetector[1]), mode='bilinear', align_corners=False).squeeze(1)
            #print("Shape of the first projection of train function after rehsaping", self.projs[0].shape)
            angles = data["train"]["angles"]
            rays = self.get_rays(angles, self.geo.tilt_angle, self.geo, device)
            self.rays = torch.cat([rays, torch.ones_like(rays[...,:1])*self.near, torch.ones_like(rays[...,:1])*self.far], dim=-1)
            self.n_samples = data["numTrain"]
            coords = torch.stack(torch.meshgrid(torch.linspace(0, self.geo.nDetector[1] - 1, self.geo.nDetector[1], device=device),
                                                torch.linspace(0, self.geo.nDetector[0] - 1, self.geo.nDetector[0], device=device), indexing="ij"), -1)
            self.coords = torch.reshape(coords, [-1, 2])
            self.image = torch.tensor(data["image"], dtype=torch.float32, device=device)
            self.voxels = torch.tensor(self.get_voxels(self.geo), dtype=torch.float32, device=device)
            # Rescale the voxels to match your desired shape (e.g., 128x128x128 or 256x256x256)
            #self.voxels = F.interpolate(self.voxels.unsqueeze(0), size=(256, 256, 256), mode='trilinear', align_corners=False).squeeze(0)

        elif type == "val":
            self.projs = torch.tensor(data["val"]["projections"], dtype=torch.float32, device=device)
            #print("Shape of the first projection: ", self.projs[0].shape)
            self.projs = F.interpolate(self.projs.unsqueeze(1), size=(self.geo.nDetector[0], self.geo.nDetector[1]), mode='bilinear', align_corners=False).squeeze(1)
            angles = data["val"]["angles"]
            rays = self.get_rays(angles, self.geo.tilt_angle, self.geo, device)
            self.rays = torch.cat([rays, torch.ones_like(rays[...,:1])*self.near, torch.ones_like(rays[...,:1])*self.far], dim=-1)
            self.n_samples = data["numVal"]
            self.image = torch.tensor(data["image"], dtype=torch.float32, device=device)
            self.voxels = torch.tensor(self.get_voxels(self.geo), dtype=torch.float32, device=device)
            # Rescale the voxels to match your desired shape (e.g., 128x128x128 or 256x256x256)
            #self.voxels = F.interpolate(self.voxels.unsqueeze(0), size=(256, 256, 256), mode='trilinear', align_corners=False).squeeze(0)

    

        
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
        #voxel_tensor = torch.tensor(voxel, dtype=torch.float32, device="cuda")
        #voxel_tensor = F.interpolate(voxel_tensor.unsqueeze(0), size=(256, 256, 256), mode='trilinear', align_corners=False).squeeze(0)

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

                import open3d as o3d
                cube1 = plot_cube(np.zeros((3,1)), geo.sVoxel[...,np.newaxis])
                cube2 = plot_cube(np.zeros((3,1)), np.ones((3,1))*geo.DSO*2)
                rays1 = plot_rays(rays_d.cpu().detach().numpy(), rays_o.cpu().detach().numpy(), 2)
                poseray = plot_camera_pose(pose.cpu().detach().numpy())
                o3d.visualization.draw_geometries([cube1, cube2, rays1, poseray])
            elif geo.mode == "parallel":

        
                i, j = torch.meshgrid(torch.linspace(0, W - 1, W, device=device),
                                        torch.linspace(0, H - 1, H, device=device), indexing="ij")  # pytorch"s meshgrid has indexing="ij"
                uu = (i.t() + 0.5 - W / 2) * geo.dDetector[0] + geo.offDetector[0]
                vv = (j.t() + 0.5 - H / 2) * geo.dDetector[1] + geo.offDetector[1]
                dirs = torch.stack([torch.zeros_like(uu), torch.zeros_like(uu), torch.ones_like(uu)], -1)
                rays_d = torch.sum(torch.matmul(pose[:3,:3], dirs[..., None]).to(device), -1) # pose[:3, :3] * 
                rays_o = torch.sum(torch.matmul(pose[:3,:3], torch.stack([uu,vv,torch.zeros_like(uu)],-1)[..., None]).to(device), -1) + pose[:3, -1].expand(rays_d.shape)


                #shift ray origin to account for tilt angle (aligning object with parallel beams)
                #translation = np.sin(tilt_angle) * geo.DSO
                #object_length = geo.sVoxel[2] # length along the z-axis
#
                #translation = geo.DSO* np.sin(tilt_angle) #- (object_length/2 + object_length/2 * np.tan(tilt_angle)) #
#
#
                ## Calculate the translation based on tilt angle
                ##translation = np.sin(tilt_angle) * geo.DSO - (geo.sVoxel[2] / 2) * np.sin(tilt_angle)
#
                ## Calculate ray origins
                #rays_o = torch.sum(torch.matmul(pose[:3, :3], torch.stack([uu, vv, torch.zeros_like(uu)], -1)[..., None]).to(device), -1) + pose[:3, -1].expand(rays_d.shape)
                #rays_o[:, :, 2] -= translation  # Shift along the z-axis

               


                ## Print the origin of the center ray and camera pose to compare
                #rays_o_np = rays_o.cpu().detach().numpy()  # Convert to NumPy for printing
                ## Access the central ray origin
                #center_ray_origin = rays_o_np[64, 64]  # Accessing the ray at (64, 64)
                #print(f"Central Ray Origin = (x: {center_ray_origin[0]:.3f}, y: {center_ray_origin[1]:.3f}, z: {center_ray_origin[2]:.3f})")
                #camera_position = pose[:3, -1]  # Extract camera position from the pose
                #print(f"Camera Position:     (x: {camera_position[0]:.3f}, y: {camera_position[1]:.3f}, z: {camera_position[2]:.3f})")
                center_i, center_j = W // 2, H // 2  # Center pixel indices
                central_ray_origin = rays_o[center_i, center_j]  # Get the origin of the central ray
            
                # Update only the Z-axis of the camera pose
                #####pose[2, -1] = central_ray_origin[2]  # Set Z position to the central ray origin's Z


                # Update pose for visualization
                #camera_position = pose[:3, -1] + torch.tensor([0, 0, translation], device=device)
                #camera_position = pose[:3, -1].to(device) + torch.tensor([0, 0, translation], device=device)  # Correctly apply translation

                # Visualization
                import open3d as o3d
                cube1 = plot_cube(np.zeros((3, 1)), geo.sVoxel[..., np.newaxis])
                cube2 = plot_cube(np.zeros((3, 1)), np.ones((3, 1)) * geo.DSO * 2)
                rays1 = plot_rays(rays_d.cpu().detach().numpy(), rays_o.cpu().detach().numpy(), 2)
                poseray = plot_camera_pose(pose.cpu().detach().numpy())  # Use updated camera position
                o3d.visualization.draw_geometries([cube1, cube2, rays1, poseray])

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

        object_length = self.geo.sVoxel[2]  # Length along z-axis
        translation_z = DSO * np.tan(tilt_angle)
        # Translation vector to place the x-ray source at DSO distance, and account for tilt
        trans = np.array([DSO * np.cos(angle), DSO * np.sin(angle), translation_z])


        #trans = np.array([DSO * np.cos(angle), DSO * np.sin(angle), 0]) #when cosidering dome shape
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

