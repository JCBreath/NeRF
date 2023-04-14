import torch
from volumetric_rendering.ray_sampler import RaySampler
from camera_utils import LookAtPoseSampler
import math, os
import json
from models.tensorBase import get_rays, get_ray_directions

def normalize_vecs(vectors: torch.Tensor) -> torch.Tensor:
    """
    Normalize vector lengths.
    """
    return vectors / (torch.norm(vectors, dim=-1, keepdim=True))

def create_cam2world_matrix(forward_vector, origin):
    """
    Takes in the direction the camera is pointing and the camera origin and returns a cam2world matrix.
    Works on batches of forward_vectors, origins. Assumes y-axis is up and that there is no camera roll.
    """
    
    forward_vector = normalize_vecs(forward_vector)
    up_vector = torch.FloatTensor([0, 1, 0]).expand_as(forward_vector)

    right_vector = -normalize_vecs(torch.cross(up_vector, forward_vector, dim=-1))
    up_vector = normalize_vecs(torch.cross(forward_vector, right_vector, dim=-1))

    rotation_matrix = torch.eye(4).unsqueeze(0).repeat(forward_vector.shape[0], 1, 1)
    rotation_matrix[:, :3, :3] = torch.stack((right_vector, up_vector, forward_vector), axis=-1)

    translation_matrix = torch.eye(4).unsqueeze(0).repeat(forward_vector.shape[0], 1, 1)
    translation_matrix[:, :3, 3] = origin
    cam2world = (translation_matrix @ rotation_matrix)[:, :, :]
    assert(cam2world.shape[1:] == (4, 4))
    return cam2world

data_path = '/mnt/g/vis_data/vol-render-256/tangaroa/DIV_iso_time_1_num_130/'
# data_path = '/mnt/g/vis_data/iso-render-256/tangaroa/DIV_058_time_1_num_181/'
img_files = os.listdir('{}'.format(data_path))
img_files.sort(key=lambda x:float(x.split('.png')[0].split('_')[2]))
r = 1.0

json_data = {}
json_data['camera_angle_x'] = math.pi * 30 / 180
json_data['frames'] = []

count = 0

lookat_point    = (0, 0, 0)
camera_pivot = torch.tensor(lookat_point)
focal_length = (1/2) / math.tan(15/180*math.pi)
intrinsics = torch.tensor([[focal_length, 0, 0.5], [0, focal_length, 0.5], [0, 0, 1]])

directions = get_ray_directions(1, 1, [focal_length,focal_length])
directions = directions / torch.norm(directions, dim=-1, keepdim=True)

for filename in img_files:
    _,timestep,theta,phi, x, y, z = filename.split('.png')[0].split('_')
    timestep = int(timestep)
    theta = float(theta) / 90.0
    phi = float(phi) / 180.0
    x = float(x)*r
    y = float(y)*r
    z = float(z)*r

    camera_origins = torch.FloatTensor([x,y,z]).view(1,3)  * 4.0311
    forward_cam2world_pose = LookAtPoseSampler.sample_point(camera_origins, camera_pivot)
    c2w = forward_cam2world_pose

    swap_row = torch.FloatTensor([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])
    mask = torch.FloatTensor([[-1,1,1,-1],[1,-1,-1,1],[1,-1,-1,1],[1,1,1,1]])
    blender2opencv = torch.FloatTensor([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    up = torch.FloatTensor([0, 1, 0])

    c2w = create_cam2world_matrix(normalize_vecs(camera_pivot + 1e-8 - camera_origins), camera_origins)
    c2w = c2w[0]
    c2w = swap_row @ c2w
    c2w = c2w * mask
    c2w = c2w + 1e-8

    frame = {}
    frame['file_path'] = './train/r_{}'.format(count)
    frame['rotation'] = 0.0
    frame['transform_matrix'] = c2w.tolist()

    json_data['frames'].append(frame)

    count += 1

json_obj = json.dumps(json_data,indent=4)
with open("transforms_train.json", "w") as outfile:
    outfile.write(json_obj)
