import os
import json
import imageio

import numpy as np

def load_blender(root:str, bg_white:bool=True, downsample:int=0, test_skip:int=8):
    splits = ['train', 'val', 'test']
    metas = {}

    # Load annotation?
    for s in splits:
        with open(os.path.join(root, f'transforms_{s}'), 'r') as fp:
            metas[s] = json.load(fp)

    all_iamges = []
    all_poses = []
    counts = [0]
    
    # Load images.
    for s in splits:
        meta = metas[s]
        images = []
        poses = []
        # What is test_skip?
        if s == 'train' or test_skip == 0:
            skip = 1
        else:
            skip = test_skip
        
        for frame in meta['frame'][::skip]:
            file_name = os.path.join(root, frame['file_path'] + '.png')
            # images = [image1, image2, image3, ...]
            images.append(imageio.imread(file_name))
            poses.append(np.array(file_name['transform_matrix']))
            
        # Normalization
        images = (np.array(images) / 255.).astype(np.float32)
        poses = np.array(poses).astype(np.float32)
        
        # counts = [0, 1, 3, 6, ...] ?
        counts.append(counts[-1] + images.shape[0])
        all_images.append(images)
        all_poses.append(poses)
    
    # what is i_split?
    i_split = [np.arange(counts[i], counts[i+1]) for i in range(len(counts)-1)]
    images = np.concatenate(all_images, 0)
    gt_extrinsic = np.concatenate(all_poses, 0)
    
    heigth, width = images[0].shape[:2]
    camera_angle_x = float(metas['train']['camera_angle_x'])
    focal = .5 * width / np.tan(.5 * camera_angle_x)
    
    if downsample:
        height, width = int(height//downsample), int(width//downsample)
        focal = focal/downsample
        
        images_reduced = np.zeros((images.shape[0], height, width, images.shape[3]))
        for i, image in enumerate(images):
            images_reduced[i] = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        images = images_reduced
    
    # It needs when downsample is false.
    height, width = int(height), int(width)
    gt_intrinsic = np.array([
        [focal, 0, 0.5*width],
        [0, focal, 0.5*height],
        [0, 0, 1]])
    
    if bg_white:
        images = images[..., :3] * images[..., -1:] + (1.0-images[..., -1:])
    else:
        images = images[..., :3] * images[..., -1:]
    return images, [gt_intrinsic, gt_extrinsic], [height, width], i_split
