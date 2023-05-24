import os
import json

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
        if s == 'train' or test_skip == 0:
            skip = 1
        else:
            skip = test_skip
        
        for frame in meta['frame'][::skip]:
            file_name = os.path.join(root, frame['file_path'] + '.png')
            images.append()