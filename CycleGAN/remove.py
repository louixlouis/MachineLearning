import os
# import olefile
from PIL import Image

def remove_gray():
    image_path = '../datasets/horse2zebra/test/B'
    image_list = os.listdir(image_path)
    for image_name in image_list:
        image = Image.open(os.path.join(image_path, image_name))
        if image.mode == 'L':
            # ole = olefile.OleFileIO(os.path.join(image_path, image_name))
            print(os.path.join(image_path, image_name))
            os.remove(os.path.join(image_path, image_name))
if __name__=='__main__':
    remove_gray()