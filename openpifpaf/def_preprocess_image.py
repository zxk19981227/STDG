import re

import matplotlib.pyplot as plt
import numpy as np
import cv2


def show(img, win_name='image'):
    if img is None:
        raise Exception("Can't display an empty image.")
    else:
        # cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        # cv2.imshow(win_name, img)
        # cv2.waitKey()
        # cv2.destroyWindow(win_name)
        plt.imshow(img)
        plt.show()


def read_pfm(pfm_file_path):
    with open(pfm_file_path, 'rb') as pfm_file:
        header = pfm_file.readline().decode().rstrip()
        channels = 3 if header == 'PF' else 1

        dim_match = re.match(r'^(\d+)\s(\d+)\s$', pfm_file.readline().decode('utf-8'))
        if dim_match:
            width, height = map(int, dim_match.groups())
        else:
            raise Exception("Malformed PFM header.")

        scale = float(pfm_file.readline().decode().rstrip())
        if scale < 0:
            endian = '<'  # littel endian
            scale = -scale
        else:
            endian = '>'  # big endian

        dispariy = np.fromfile(pfm_file, endian + 'f')

    img = np.reshape(dispariy, newshape=(height, width, channels))
    img = np.flipud(img).astype('uint8')
    print(img.shape)
    print(img)

    show(img, "disparity")

    return img, [(height, width, channels), scale]


from PIL import Image
import os

def convertPNG(pngfile, outdir):
    # READ THE DEPTH
    im_depth = read_pfm(pngfile)[0]
    # apply colormap on deoth image(image must be converted to 8-bit per pixel first)
    im_color = cv2.applyColorMap(cv2.convertScaleAbs(im_depth, alpha=15), cv2.COLORMAP_JET)
    # convert to mat png
    im = Image.fromarray(im_color)
    # save image
    # im.save(os.path.join(outdir, os.path.basename(pngfile)))
    show(im,'name')
img_file='/data2/zhouxukun/MiDaS-master/output/2339210-dpt_beit_large_512.pfm'
# disparity, [(h, w, c), s] = read_pfm('/data2/zhouxukun/MiDaS-master/output/2339210-dpt_beit_large_512.pfm')
# print(disparity.shape)
# print(disparity)
convertPNG(img_file,'')