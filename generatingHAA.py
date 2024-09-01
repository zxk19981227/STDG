import cv2
import h5py
import numpy as np
import re
from utils.rgbd_util import *
from matplotlib import pyplot as plt
from getHHA import getHHA
from tqdm import tqdm
# D=read
# camera_matrix = getCameraParam('color')
# print('max gray value: ', np.max(D))  # make sure that the image is in 'meter'
# hha = getHHA(camera_matrix, D, RD)
# hha_complete = getHHA(camera_matrix, D, D)
# cv2.imwrite('demo/hha.png', hha)
# cv2.imwrite('demo/hha_complete.png', hha_complete)

''' multi-peocessing example '''
'''
from multiprocessing import Pool

def generate_hha(i):
    # generate hha for the i-th image
    return

processNum = 16
pool = Pool(processNum)

for i in range(img_num):
    print(i)
    pool.apply_async(generate_hha, args=(i,))
    pool.close()
    pool.join()
'''
#
def read_pfm(path):
    """Read pfm file.

    Args:
        path (str): path to file

    Returns:
        tuple: (data, scale)
    """
    with open(path, "rb") as file:

        color = None
        width = None
        height = None
        scale = None
        endian = None

        header = file.readline().rstrip()
        if header.decode("ascii") == "PF":
            color = True
        elif header.decode("ascii") == "Pf":
            color = False
        else:
            raise Exception("Not a PFM file: " + path)

        dim_match = re.match(r"^(\d+)\s(\d+)\s$", file.readline().decode("ascii"))
        if dim_match:
            width, height = list(map(int, dim_match.groups()))
        else:
            raise Exception("Malformed PFM header.")

        scale = float(file.readline().decode("ascii").rstrip())
        if scale < 0:
            # little-endian
            endian = "<"
            scale = -scale
        else:
            # big-endian
            endian = ">"

        data = np.fromfile(file, endian + "f")
        shape = (height, width, 3) if color else (height, width)

        data = np.reshape(data, shape)
        data = np.flipud(data)

        return data, scale



# def normalize_depth(depth):
#     # 将深度值归一化到0到255的范围
#     clipped_depth = np.clip(depth, min(depth), max(depth))  # 这里假设最大深度为10米
#     normalized_depth = (clipped_depth / max(depth)) * 255
#     return normalized_depth.astype(np.uint8)


    # return hha_image
data=h5py.File('openpifpaf/data/visual_genome/imdb_512.h5')
from multiprocessing import Pool


def generate_hha(i,image_ids2):
    # generate hha for the i-th image
    # image = data['images'][i]
    # print(image.shape)
    image_ids = image_ids2[i]
    depth_image_name = f"{image_ids}-dpt_beit_large_512.pfm"
    # image = image.transpose(1, 2, 0)
    # 读取RGB和深度图像
    original_depth, scale = read_pfm(f"openpifpaf/data/output/{depth_image_name}")
    depth_information = np.zeros((512, 512), dtype=np.float32)
    mask_location = np.where(original_depth == 0)
    original_depth = ((original_depth - np.min(original_depth))) / (
            np.max(original_depth) - np.min(original_depth))
    # real_depth = 1/original_depth
    original_depth = 1 / (original_depth + 1e-8)
    original_depth[mask_location] = 0
    original_depth = np.clip(original_depth, 0, 10)
    # original_depth = ((original_depth - np.min(original_depth)) + 1e-8)/( np.max(original_depth) - np.min(original_depth))
    # original_depth = original_depth
    height, width = original_depth.shape[:2]
    resize_scale = 512 / max(height, width)
    new_height = int(resize_scale * height)
    new_widht = int(resize_scale * width)
    original_depth = cv2.resize(original_depth, (new_widht, new_height))
    depth_information[:new_height, :new_widht] = original_depth
    # depth_information=depth_information.astype('uint8')
    # depth_image = cv2.imread('depth_image.png', cv2.IMREAD_ANYDEPTH)
    # cv2.imshow('RGB Image', image)
    cam_instrics = np.array([[500, 0, height / 2], [0, 500, width / 2], [0, 0, 1]])
    # RD=np.concatenate([image,depth_information[...,np.newaxis]],axis=2)
    # print(RD.shape)
    D = depth_information
    # D=cv2.cvtColor(depth_information,cv2.COLOR_BGR2GRAY)
    plt.imshow(depth_information)
    plt.show()
    # 转换为HHA表示形式
    # plt.imshow(image)
    # plt.show()
    hha = getHHA(cam_instrics, D, D)

    # plt.imshow(image)
    # plt.show()
    #
    # plt.figure(figsize=(8, 6))
    # plt.imshow(depth_information, cmap='gray')
    # plt.title('depth')
    # plt.colorbar()
    # plt.show()
    def visualize_hha_channels(hha_image):
        # 分别提取HHA图像的三个通道
        height_channel = hha_image[..., 0]
        horizon_channel = hha_image[..., 1]
        distance_channel = hha_image[..., 2]

        # 可视化高度通道
        plt.figure(figsize=(8, 6))
        plt.imshow(height_channel, cmap='gray')
        plt.title(' angle the pixel’s local surface normal makes with the inferred gravity direction.')
        plt.colorbar()
        plt.show()

        # 可视化水平角度通道
        plt.figure(figsize=(8, 6))
        plt.imshow(horizon_channel, cmap='gray')
        plt.title('height above ground')
        plt.colorbar()
        plt.show()

        # 可视化距离通道
        plt.figure(figsize=(8, 6))
        plt.imshow(distance_channel, cmap='gray')
        plt.title('horizontal disparity,')
        plt.colorbar()
        plt.show()

    # 可视化HHA图像的每个通道
    visualize_hha_channels(hha)
    print(f"save hha image toopenpifpaf/data/hha/{image_ids}")
    cv2.imwrite(f'openpifpaf/data/hha/{image_ids}.png', hha)
    # pbtr.update(1)

    return

processNum = 16
pool = Pool(processNum)
image_number=len(data['images'])
# images=data['images'][0]
# images=images.transpose(1,2,0)
# images=cv2.cvtColor(images,cv2.COLOR_BGR2RGB)
# plt.imshow(images)
# plt.show()
image_ids=data['image_ids']
image_ids_array=np.array(image_ids)
# print(type(numpy.array(images)))
# exit(0)
def err_call_back(err):
    print(f'出错啦~ error：{str(err)}')
with tqdm(total=image_number) as pbar:
# image_number=1
    for i in range(image_number):
    # print(i)
        pool.apply_async(generate_hha, (i,image_ids_array),error_callback=err_call_back)
pool.close()
pool.join()
