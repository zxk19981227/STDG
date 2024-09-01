import json

import cv2
import h5py
import numpy as np
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import torch
import argparse
import logging
from PIL import Image

from openpifpaf import datasets, decoder, logger, network, plugin, show, visualizer, __version__


class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter,
                      argparse.RawDescriptionHelpFormatter):
    pass

LOG = logging.getLogger(__name__)
def show_image(image):
    plt.imshow(image)
    plt.show()
def cli():  # pylint: disable=too-many-statements,too-many-branches
    plugin.register()

    parser = argparse.ArgumentParser(
        prog='python3 -m openpifpaf.eval',
        usage='%(prog)s [options]',
        description=__doc__,
        formatter_class=CustomFormatter,
    )
    parser.add_argument('--version', action='version',
                        version='OpenPifPaf {version}'.format(version=__version__))

    datasets.cli(parser)
    decoder.cli(parser)
    logger.cli(parser)
    network.Factory.cli(parser)
    show.cli(parser)
    visualizer.cli(parser)

    parser.add_argument('--output', default=None,
                        help='output filename without file extension')
    parser.add_argument('--skip-existing', default=False, action='store_true',
                        help='skip if output eval file exists already')
    parser.add_argument('--no-skip-epoch0', dest='skip_epoch0',
                        default=True, action='store_false',
                        help='do not skip eval for epoch 0')
    parser.add_argument('--watch', default=False, const=60, nargs='?', type=int)
    parser.add_argument('--disable-cuda', action='store_true',
                        help='disable CUDA')
    parser.add_argument('--write-predictions', default=False, action='store_true',
                        help='write a json and a zip file of the predictions')
    parser.add_argument('--show-final-image', default=False, action='store_true')
    parser.add_argument('--show-final-ground-truth', default=False, action='store_true')
    parser.add_argument('--flip-test', default=False, action='store_true')
    parser.add_argument('--run-metric', default=False, action='store_true')
    parser.add_argument('--use-gt-image', default=False, action='store_true')
    args = parser.parse_args()

    logger.configure(args, LOG)

    # add args.device
    args.device = torch.device('cpu')
    args.pin_memory = False
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        args.pin_memory = True
    LOG.debug('neural network device: %s', args.device)

    datasets.configure(args)
    decoder.configure(args)
    network.Factory.configure(args)
    show.configure(args)
    visualizer.configure(args)

    return args
def visualize_deformer_offset(offset):
    # 生成网格点坐标
    x = np.arange(0, offset.shape[1], 1)
    y = np.arange(0, offset.shape[0], 1)
    X, Y = np.meshgrid(x, y)

    # 偏移后的点坐标
    X_offset = X + offset[:, :, 0]
    Y_offset = Y + offset[:, :, 1]

    # 绘制原始网格
    plt.subplot(1, 2, 1)
    plt.scatter(X, Y, color='blue', label='Original')
    plt.title('Original Grid')
    plt.xlabel('X')
    plt.ylabel('Y')

    # 绘制偏移后的网格
    plt.subplot(1, 2, 2)
    plt.scatter(X_offset, Y_offset, color='red', label='Deformed')
    plt.title('Deformed Grid')
    plt.xlabel('X')
    plt.ylabel('Y')

    # 显示图像
    plt.tight_layout()
    plt.show()

# 示例偏移矩阵
offset = np.array([
    [[1, 1], [0, 0], [-1, -1]],
    [[2, 2], [0, 0], [-2, -2]],
    [[3, 3], [0, 0], [-3, -3]]
])
parser=argparse.ArgumentParser()
# 可视化偏移
args=cli()

datamodule = datasets.factory('vg')
model_cpu, _ = network.Factory().factory(head_metas=datamodule.head_metas)
# model.load_state_dict(torch.load('/data/zhouxukun/'))
loader = datamodule.eval_loader()
for i in range(len(loader)):
    image=Image.fromarray(loader.dataset._im_getter(i))
    image_processed,anns,anns_gt,depth=loader.dataset.__getitem__(i)
    print(anns)
    print(anns_gt)
    show_image(np.array(image))
    image=np.array(image)
    print(image_processed.shape)
    # image.show()
    print(anns[0][0][0].obj.bbox[:2])
    print(anns[0][0][0].obj.bbox[2:])
    cv2.rectangle(image,(int(anns[0][0][0].obj.bbox[0]),int(anns[0][0][0].obj.bbox[1])),(int(anns[0][0][0].obj.bbox[2]),int(anns[0][0][0].obj.bbox[3])),(0, 0, 255))
    cv2.imwrite('/data/zhouxukun/test.jpg',image)
    print(image,anns[0][0][0].obj.category_id[0])

    print(f"name is {image,anns[0][0][0].obj.categories[image,anns[0][0][0].obj.category_id[0]]}")
    # cv2.imshow('text',np.array(image))
    basenet_output=model_cpu.base_net(image_processed.unsqueeze(0))
    model_output,center_offset=model_cpu.get_head_results(model_cpu.head_nets,basenet_output)
    gt_predictions_location=anns_gt







visualize_deformer_offset(offset)

