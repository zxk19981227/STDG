# STDG: Semi-Teacher-Student Training Paradigm for Depth-guided One-stage Scene Graph Generation

This codebase is the official implementation of "STDG: Semi-Teacher-Student Training Paradigm for Depth-guided One-stage Scene Graph Generation" (Accepted at ICMR2024).




## Requirements

This codebase is based on the publicly-available repository [SGG-CoRF](https://github.com/vita-epfl/SGG-CoRF). We modify certain files from OpenPifPaf and add other parts as plugins. We also include a modified version of [apex](https://github.com/NVIDIA/apex) that [Scene-Graph-Benchmark.pytorch](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch) relies on for evaluation. The main dependencies of this codebase are:

* Python 3.8.5
* [Apex](https://github.com/NVIDIA/apex)
* [Openpifpaf](https://github.com/openpifpaf/openpifpaf)
* [Scene-Graph-Benchmark.pytorch](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch)

We recommend before installing the requirements to create a virtual environment where all packages will be installed ([link](https://realpython.com/python-virtual-environments-a-primer/)).

First, make sure that inside the main folder (`SGG-CoRF`) you have the *openpifpaf* and *apex* folder. Activate the virtual environment (optional)

Then, install the requirements:

```setup

cd STDG

pip install numpy Cython

cd openpifpaf

pip install --editable '.[dev,train,test]'

pip install tqdm h5py graphviz ninja yacs cython matplotlib tqdm opencv-python overrides timm


# Make sure to re-install the correct pytorch version for your GPU from https://pytorch.org/get-started/locally/

# install apex
cd ../apex
python setup.py install --cuda_ext --cpp_ext

# install PyTorch Detection
cd ../
git clone https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch.git
cd Scene-Graph-Benchmark.pytorch

python setup.py build develop

cd ../openpifpaf

```

**Note**, when running the training or evaluation, if your code crashes because of an error related to torch_six.PY3, follow these steps:

```fixing
cd Scene-Graph-Benchmark.pytorch

vim maskrcnn_benchmark/utils/imports.py

# change the line torch_six.PY3 to torch_six.PY37

```

To perform the following steps, make sure to be in the main **openpifpaf** directory (`STDG/openpifpaf`).

In order to train the model, the dataset needs to be downloaded and pre-processed:

1. Create a folder called `data` and inside it a folder called `visual_genome`
2. Download images from [Visual Genome](http://visualgenome.org/api/v0/api_home.html) (parts 1 and 2)
3. Place all images into `data/visual_genome/VG_100K/`
4. Create VG-SGG.h5, imdb_512.h5, imdb_1024.h5, VG-SGG-dicts.json by following [here](https://github.com/danfeiX/scene-graph-TF-release/tree/master/data_tools) and place them in `data/visual_genome/`. To create imdb_512.h5, you will need to change the 1024 to 512 in `create_imdb.sh`.
5. Generate Depth images from given dataset with following commands:
    ```
   cd MiDaS
   cd weights
   wget https://github.com/isl-org/MiDaS?tab=readme-ov-file#:~:text=For%20highest%20quality%3A-,dpt_beit_large_512,-For%20moderately%20less
   cd ..
   python run.py --model_type  dpt_beit_large_512 --input_path path/to/your/dataset/image --output_path path/to/output
   ```
7. Generate HAA image feature with following command:
```
   python generatingHAA.py
```
## Training

To train the model(s) in the paper, run these commands from the main **openpifpaf** directory:


To train a ResNet-50 model  modules:
```
CUDA_VISIBLE_DEVICES=3 python -m openpifpaf.train  --lr=5e-4 --lr-basenet=2e-5 --b-scale=10.0 --lr-raf=6e-4 \
--epochs=60 --lr-decay 10 40 50 --batch-size=32 --weight-decay=5e-5 --basenet=resnet50 \
--resnet-pool0-stride=2 --resnet-block5-dilation=2 --vg-cn-upsample 1 --dataset vg --vg-cn-square-edge 512 \
--vg-cn-use-512 --vg-cn-group-deform --vg-cn-single-supervision --cf3-deform-deform4-head \
--cntrnet-deform-deform4-head  --adamw  --output train_res50_depth_deformer/model --mode depth

CUDA_VISIBLE_DEVICES=3 python -m openpifpaf.train  --lr=5e-4 --lr-basenet=5e-5 --b-scale=10.0 --basenet resnet50 \
--epochs=60 --lr-decay 10 40 50 --batch-size=40 --weight-decay=1e-5 \
--resnet-pool0-stride=2 --resnet-block5-dilation=2 --vg-cn-upsample 1 --dataset vg --vg-cn-square-edge 512 \
--vg-cn-use-512 --vg-cn-group-deform --vg-cn-single-supervision --cf3-deform-deform4-head \
--cntrnet-deform-deform4-head  --adamw  --output train_res50_combine_training/model --mode combine_train --depth-checkpoint train_res50_depth_deformer/model.epoch060
```


To train a Swin-Transformer model  modules:

```
CUDA_VISIBLE_DEVICES=0,1 python -m openpifpaf.train --mode depth --lr=1e-4 --lr-basenet=1e-5 --b-scale=10.0 --epochs=60 --lr-decay 40 50  --batch-size=40 --weight-decay=1e-5 --swin-use-fpn --basenet=swin_s --vg-cn-upsample 1 --dataset vg --vg-cn-square-edge 512 --vg-cn-use-512 --vg-cn-group-deform --vg-cn-single-supervision --cf3-deform-deform4-head --cntrnet-deform-deform4-head --adamw --vg-cn-single-head

CUDA_VISIBLE_DEVICES=2,3 python -m openpifpaf.train --mode combine --lr=5e-4 --lr-basenet=5e-5 --b-scale=10.0 --epochs=60 --lr-decay 10 40 50  --batch-size=40 --weight-decay=2e-5 --swin-use-fpn --basenet=swin_s --vg-cn-upsample 1 --dataset vg --vg-cn-square-edge 512 --vg-cn-use-512 --vg-cn-group-deform --vg-cn-single-supervision --cf3-deform-deform4-head --cntrnet-deform-deform4-head --adamw --vg-cn-single-head --depth-checkpoint train_swin_depth/swin_epoch.epoch060 --output train_swin_combine_v2/model
```

**Note**, to perform distributed training on multiple GPUs, as mentioned in the paper, add the following argument `--ddp` after `openpifpaf.train` in the commands above.

## Pre-trained Models

You can download the pretrained models from here:

- [Pretrained ResNet-50 Model](https://drive.google.com/file/d/1NsNy1zSosKEWFwC0TRvwunGA0vn6EqNu/view?usp=drive_link) trained on Visual Genome.
- [Pretrained Swin-S Model](https://drive.google.com/file/d/1oCUy7hsbXwJQtbSemV7rymA2ll0NoW2J/view?usp=drive_link) trained on Visual Genome.


Then follow these steps:

1. Create the folder `outputs` inside the main **openpifpaf** directory (if necessary)
2. Place the downloaded models inside the `outputs` folder

These models will produce the results reported in the paper.

## Evaluation

To evaluate the model on Visual Genome, go to the main **openpifpaf*** directory.


To evaluate the ResNet-50 model:

```eval

CUDA_VISIBLE_DEVICES=3 python3 -m openpifpaf.eval_cn  --checkpoint  train_res50_combine_deformer_v12/model.epoch060  --loader-workers=2   --resnet-pool0-stride=2 --resnet-block5-dilation=2   --dataset vg --decoder cifdetraf_cn --vg-cn-use-512  --vg-cn-group-deform --vg-cn-single-supervision --run-metric   --cf3-deform-bn --cf3-deform-deform4-head --cntrnet-deform-deform4-head

```

To evaluate the Swin-S model :

```eval

CUDA_VISIBLE_DEVICES=2 python3 -m openpifpaf.eval_cn --checkpoint train_swin_combine_v2/model.epoch060 --mode rgb  --loader-workers=2 --resnet-pool0-stride=2 --resnet-block5-dilation=2 --dataset vg --decoder cifdetraf_cn --vg-cn-use-512  --vg-cn-group-deform --vg-cn-single-supervision --run-metric  --cf3-deform-bn --cntrnet-deform-bn --cf3-deform-deform4-head --cntrnet-deform-deform4-head
~                                                                            

```

