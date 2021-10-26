# BEVNet

Source code for our work **"Semantic Terrain Classification for Off-Road Autonomous Driving"**

[website](https://sites.google.com/view/terrain-traversability/home)
![Alt Text](figs/canal.gif)
Our BEVNet-R on completly unseen data/envrionment. 

## TODOs
- [x] source code upload
- [ ] model weights upload
- [ ] dataset upload
- [ ] Instructions on dataset generation
- [ ] Instructions on inference
- [ ] experiment results
- [ ] arxiv link

## Setup (Incomplete)

Our setup runs python3.6+, the easiest way to setup would be to use [Anaconda](https://www.anaconda.com/):
```
conda create -n bevnet python=3.7
conda activate bevnet
# pip usually prevents issues with compiling spconv.
# please make sure you're installing the version that matches your CUDA env.
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
# rest may be installed with this
pip install -r requirements.txt
```

### SpConv: PyTorch Spatially Sparse Convolution Library
We utlize spconv for our 3D convolution network. To install:
```
git clone https://github.com/traveller59/spconv.git
git checkout fad3000249d27ca918f2655ff73c41f39b0f3127
git submodule update
python setup.py bdist_wheel
cd dist && pip install *.whl
```

## Datasets
Datasets should be put inside `data/`. For example, `data/semantic_kitti_4class_100x100`.

## Training

### BEVNet-S
Example:
```
cd experiments
bash train_kitti4-unknown_single.sh kitti4_100/single/include_unknown/default.yaml <tag> arg1 arg2 ...
```
Logs and model weights will be stored in a subdirectory of the config file like this: 
`experiments/kitti4_100/single/include_unknown/default-<tag>-logs/`
* `<tag>` is useful when you want to use the same config file but different hyperparameters. For example, if you
  want to do some debugging you can use set `<tag>` to `debug`.
* `arg1 arg2 ...` are command line arguments supported by `train_single.py`. For example, you can pass 
  `--batch_size=4 --log_interval=100`, etc.


### BEVNet-R
The command line formats are the same as BEVNet-S
Example:
```
cd experiments
bash train_kitti4-unknown_recurrent.sh kitti4_100/recurrent/include_unknown/default.yaml <tag> \
--n_frame=6 --seq_len=20 --frame_strides 1 10 20 \
--resume kitti4_100/single/include_unknown/default-logs/model.pth.4 \
--resume_epoch 0
```
Logs and model weights will be stored in a subdirectory of the config file 
`experiments/kitti4_100/recurrent/include_unknown/default-<tag>-logs/`.
