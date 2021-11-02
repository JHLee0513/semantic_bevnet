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

## Datasets
Datasets should be put inside `data/`. For example, `data/semantic_kitti_4class_100x100`. You can either generate the datset or download them here(link TBD).

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

## Evaluation
to test the model add `--test` argument to the same command that would be (was) for training. This runs evaluation and saves predictions in a new directory in the log directory. To create video of the visualized predictions
```
python tools/create_videos.py -i /path/to/logfile/test/epoch-x/vis/
```