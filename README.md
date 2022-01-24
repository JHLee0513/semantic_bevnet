# BEVNet

Source code for our work **"Semantic Terrain Classification for Off-Road Autonomous Driving"**

## TODOs
- [x] source code upload
- [ ] model weights upload
- [ ] dataset upload
- [ ] Instructions on dataset generation
- [ ] Instructions on inference

## Datasets
Datasets should be put inside `data/`. For example, `data/semantic_kitti_4class_100x100`.

### Download links
SemanticKITTI: [Google Drive](https://drive.google.com/file/d/1PsU0v5wC6n5gn7sK7uJS6p_8zbeK8szu/view?usp=sharing)
RELLIS: (to be uploaded)

## Running the pretrained models

### Model weights

SemanticKITTI
* Single-frame: [Google Drive](https://drive.google.com/file/d/1vtuowdWECV3agyFPQllArxpyKXio4WHl/view?usp=sharing)
* Recurrent: [Google Drive](https://drive.google.com/file/d/1vtuowdWECV3agyFPQllArxpyKXio4WHl/view?usp=sharing)

RELLIS
* (to be uploaded)
* (to be uploaded)

### Run the models
First extract the model weights
``` shell
cd /path/to/bevnet/experiments
unzip /path/to/zip/file
```

To run the models on the validation set, `cd` to `bevnet/bevnet`, then run
``` shell
# Single-frame model
python test_single.py --model_file ../experiments/kitti4_100/single/include_unknown/default-logs/model.pth.4 --test_env kitti4

# Recurrent model
python test_recurrent.py --model_file ../experiments/kitti4_100/recurrent/include_unknown/default-logs/model.pth.2 --test_env kitti4
```


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
