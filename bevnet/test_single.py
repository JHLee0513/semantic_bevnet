import os
import cv2
import numpy as np
import torch
import tqdm
from PIL import Image
from bevnet.inference import BEVNetSingle
from bevnet.bev_utils import Evaluator
from bevnet.train_fixture_utils import make_label_vis, get_colormap
from bevnet.test_sequence_factory import make


# MODEL_FILE = '../experiments/kitti4_100/single/include_unknown/default-logs/model.pth.4'
# TEST_ENV = 'kitti4'

MODEL_FILE = '../experiments/rellis4_100/single/minkowski_maxpool_v2/model.pth.8'
TEST_ENV = 'rellis4'


model = BEVNetSingle(MODEL_FILE)
test_data = make(TEST_ENV)

if model.g.include_unknown:
    ignore_idx = model.g.num_class - 1
else:
    ignore_idx = 255
e = Evaluator(num_classes=model.g.num_class, ignore_label=ignore_idx)

for i in tqdm.trange(len(test_data['scan_files'])):
    scan_fn = test_data['scan_files'][i]
    name = os.path.basename(os.path.splitext(scan_fn)[0])
    label_fn = os.path.join(test_data['label_dir'], name + '.png')

    scan = np.fromfile(scan_fn, dtype=np.float32).reshape(-1, 4)
    label = np.array(Image.open(label_fn), dtype=np.uint8)
    label_th = torch.as_tensor(label, device='cuda').long()

    if model.g.include_unknown:
        # Note that `num_class` already includes the unknown label
        label_th[label_th == 255] = model.g.num_class - 1

    logits = model.predict(scan)[0]
    pred = torch.argmax(logits, dim=0)

    e.append(pred[None], label_th[None])

    cmap = get_colormap(model.g.dataset_type)
    label_vis = make_label_vis(label, cmap)
    pred_vis = make_label_vis(pred.cpu().numpy(), cmap)
    vis = np.concatenate([pred_vis, label_vis], axis=1)
    cv2.imshow('', cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    cv2.waitKey(1)

    if (i + 1) % 100 == 0:
        print(e.meanIoU())


print(e.classwiseIoU())
print(e.meanIoU())
