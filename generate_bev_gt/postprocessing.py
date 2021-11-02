__author__ = 'Amirreza Shaban'
__email__ = 'ashaban@uw.edu'

'''
Usage:
    Args:
        points: tensor of points with shape nx3 (xyz)
        pred: tensor of final predictions with shape (nx4) (class 0 to 3)
        config: a dictionary with the following structure:
           map:
                width: 40 # map width in meters
                height: 40 # map height in meters
                resx: 100 # map x axis resolution
                resy: 100 # map y axis resolution
           meanz_kernel:
                resw = 10
                resh = 10
                stride = 5
           threshold:
                class2 = .5
                class3 = .2
                sky = 2.5
    Returns:
        new_preds: tensor of corrected predictions with shape (nx5) (original classes + sky)
bp = BinningPostprocess(config, device='cuda')
new_pred = bp.process_pc(points, pred)
'''


import torch
import torch_scatter

class Map2D(object):
    def __init__(self, width, height, resx, resy):
        '''
            Center of the map is assumed to be at (0,0).
            One can adjust width and height and the map
            resolution.
        '''
        self.width = width # map width in global coordinate
        self.height = height # map height in global coordinate
        self.resx = resx
        self.resy = resy

    def init_map(self, kw, kh, stride=1):
        '''
            kw, kh: size of the kernel in resolution
        '''

        last_indx = self.resx - kw
        last_indy = self.resy - kh

        # check if the center stays at (0,0)
        if last_indx % stride != 0 or last_indy % stride != 0:
            raise ValueError('Choose the kernel size and stride so the center stays at (0,0)')

        # output map resolution
        resx = int(last_indx / stride + 1)
        resy = int(last_indy / stride + 1)

        # output map cell size
        cellw = stride*self.width/float(self.resx)
        cellh = stride*self.height/float(self.resy)

        # output map
        width = cellw * resx
        height = cellh * resy

        return Map2D(width, height, resx, resy)


    def apply_kernel(self, w, h, stride=1, op='mean'):
        assert(op in ['mean', 'max', 'min'])

        outmap = self.init_map(w, h, stride)

        unfold = torch.nn.Unfold((h,w), stride=stride)

        dt = unfold(self.map[None])[0]
        dt = dt.view(2, w*h, outmap.resy, outmap.resx)

        # count the number of valid points within each bin
        valid_count = dt[1].sum(axis=0)
        map_mask = valid_count > 0
        if op == 'mean':
            map_val = (dt[0] * dt[1]).sum(axis=0)
            map_val = map_val / (valid_count+1e-6)
        elif op == 'max':
            map_val = dt[0].detach().clone()
            map_val[dt[1] > 0] = float('-inf')
            map_val = map_val.max(axis=0)
        elif op == 'min':
            map_val = dt[0].detach().clone()
            map_val[dt[1] > 0] = float('inf')
            map_val = map_val.min(axis=0)

        map_val[~map_mask] = 0.0
        outmap.map = torch.stack((map_val, map_mask.float()), axis=0)

        return outmap

    def fill(self, points, inrange=None, proj_ind=None):
        ''' Fill the map with the points in global coordinate system.
            x,y will be used to to find map cell. The map cell will
            be set by the z value.
            Args:
                points: xyz
        '''

        if proj_ind is None:
            proj_ind, inrange = self.locs(points, inrange)

        # First channel holds z value, second channel is mask
        self.map = points.new_zeros((2, self.resy, self.resx))

        pt = proj_ind[inrange]
        # self.map[0].view(-1)[pt] = points[inrange, 2] # z value
        self.map[0].view(-1)[...] = torch_scatter.scatter_mean(
            points[inrange, 2],
            pt,
            dim_size=self.map[0].view(-1).shape[0])
        self.map[1].view(-1)[pt] = 1.0


    def locs(self, points, inrange=None):
        projx = ((points[:, 0] / self.width + 0.5) * self.resx).to(torch.int64)
        projy = ((points[:, 1] / self.height + 0.5) * self.resy).to(torch.int64)
        proj_ind = projx + projy * self.resx
        if inrange is None:
            inrange = (0 <= projx) & (projx < self.resx) & (0 <= projy) & (projy < self.resy)
        else:
            inrange &= (0 <= projx) & (projx < self.resx) & (0 <= projy) & (projy < self.resy)

        return proj_ind, inrange

    def query(self, points):
        ind, inrange = self.locs(points)
        inrange_ind = ind[inrange]

        # Read value
        values = torch.zeros_like(inrange, dtype=torch.float32)
        values[inrange] = self.map[0].view(-1)[inrange_ind]

        # Read mask
        mask = torch.zeros_like(inrange, dtype=torch.float32)
        mask[inrange] = self.map[1].view(-1)[inrange_ind]

        return values, mask.to(torch.bool), inrange


class BinningPostprocess(object):
    def __init__(self, config, device):
        '''
            config: a dictionary with the following structure:
               map:
                    width: 40 # map width in meters
                    height: 40 # map height in meters
                    resx: 100 # map x axis resolution
                    resy: 100 # map y axis resolution
               meanz_kernel:
                    resw = 10
                    resh = 10
                    stride = 5
               threshold:
                    class2 = .5
                    class3 = .2
                    sky = 2.5
        '''

        # map
        width = config['map']['width']
        height = config['map']['height']
        resx = config['map']['resx']
        resy = config['map']['resy']


        ## We apply a mean kernel with these parameters to estimate ground z
        self.kernel_resw = config['meanz_kernel']['resw']
        self.kernel_resh = config['meanz_kernel']['resh']
        self.kernel_stride = config['meanz_kernel']['stride']

        ## Elavation parameters
        self.class2to1_threshold = config['threshold']['class2to1']
        self.class2to3_threshold = config['threshold']['class2to3']
        self.class3_threshold = config['threshold']['class3']
        self.sky_threshold = config['threshold']['sky']
        ###

        self.device = device

        # build ground map
        self.traversable_map = Map2D(width, height, resx, resy)

    def process_pc(self, points, preds):
        ''' We assume the input prediction has 4 classes (pred.shape == nx4):
                + class 0 and 1 are traversable
                + class 2 and 3 are non-traversable
            We start off by estimating ground elavation by locally averaging z values
            of the traversable points. Then, we compute each points distance to the estimated
            ground elavation value and correct the predictions as follows:
                + (LOGIC 0) Any point outside the 2d map boundary is classified as unknown.
                + (LOGIC 1) Any point above the sky_threshold is classified as sky (a new class).
                + (LOGIC 2) Any point in class 2 that is lower than g2_threshold is labeld as class 1.
                + (LOGIC 3) Any point in class 3 that it lower than g3_threshold is labeld as class 1.
            The output is a nx5 dimensional prediction tensor.
        '''
        assert(preds.shape[1] == 4)

        def helper():
            known_cls = preds.sum(axis=1) > 0
            _, class_preds = preds.max(axis=1)
            # class_preds = preds.max(axis=1)
            class_preds[~known_cls] = -1

            # Usefull masks
            traversable = (class_preds == 0) | (class_preds == 1)

            # Project all the points
            proj_ind, inrange = self.traversable_map.locs(points)

            # build traversable points map and estimate ground z
            self.traversable_map.fill(points, inrange=traversable & inrange, proj_ind=proj_ind)
            meanz_traversable_map = self.traversable_map.apply_kernel(
                self.kernel_resw, self.kernel_resh, self.kernel_stride)

            # get ground z value for all points and compute their elavation from ground
            # we also update inrange as meanz_traversable_map might be smaller
            groundz, valid_groundz, inrange = meanz_traversable_map.query(points)
            elevations = points[:, 2] - groundz

            return valid_groundz, inrange, elevations, class_preds

        valid_groundz, inrange, elevations, class_preds = helper()

        ### Copy the original preds matrix and update it
        new_preds = preds.detach().clone()

        ## LOGIC 0: Anything outside the meanz traversable map is labeled as unknown
        new_preds[~inrange] = 0.0

        ## mask for the labels that might get updated
        class2 = (class_preds == 2) & valid_groundz # if it is not valid it is not inrange either
        class3 = (class_preds == 3) & valid_groundz
        nontraversable = class2 | class3

        # LOGIC 2:  class 2--> class 1
        class2to1 = new_preds.new_zeros((new_preds.shape[0],), dtype=torch.bool)
        class2to1[class2] = elevations[class2] < self.class2to1_threshold
        new_preds[class2to1, ...] = 0.0
        # new_pres.shape == Nx4
        new_preds[class2to1, 1] = 1.0

        # LOGIC 2.2:  class 2--> class 3
        class2to3 = new_preds.new_zeros((new_preds.shape[0],), dtype=torch.bool)
        class2to3[class2] = elevations[class2] > self.class2to3_threshold
        new_preds[class2to3, ...] = 0.0
        new_preds[class2to3, 3] = 1.0

        # LOGIC 3: class 3 --> class 1
        class3to1 = new_preds.new_zeros((new_preds.shape[0],), dtype=torch.bool)
        class3to1[class3] = elevations[class3] < self.class3_threshold
        new_preds[class3to1, ...] = 0.0
        new_preds[class3to1, 1] = 1.0

        # LOGIC 1: Sky
        sky = new_preds.new_zeros((new_preds.shape[0],), dtype=torch.bool)
        sky[nontraversable] = elevations[nontraversable] > self.sky_threshold
        new_preds[sky, ...] = 0.0

        # Append the sky prediction to the prediction vector
        new_preds = torch.cat((new_preds, sky.float()[:, None]), axis=1)

        return new_preds
