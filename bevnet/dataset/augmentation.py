from functools import partial
import numpy as np
import cv2
import transforms3d
import torch
import kornia

class PointCloudAugmentation(object):
    def __init__(self, augmentations):
        self.augmentations = dict()
        for aug in augmentations:
            kwargs = aug.copy()
            aug_name = kwargs.pop('name')

            if aug_name == 'noise':
                std = kwargs.pop('std', [0.0, 0.0, 0.5, 0.1])
                self.augmentations['noise'] = std
            elif aug_name == 'rotate':
                max_angle = kwargs.pop('max_angle', 45)
                self.augmentations['rotate'] = max_angle
            elif aug_name == 'translate':
                max_translation = kwargs.pop('max_translation', 5.0)
                self.augmentations['translate'] = max_translation
            elif aug_name == 'flip_x':
                probability = kwargs.pop('probability', 0.5)
                self.augmentations['flip_x'] = probability
            elif aug_name == 'flip_y':
                probability = kwargs.pop('probability', 0.5)
                self.augmentations['flip_y'] = probability
            elif aug_name == 'drop_points':
                keep_prop = kwargs.pop('keep_prop', 0.8)
                self.augmentations['drop_points'] = keep_prop
            else:
                raise Exception(f'{aug_name} is not supported.')

            # All the arugments should be used
            assert(len(kwargs) == 0)

    def transform_map(self, map, proj, inv_proj, interpolation='nearest'):
        # 1) map -> pc: inv_proj
        # 2) pc -> augmented pc: RT
        # 3) augmented pc -> augmented map: proj
        P = proj @ self.RT @ inv_proj
        # Remove z
        P2d = P[:, (0,1,3)][(0, 1), :]

        assert(map.ndim == 2)

        map_mask = torch.ones((1, 2) + map.shape, dtype=torch.float64)
        map_mask[0, 0] = map
        tmap_mask = kornia.warp_affine(map_mask,
                                       P2d[None], # adding batch
                                       dsize=map.shape,
                                       align_corners=False,
                                       mode=interpolation)

        return tmap_mask[0, 0].to(map.dtype), tmap_mask[0, 1].to(torch.bool)

    def transform(self, points):
        tpoints = torch.zeros_like(points)

        # transform xyz
        tpoints[:, :3] = points[:, :3].to(self.R.dtype) @ self.R.T + self.T

        # copy the rest
        tpoints[:, 3:] = points[:, 3:]
        return tpoints.to(points.dtype)

    def apply_others(self, points):
        '''
            Apply noise and drop_points augmentation.
            Make sure you have called renew_others with the
            correct number of points before calling apply_others.
        '''
        if self.noise is not None:
            points = points + self.noise[:, :points.shape[1]]

        if self.keep_points is not None:
            points = points[self.keep_points]

        return points

    def correct_pose(self, pose):
        return pose @ self.inv_RT

    def renew_transformation(self):
        self.T = torch.zeros(3, dtype=torch.float64)
        if 'translate' in self.augmentations:
            max_translation = self.augmentations['translate']
            self.T[:2] = (2 * torch.rand(2) - 1) * max_translation

        if 'rotate' in self.augmentations:
            max_angle = np.deg2rad(self.augmentations['rotate'])
            angle = (2 * torch.rand(1)[0] - 1) * max_angle

            ####### DEBUG
            #if hasattr(self, '_stat'):
            #    self._stat = not self._stat
            #else:
            #    self._stat = True

            #if self._stat:
            #    angle = np.deg2rad(0)
            #else:
            #    angle = np.deg2rad(45)
            #print(f'REMOVE ME {angle}')
            #######

            R = transforms3d.euler.euler2mat(0.0, 0.0, angle)
            R = torch.from_numpy(R.astype(np.float64))
        else:
            R = torch.eye(3, dtype=torch.float64)


        if 'flip_x' in self.augmentations:
            probability = self.augmentations['flip_x']
            enable = torch.rand(1) < probability
            if enable:
                # negate x
                R[:, 0] = -R[:, 0]


        if 'flip_y' in self.augmentations:
            probability = self.augmentations['flip_y']
            enable = torch.rand(1) < probability
            if enable:
                # negate y
                R[:, 1] = -R[:, 1]

        self.R = R

        self.RT = torch.eye(4, dtype=torch.float64)
        self.RT[:3, :3] = R
        self.RT[:3, 3] = self.T

        # We can use R.T = inv(R) property
        # but it won't be really faster.
        self.inv_RT = torch.linalg.inv(self.RT)

    def renew_others(self, npoints):
        if 'noise' in self.augmentations:
            std = self.augmentations['noise']
            noise = torch.normal(mean=0.0, std=1.0, size=(npoints, len(std)))
            clamped_noise = []
            for i in range(len(std)):
                n = torch.clamp(std[i] * noise[:, i], min=-2*std[i], max=2*std[i])
                clamped_noise.append(n)
            self.noise = torch.stack(clamped_noise, axis=-1)
        else:
            self.noise = None


        if 'drop_points' in self.augmentations:
            keep_prop = self.augmentations['drop_points']
            self.keep_points = torch.rand(npoints) < keep_prop
        else:
            self.keep_points = None
