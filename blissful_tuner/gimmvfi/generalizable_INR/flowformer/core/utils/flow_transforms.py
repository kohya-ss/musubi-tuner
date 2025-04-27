from __future__ import division
import torch
import random
import numpy as np
import numbers
import types
import scipy.ndimage as ndimage
import pdb
import torchvision
import PIL.Image as Image
import cv2
from torch.nn import functional as F


class Compose(object):
    """Composes several co_transforms together.
    For example:
    >>> co_transforms.Compose([
    >>>     co_transforms.CenterCrop(10),
    >>>     co_transforms.ToTensor(),
    >>>  ])
    """

    def __init__(self, co_transforms):
        self.co_transforms = co_transforms

    def __call__(self, input, target):
        for t in self.co_transforms:
            input, target = t(input, target)
        return input, target


class Scale(object):
    """Rescales the inputs and target arrays to the given 'size'.
    'size' will be the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the smaller edge
    interpolation order: Default: 2 (bilinear)
    """

    def __init__(self, size, order=1):
        self.ratio = size
        self.order = order
        if order == 0:
            self.code = cv2.INTER_NEAREST
        elif order == 1:
            self.code = cv2.INTER_LINEAR
        elif order == 2:
            self.code = cv2.INTER_CUBIC

    def __call__(self, inputs, target):
        if self.ratio == 1:
            return inputs, target
        h, w, _ = inputs[0].shape
        ratio = self.ratio

        inputs[0] = cv2.resize(
            inputs[0], None, fx=ratio, fy=ratio, interpolation=cv2.INTER_LINEAR
        )
        inputs[1] = cv2.resize(
            inputs[1], None, fx=ratio, fy=ratio, interpolation=cv2.INTER_LINEAR
        )
        # keep the mask same
        tmp = cv2.resize(
            target[:, :, 2], None, fx=ratio, fy=ratio, interpolation=cv2.INTER_NEAREST
        )
        target = (
            cv2.resize(target, None, fx=ratio, fy=ratio, interpolation=self.code)
            * ratio
        )
        target[:, :, 2] = tmp

        return inputs, target


class SpatialAug(object):
    def __init__(
        self,
        crop,
        scale=None,
        rot=None,
        trans=None,
        squeeze=None,
        schedule_coeff=1,
        order=1,
        black=False,
    ):
        self.crop = crop
        self.scale = scale
        self.rot = rot
        self.trans = trans
        self.squeeze = squeeze
        self.t = np.zeros(6)
        self.schedule_coeff = schedule_coeff
        self.order = order
        self.black = black

    def to_identity(self):
        self.t[0] = 1
        self.t[2] = 0
        self.t[4] = 0
        self.t[1] = 0
        self.t[3] = 1
        self.t[5] = 0

    def left_multiply(self, u0, u1, u2, u3, u4, u5):
        result = np.zeros(6)
        result[0] = self.t[0] * u0 + self.t[1] * u2
        result[1] = self.t[0] * u1 + self.t[1] * u3

        result[2] = self.t[2] * u0 + self.t[3] * u2
        result[3] = self.t[2] * u1 + self.t[3] * u3

        result[4] = self.t[4] * u0 + self.t[5] * u2 + u4
        result[5] = self.t[4] * u1 + self.t[5] * u3 + u5
        self.t = result

    def inverse(self):
        result = np.zeros(6)
        a = self.t[0]
        c = self.t[2]
        e = self.t[4]
        b = self.t[1]
        d = self.t[3]
        f = self.t[5]

        denom = a * d - b * c

        result[0] = d / denom
        result[1] = -b / denom
        result[2] = -c / denom
        result[3] = a / denom
        result[4] = (c * f - d * e) / denom
        result[5] = (b * e - a * f) / denom

        return result

    def grid_transform(self, meshgrid, t, normalize=True, gridsize=None):
        if gridsize is None:
            h, w = meshgrid[0].shape
        else:
            h, w = gridsize
        vgrid = torch.cat(
            [
                (meshgrid[0] * t[0] + meshgrid[1] * t[2] + t[4])[:, :, np.newaxis],
                (meshgrid[0] * t[1] + meshgrid[1] * t[3] + t[5])[:, :, np.newaxis],
            ],
            -1,
        )
        if normalize:
            vgrid[:, :, 0] = 2.0 * vgrid[:, :, 0] / max(w - 1, 1) - 1.0
            vgrid[:, :, 1] = 2.0 * vgrid[:, :, 1] / max(h - 1, 1) - 1.0
        return vgrid

    def __call__(self, inputs, target):
        h, w, _ = inputs[0].shape
        th, tw = self.crop
        meshgrid = torch.meshgrid([torch.Tensor(range(th)), torch.Tensor(range(tw))])[
            ::-1
        ]
        cornergrid = torch.meshgrid(
            [torch.Tensor([0, th - 1]), torch.Tensor([0, tw - 1])]
        )[::-1]

        for i in range(50):
            # im0
            self.to_identity()
            # TODO add mirror
            if np.random.binomial(1, 0.5):
                mirror = True
            else:
                mirror = False
            ##TODO
            # mirror = False
            if mirror:
                self.left_multiply(-1, 0, 0, 1, 0.5 * tw, -0.5 * th)
            else:
                self.left_multiply(1, 0, 0, 1, -0.5 * tw, -0.5 * th)
            scale0 = 1
            scale1 = 1
            squeeze0 = 1
            squeeze1 = 1
            if not self.rot is None:
                rot0 = np.random.uniform(-self.rot[0], +self.rot[0])
                rot1 = (
                    np.random.uniform(
                        -self.rot[1] * self.schedule_coeff,
                        self.rot[1] * self.schedule_coeff,
                    )
                    + rot0
                )
                self.left_multiply(
                    np.cos(rot0), np.sin(rot0), -np.sin(rot0), np.cos(rot0), 0, 0
                )
            if not self.trans is None:
                trans0 = np.random.uniform(-self.trans[0], +self.trans[0], 2)
                trans1 = (
                    np.random.uniform(
                        -self.trans[1] * self.schedule_coeff,
                        +self.trans[1] * self.schedule_coeff,
                        2,
                    )
                    + trans0
                )
                self.left_multiply(1, 0, 0, 1, trans0[0] * tw, trans0[1] * th)
            if not self.squeeze is None:
                squeeze0 = np.exp(np.random.uniform(-self.squeeze[0], self.squeeze[0]))
                squeeze1 = (
                    np.exp(
                        np.random.uniform(
                            -self.squeeze[1] * self.schedule_coeff,
                            self.squeeze[1] * self.schedule_coeff,
                        )
                    )
                    * squeeze0
                )
            if not self.scale is None:
                scale0 = np.exp(
                    np.random.uniform(
                        self.scale[2] - self.scale[0], self.scale[2] + self.scale[0]
                    )
                )
                scale1 = (
                    np.exp(
                        np.random.uniform(
                            -self.scale[1] * self.schedule_coeff,
                            self.scale[1] * self.schedule_coeff,
                        )
                    )
                    * scale0
                )
            self.left_multiply(
                1.0 / (scale0 * squeeze0), 0, 0, 1.0 / (scale0 / squeeze0), 0, 0
            )

            self.left_multiply(1, 0, 0, 1, 0.5 * w, 0.5 * h)
            transmat0 = self.t.copy()

            # im1
            self.to_identity()
            if mirror:
                self.left_multiply(-1, 0, 0, 1, 0.5 * tw, -0.5 * th)
            else:
                self.left_multiply(1, 0, 0, 1, -0.5 * tw, -0.5 * th)
            if not self.rot is None:
                self.left_multiply(
                    np.cos(rot1), np.sin(rot1), -np.sin(rot1), np.cos(rot1), 0, 0
                )
            if not self.trans is None:
                self.left_multiply(1, 0, 0, 1, trans1[0] * tw, trans1[1] * th)
            self.left_multiply(
                1.0 / (scale1 * squeeze1), 0, 0, 1.0 / (scale1 / squeeze1), 0, 0
            )
            self.left_multiply(1, 0, 0, 1, 0.5 * w, 0.5 * h)
            transmat1 = self.t.copy()
            transmat1_inv = self.inverse()

            if self.black:
                # black augmentation, allowing 0 values in the input images
                # https://github.com/lmb-freiburg/flownet2/blob/master/src/caffe/layers/black_augmentation_layer.cu
                break
            else:
                if (
                    (
                        self.grid_transform(
                            cornergrid, transmat0, gridsize=[float(h), float(w)]
                        ).abs()
                        > 1
                    ).sum()
                    + (
                        self.grid_transform(
                            cornergrid, transmat1, gridsize=[float(h), float(w)]
                        ).abs()
                        > 1
                    ).sum()
                ) == 0:
                    break
        if i == 49:
            print("max_iter in augmentation")
            self.to_identity()
            self.left_multiply(1, 0, 0, 1, -0.5 * tw, -0.5 * th)
            self.left_multiply(1, 0, 0, 1, 0.5 * w, 0.5 * h)
            transmat0 = self.t.copy()
            transmat1 = self.t.copy()

        # do the real work
        vgrid = self.grid_transform(meshgrid, transmat0, gridsize=[float(h), float(w)])
        inputs_0 = F.grid_sample(
            torch.Tensor(inputs[0]).permute(2, 0, 1)[np.newaxis], vgrid[np.newaxis]
        )[0].permute(1, 2, 0)
        if self.order == 0:
            target_0 = F.grid_sample(
                torch.Tensor(target).permute(2, 0, 1)[np.newaxis],
                vgrid[np.newaxis],
                mode="nearest",
            )[0].permute(1, 2, 0)
        else:
            target_0 = F.grid_sample(
                torch.Tensor(target).permute(2, 0, 1)[np.newaxis], vgrid[np.newaxis]
            )[0].permute(1, 2, 0)

        mask_0 = target[:, :, 2:3].copy()
        mask_0[mask_0 == 0] = np.nan
        if self.order == 0:
            mask_0 = F.grid_sample(
                torch.Tensor(mask_0).permute(2, 0, 1)[np.newaxis],
                vgrid[np.newaxis],
                mode="nearest",
            )[0].permute(1, 2, 0)
        else:
            mask_0 = F.grid_sample(
                torch.Tensor(mask_0).permute(2, 0, 1)[np.newaxis], vgrid[np.newaxis]
            )[0].permute(1, 2, 0)
        mask_0[torch.isnan(mask_0)] = 0

        vgrid = self.grid_transform(meshgrid, transmat1, gridsize=[float(h), float(w)])
        inputs_1 = F.grid_sample(
            torch.Tensor(inputs[1]).permute(2, 0, 1)[np.newaxis], vgrid[np.newaxis]
        )[0].permute(1, 2, 0)

        # flow
        pos = target_0[:, :, :2] + self.grid_transform(
            meshgrid, transmat0, normalize=False
        )
        pos = self.grid_transform(pos.permute(2, 0, 1), transmat1_inv, normalize=False)
        if target_0.shape[2] >= 4:
            # scale
            exp = target_0[:, :, 3:] * scale1 / scale0
            target = torch.cat(
                [
                    (pos[:, :, 0] - meshgrid[0]).unsqueeze(-1),
                    (pos[:, :, 1] - meshgrid[1]).unsqueeze(-1),
                    mask_0,
                    exp,
                ],
                -1,
            )
        else:
            target = torch.cat(
                [
                    (pos[:, :, 0] - meshgrid[0]).unsqueeze(-1),
                    (pos[:, :, 1] - meshgrid[1]).unsqueeze(-1),
                    mask_0,
                ],
                -1,
            )
        #                               target_0[:,:,2].unsqueeze(-1) ], -1)
        inputs = [np.asarray(inputs_0), np.asarray(inputs_1)]
        target = np.asarray(target)
        return inputs, target


class pseudoPCAAug(object):
    """
    Chromatic Eigen Augmentation: https://github.com/lmb-freiburg/flownet2/blob/master/src/caffe/layers/data_augmentation_layer.cu
    This version is faster.
    """

    def __init__(self, schedule_coeff=1):
        self.augcolor = torchvision.transforms.ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.5, hue=0.5 / 3.14
        )

    def __call__(self, inputs, target):
        inputs[0] = (
            np.asarray(self.augcolor(Image.fromarray(np.uint8(inputs[0] * 255))))
            / 255.0
        )
        inputs[1] = (
            np.asarray(self.augcolor(Image.fromarray(np.uint8(inputs[1] * 255))))
            / 255.0
        )
        return inputs, target


class PCAAug(object):
    """
    Chromatic Eigen Augmentation: https://github.com/lmb-freiburg/flownet2/blob/master/src/caffe/layers/data_augmentation_layer.cu
    """

    def __init__(
        self,
        lmult_pow=[0.4, 0, -0.2],
        lmult_mult=[
            0.4,
            0,
            0,
        ],
        lmult_add=[
            0.03,
            0,
            0,
        ],
        sat_pow=[
            0.4,
            0,
            0,
        ],
        sat_mult=[0.5, 0, -0.3],
        sat_add=[
            0.03,
            0,
            0,
        ],
        col_pow=[
            0.4,
            0,
            0,
        ],
        col_mult=[
            0.2,
            0,
            0,
        ],
        col_add=[
            0.02,
            0,
            0,
        ],
        ladd_pow=[
            0.4,
            0,
            0,
        ],
        ladd_mult=[
            0.4,
            0,
            0,
        ],
        ladd_add=[
            0.04,
            0,
            0,
        ],
        col_rotate=[
            1.0,
            0,
            0,
        ],
        schedule_coeff=1,
    ):
        # no mean
        self.pow_nomean = [1, 1, 1]
        self.add_nomean = [0, 0, 0]
        self.mult_nomean = [1, 1, 1]
        self.pow_withmean = [1, 1, 1]
        self.add_withmean = [0, 0, 0]
        self.mult_withmean = [1, 1, 1]
        self.lmult_pow = 1
        self.lmult_mult = 1
        self.lmult_add = 0
        self.col_angle = 0
        if not ladd_pow is None:
            self.pow_nomean[0] = np.exp(np.random.normal(ladd_pow[2], ladd_pow[0]))
        if not col_pow is None:
            self.pow_nomean[1] = np.exp(np.random.normal(col_pow[2], col_pow[0]))
            self.pow_nomean[2] = np.exp(np.random.normal(col_pow[2], col_pow[0]))

        if not ladd_add is None:
            self.add_nomean[0] = np.random.normal(ladd_add[2], ladd_add[0])
        if not col_add is None:
            self.add_nomean[1] = np.random.normal(col_add[2], col_add[0])
            self.add_nomean[2] = np.random.normal(col_add[2], col_add[0])

        if not ladd_mult is None:
            self.mult_nomean[0] = np.exp(np.random.normal(ladd_mult[2], ladd_mult[0]))
        if not col_mult is None:
            self.mult_nomean[1] = np.exp(np.random.normal(col_mult[2], col_mult[0]))
            self.mult_nomean[2] = np.exp(np.random.normal(col_mult[2], col_mult[0]))

        # with mean
        if not sat_pow is None:
            self.pow_withmean[1] = np.exp(
                np.random.uniform(sat_pow[2] - sat_pow[0], sat_pow[2] + sat_pow[0])
            )
            self.pow_withmean[2] = self.pow_withmean[1]
        if not sat_add is None:
            self.add_withmean[1] = np.random.uniform(
                sat_add[2] - sat_add[0], sat_add[2] + sat_add[0]
            )
            self.add_withmean[2] = self.add_withmean[1]
        if not sat_mult is None:
            self.mult_withmean[1] = np.exp(
                np.random.uniform(sat_mult[2] - sat_mult[0], sat_mult[2] + sat_mult[0])
            )
            self.mult_withmean[2] = self.mult_withmean[1]

        if not lmult_pow is None:
            self.lmult_pow = np.exp(
                np.random.uniform(
                    lmult_pow[2] - lmult_pow[0], lmult_pow[2] + lmult_pow[0]
                )
            )
        if not lmult_mult is None:
            self.lmult_mult = np.exp(
                np.random.uniform(
                    lmult_mult[2] - lmult_mult[0], lmult_mult[2] + lmult_mult[0]
                )
            )
        if not lmult_add is None:
            self.lmult_add = np.random.uniform(
                lmult_add[2] - lmult_add[0], lmult_add[2] + lmult_add[0]
            )
        if not col_rotate is None:
            self.col_angle = np.random.uniform(
                col_rotate[2] - col_rotate[0], col_rotate[2] + col_rotate[0]
            )

        # eigen vectors
        self.eigvec = np.reshape(
            [0.51, 0.56, 0.65, 0.79, 0.01, -0.62, 0.35, -0.83, 0.44], [3, 3]
        ).transpose()

    def __call__(self, inputs, target):
        inputs[0] = self.pca_image(inputs[0])
        inputs[1] = self.pca_image(inputs[1])
        return inputs, target

    def pca_image(self, rgb):
        eig = np.dot(rgb, self.eigvec)
        max_rgb = np.clip(rgb, 0, np.inf).max((0, 1))
        min_rgb = rgb.min((0, 1))
        mean_rgb = rgb.mean((0, 1))
        max_abs_eig = np.abs(eig).max((0, 1))
        max_l = np.sqrt(np.sum(max_abs_eig * max_abs_eig))
        mean_eig = np.dot(mean_rgb, self.eigvec)

        # no-mean stuff
        eig -= mean_eig[np.newaxis, np.newaxis]

        for c in range(3):
            if max_abs_eig[c] > 1e-2:
                mean_eig[c] /= max_abs_eig[c]
                eig[:, :, c] = eig[:, :, c] / max_abs_eig[c]
                eig[:, :, c] = (
                    np.power(np.abs(eig[:, :, c]), self.pow_nomean[c])
                    * ((eig[:, :, c] > 0) - 0.5)
                    * 2
                )
                eig[:, :, c] = eig[:, :, c] + self.add_nomean[c]
                eig[:, :, c] = eig[:, :, c] * self.mult_nomean[c]
        eig += mean_eig[np.newaxis, np.newaxis]

        # withmean stuff
        if max_abs_eig[0] > 1e-2:
            eig[:, :, 0] = (
                np.power(np.abs(eig[:, :, 0]), self.pow_withmean[0])
                * ((eig[:, :, 0] > 0) - 0.5)
                * 2
            )
            eig[:, :, 0] = eig[:, :, 0] + self.add_withmean[0]
            eig[:, :, 0] = eig[:, :, 0] * self.mult_withmean[0]

        s = np.sqrt(eig[:, :, 1] * eig[:, :, 1] + eig[:, :, 2] * eig[:, :, 2])
        smask = s > 1e-2
        s1 = np.power(s, self.pow_withmean[1])
        s1 = np.clip(s1 + self.add_withmean[1], 0, np.inf)
        s1 = s1 * self.mult_withmean[1]
        s1 = s1 * smask + s * (1 - smask)

        # color angle
        if self.col_angle != 0:
            temp1 = (
                np.cos(self.col_angle) * eig[:, :, 1]
                - np.sin(self.col_angle) * eig[:, :, 2]
            )
            temp2 = (
                np.sin(self.col_angle) * eig[:, :, 1]
                + np.cos(self.col_angle) * eig[:, :, 2]
            )
            eig[:, :, 1] = temp1
            eig[:, :, 2] = temp2

        # to origin magnitude
        for c in range(3):
            if max_abs_eig[c] > 1e-2:
                eig[:, :, c] = eig[:, :, c] * max_abs_eig[c]

        if max_l > 1e-2:
            l1 = np.sqrt(
                eig[:, :, 0] * eig[:, :, 0]
                + eig[:, :, 1] * eig[:, :, 1]
                + eig[:, :, 2] * eig[:, :, 2]
            )
            l1 = l1 / max_l

        eig[:, :, 1][smask] = (eig[:, :, 1] / s * s1)[smask]
        eig[:, :, 2][smask] = (eig[:, :, 2] / s * s1)[smask]
        # eig[:,:,1] = (eig[:,:,1] / s * s1) * smask + eig[:,:,1] * (1-smask)
        # eig[:,:,2] = (eig[:,:,2] / s * s1) * smask + eig[:,:,2] * (1-smask)

        if max_l > 1e-2:
            l = np.sqrt(
                eig[:, :, 0] * eig[:, :, 0]
                + eig[:, :, 1] * eig[:, :, 1]
                + eig[:, :, 2] * eig[:, :, 2]
            )
            l1 = np.power(l1, self.lmult_pow)
            l1 = np.clip(l1 + self.lmult_add, 0, np.inf)
            l1 = l1 * self.lmult_mult
            l1 = l1 * max_l
            lmask = l > 1e-2
            eig[lmask] = (eig / l[:, :, np.newaxis] * l1[:, :, np.newaxis])[lmask]
            for c in range(3):
                eig[:, :, c][lmask] = (np.clip(eig[:, :, c], -np.inf, max_abs_eig[c]))[
                    lmask
                ]
        #      for c in range(3):
        #     #           eig[:,:,c][lmask] = (eig[:,:,c] / l * l1)[lmask] * lmask + eig[:,:,c] * (1-lmask)
        #          eig[:,:,c][lmask] = (eig[:,:,c] / l * l1)[lmask]
        #          eig[:,:,c] = (np.clip(eig[:,:,c], -np.inf, max_abs_eig[c])) * lmask + eig[:,:,c] * (1-lmask)

        return np.clip(np.dot(eig, self.eigvec.transpose()), 0, 1)


class ChromaticAug(object):
    """
    Chromatic augmentation: https://github.com/lmb-freiburg/flownet2/blob/master/src/caffe/layers/data_augmentation_layer.cu
    """

    def __init__(
        self,
        noise=0.06,
        gamma=0.02,
        brightness=0.02,
        contrast=0.02,
        color=0.02,
        schedule_coeff=1,
    ):
        self.noise = np.random.uniform(0, noise)
        self.gamma = np.exp(np.random.normal(0, gamma * schedule_coeff))
        self.brightness = np.random.normal(0, brightness * schedule_coeff)
        self.contrast = np.exp(np.random.normal(0, contrast * schedule_coeff))
        self.color = np.exp(np.random.normal(0, color * schedule_coeff, 3))

    def __call__(self, inputs, target):
        inputs[1] = self.chrom_aug(inputs[1])
        # noise
        inputs[0] += np.random.normal(0, self.noise, inputs[0].shape)
        inputs[1] += np.random.normal(0, self.noise, inputs[0].shape)
        return inputs, target

    def chrom_aug(self, rgb):
        # color change
        mean_in = rgb.sum(-1)
        rgb = rgb * self.color[np.newaxis, np.newaxis]
        brightness_coeff = mean_in / (rgb.sum(-1) + 0.01)
        rgb = np.clip(rgb * brightness_coeff[:, :, np.newaxis], 0, 1)
        # gamma
        rgb = np.power(rgb, self.gamma)
        # brightness
        rgb += self.brightness
        # contrast
        rgb = 0.5 + (rgb - 0.5) * self.contrast
        rgb = np.clip(rgb, 0, 1)
        return
