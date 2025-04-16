import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F

import os
import math
import random
from glob import glob
import os.path as osp

from utils import frame_utils
from utils.augmentor import FlowAugmentor, SparseFlowAugmentor

# from utils import flow_transforms

from torchvision.utils import save_image

from utils import flow_viz
import cv2
from utils.utils import coords_grid, bilinear_sampler


class FlowDataset(data.Dataset):
    def __init__(self, aug_params=None, sparse=False):
        self.augmentor = None
        self.sparse = sparse

        if aug_params is not None:
            if sparse:
                self.augmentor = SparseFlowAugmentor(**aug_params)
            else:
                self.augmentor = FlowAugmentor(**aug_params)

        self.is_test = False
        self.init_seed = False
        self.flow_list = []
        self.image_list = []
        self.extra_info = []

    def __getitem__(self, index):
        # print(self.flow_list[index])
        if self.is_test:
            img1 = frame_utils.read_gen(self.image_list[index][0], test=self.is_test)
            img2 = frame_utils.read_gen(self.image_list[index][1], test=self.is_test)
            img1 = np.array(img1).astype(np.uint8)[..., :3]
            img2 = np.array(img2).astype(np.uint8)[..., :3]
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            return img1, img2, self.extra_info[index]

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)
        valid = None
        if self.sparse:
            flow, valid = frame_utils.readFlowKITTI(self.flow_list[index])
        else:
            flow = frame_utils.read_gen(self.flow_list[index])

        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])

        flow = np.array(flow).astype(np.float32)
        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)
        # grayscale images
        if len(img1.shape) == 2:
            img1 = np.tile(img1[..., None], (1, 1, 3))
            img2 = np.tile(img2[..., None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]

        if self.augmentor is not None:
            if self.sparse:
                img1, img2, flow, valid = self.augmentor(img1, img2, flow, valid)
            else:
                img1, img2, flow = self.augmentor(img1, img2, flow)

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()

        if valid is not None:
            valid = torch.from_numpy(valid)
        else:
            valid = (flow[0].abs() < 1000) & (flow[1].abs() < 1000)
        return img1, img2, flow, valid.float()

    def __rmul__(self, v):
        self.flow_list = v * self.flow_list
        self.image_list = v * self.image_list
        return self

    def __len__(self):
        return len(self.image_list)


class MpiSintel_submission(FlowDataset):
    def __init__(
        self, aug_params=None, split="test", root="datasets/Sintel", dstype="clean"
    ):
        super(MpiSintel_submission, self).__init__(aug_params)
        flow_root = osp.join(root, split, "flow")
        image_root = osp.join(root, split, dstype)

        if split == "test":
            self.is_test = True

        for scene in os.listdir(image_root):
            image_list = sorted(glob(osp.join(image_root, scene, "*.png")))
            for i in range(len(image_list) - 1):
                self.image_list += [[image_list[i], image_list[i + 1]]]
                self.extra_info += [(scene, i)]  # scene and frame_id

            if split != "test":
                self.flow_list += sorted(glob(osp.join(flow_root, scene, "*.flo")))


class MpiSintel(FlowDataset):
    def __init__(
        self, aug_params=None, split="training", root="datasets/Sintel", dstype="clean"
    ):
        super(MpiSintel, self).__init__(aug_params)

        root = "s3://"

        self.image_list = []
        with open("./flow_dataset/Sintel/Sintel_" + dstype + "_png.txt") as f:
            images = f.readlines()
            for img1, img2 in zip(images[0::2], images[1::2]):
                self.image_list.append([root + img1.strip(), root + img2.strip()])

        self.flow_list = []
        with open("./flow_dataset/Sintel/Sintel_" + dstype + "_flo.txt") as f:
            flows = f.readlines()
            for flow in flows:
                self.flow_list.append(root + flow.strip())

        assert len(self.image_list) == len(self.flow_list)

        self.extra_info = []
        with open("./flow_dataset/Sintel/Sintel_" + dstype + "_extra_info.txt") as f:
            info = f.readlines()
            for scene, id in zip(info[0::2], info[1::2]):
                self.extra_info.append((scene.strip(), int(id.strip())))
        # flow_root = osp.join(root, split, 'flow')
        # image_root = osp.join(root, split, dstype)

        # if split == 'test':
        #     self.is_test = True

        # for scene in os.listdir(image_root):
        #     image_list = sorted(glob(osp.join(image_root, scene, '*.png')))
        #     for i in range(len(image_list)-1):
        #         self.image_list += [ [image_list[i], image_list[i+1]] ]
        #         self.extra_info += [ (scene, i) ] # scene and frame_id

        #     if split != 'test':
        #         self.flow_list += sorted(glob(osp.join(flow_root, scene, '*.flo')))


class FlyingChairs(FlowDataset):
    def __init__(
        self, aug_params=None, split="train", root="datasets/FlyingChairs_release/data"
    ):
        super(FlyingChairs, self).__init__(aug_params)

        root = "s3://"

        with open("./flow_dataset/flying_chairs/flyingchairs_ppm.txt") as f:
            images = f.readlines()
            images = [root + img.strip() for img in images]
        with open("./flow_dataset/flying_chairs/flyingchairs_flo.txt") as f:
            flows = f.readlines()
            flows = [root + flo.strip() for flo in flows]

        # images = sorted(glob(osp.join(root, '*.ppm')))
        # flows = sorted(glob(osp.join(root, '*.flo')))
        assert len(images) // 2 == len(flows)

        split_list = np.loadtxt("chairs_split.txt", dtype=np.int32)
        for i in range(len(flows)):
            xid = split_list[i]
            if (split == "training" and xid == 1) or (
                split == "validation" and xid == 2
            ):
                self.flow_list += [flows[i]]
                self.image_list += [[images[2 * i], images[2 * i + 1]]]


class FlyingThings3D(FlowDataset):
    def __init__(
        self, aug_params=None, root="datasets/FlyingThings3D", dstype="frames_cleanpass"
    ):
        super(FlyingThings3D, self).__init__(aug_params)

        root = "s3://"

        self.image_list = []
        with open(
            "./flow_dataset/flying_things/flyingthings_" + dstype + "_png.txt"
        ) as f:
            images = f.readlines()
            for img1, img2 in zip(images[0::2], images[1::2]):
                self.image_list.append([root + img1.strip(), root + img2.strip()])
        self.flow_list = []
        with open(
            "./flow_dataset/flying_things/flyingthings_" + dstype + "_pfm.txt"
        ) as f:
            flows = f.readlines()
            for flow in flows:
                self.flow_list.append(root + flow.strip())

        # for cam in ['left']:
        #     for direction in ['into_future', 'into_past']:
        #         image_dirs = sorted(glob(osp.join(root, dstype, 'TRAIN/*/*')))
        #         image_dirs = sorted([osp.join(f, cam) for f in image_dirs])

        #         flow_dirs = sorted(glob(osp.join(root, 'optical_flow/TRAIN/*/*')))
        #         flow_dirs = sorted([osp.join(f, direction, cam) for f in flow_dirs])

        #         for idir, fdir in zip(image_dirs, flow_dirs):
        #             images = sorted(glob(osp.join(idir, '*.png')) )
        #             flows = sorted(glob(osp.join(fdir, '*.pfm')) )
        #             for i in range(len(flows)-1):
        #                 if direction == 'into_future':
        #                     self.image_list += [ [images[i], images[i+1]] ]
        #                     self.flow_list += [ flows[i] ]
        #                 elif direction == 'into_past':
        #                     self.image_list += [ [images[i+1], images[i]] ]
        #                     self.flow_list += [ flows[i+1] ]


class KITTI(FlowDataset):
    def __init__(self, aug_params=None, split="training", root="datasets/KITTI"):
        super(KITTI, self).__init__(aug_params, sparse=True)
        if split == "testing":
            self.is_test = True

        root = "s3://"

        self.image_list = []
        with open("./flow_dataset/KITTI/KITTI_{}_image.txt".format(split)) as f:
            images = f.readlines()
            for img1, img2 in zip(images[0::2], images[1::2]):
                self.image_list.append([root + img1.strip(), root + img2.strip()])

        self.extra_info = []
        with open("./flow_dataset/KITTI/KITTI_{}_extra_info.txt".format(split)) as f:
            info = f.readlines()
            for id in info:
                self.extra_info.append([id.strip()])

        if split == "training":
            self.flow_list = []
            with open("./flow_dataset/KITTI/KITTI_{}_flow.txt".format(split)) as f:
                flow = f.readlines()
                for flo in flow:
                    self.flow_list.append(root + flo.strip())
        # root = osp.join(root, split)
        # images1 = sorted(glob(osp.join(root, 'image_2/*_10.png')))
        # images2 = sorted(glob(osp.join(root, 'image_2/*_11.png')))

        # for img1, img2 in zip(images1, images2):
        #     frame_id = img1.split('/')[-1]
        #     self.extra_info += [ [frame_id] ]
        #     self.image_list += [ [img1, img2] ]

        # if split == 'training':
        #     self.flow_list = sorted(glob(osp.join(root, 'flow_occ/*_10.png')))


class AutoFlow(data.Dataset):
    def __init__(self, num_steps, crop_size, log_dir, root="datasets/"):
        super(AutoFlow, self).__init__()

        root = "s3://"
        self.image_list = []
        with open("./flow_dataset/AutoFlow/AutoFlow_image.txt") as f:
            images = f.readlines()
            for img1, img2 in zip(images[0::2], images[1::2]):
                self.image_list.append([root + img1.strip(), root + img2.strip()])
        self.flow_list = []
        with open("./flow_dataset/AutoFlow/AutoFlow_flow.txt") as f:
            flows = f.readlines()
            for flow in flows:
                self.flow_list.append(root + flow.strip())

        self.crop_size = crop_size
        self.log_dir = log_dir
        self.num_steps = num_steps
        self.scale = 1
        self.order = 1
        self.black = False
        self.noise = 0
        self.is_test = False
        self.init_seed = False

        self.iter_counts = 0

    def __rmul__(self, v):
        self.flow_list = v * self.flow_list
        self.image_list = v * self.image_list
        return self

    def __len__(self):
        return len(self.image_list) * 100

    def __getitem__(self, index):
        # print(self.flow_list[index])
        if self.is_test:
            img1 = frame_utils.read_gen(self.image_list[index][0], test=self.is_test)
            img2 = frame_utils.read_gen(self.image_list[index][1], test=self.is_test)
            img1 = np.array(img1).astype(np.uint8)[..., :3]
            img2 = np.array(img2).astype(np.uint8)[..., :3]
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            return img1, img2, self.extra_info[index]

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True
        index = index % len(self.image_list)
        valid = None

        flow = frame_utils.read_gen(self.flow_list[index])

        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])

        flow = np.array(flow).astype(np.float32)
        # For PWC-style augmentation, pixel values are in [0, 1]
        img1 = np.array(img1).astype(np.uint8) / 255.0
        img2 = np.array(img2).astype(np.uint8) / 255.0

        # grayscale images
        if len(img1.shape) == 2:
            img1 = np.tile(img1[..., None], (1, 1, 3))
            img2 = np.tile(img2[..., None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]

        iter_counts = self.iter_counts
        self.iter_counts = self.iter_counts + 1
        print(self.iter_counts)
        th, tw = self.crop_size
        schedule = [0.5, 1.0, self.num_steps]  # initial coeff, final_coeff, half life
        schedule_coeff = schedule[0] + (schedule[1] - schedule[0]) * (
            2 / (1 + np.exp(-1.0986 * iter_counts / schedule[2])) - 1
        )

        co_transform = flow_transforms.Compose(
            [
                flow_transforms.Scale(self.scale, order=self.order),
                flow_transforms.SpatialAug(
                    [th, tw],
                    scale=[0.4, 0.03, 0.2],
                    rot=[0.4, 0.03],
                    trans=[0.4, 0.03],
                    squeeze=[0.3, 0.0],
                    schedule_coeff=schedule_coeff,
                    order=self.order,
                    black=self.black,
                ),
                flow_transforms.PCAAug(schedule_coeff=schedule_coeff),
                flow_transforms.ChromaticAug(
                    schedule_coeff=schedule_coeff, noise=self.noise
                ),
            ]
        )

        flow = np.concatenate(
            [flow, np.ones((flow.shape[0], flow.shape[1], 1))], axis=-1
        )
        augmented, flow_valid = co_transform([img1, img2], flow)
        flow = flow_valid[:, :, :2]
        valid = flow_valid[:, :, 2:3]

        img1 = augmented[0]
        img2 = augmented[1]
        if np.random.binomial(1, 0.5):
            # sx = int(np.random.uniform(25,100))
            # sy = int(np.random.uniform(25,100))
            sx = int(np.random.uniform(50, 125))
            sy = int(np.random.uniform(50, 125))
            # sx = int(np.random.uniform(50,150))
            # sy = int(np.random.uniform(50,150))
            cx = int(np.random.uniform(sx, img2.shape[0] - sx))
            cy = int(np.random.uniform(sy, img2.shape[1] - sy))
            img2[cx - sx : cx + sx, cy - sy : cy + sy] = np.mean(np.mean(img2, 0), 0)[
                np.newaxis, np.newaxis
            ]

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()

        if valid is not None:
            valid = torch.from_numpy(valid).permute(2, 0, 1).float()
            valid = valid[0]
        else:
            valid = (flow[0].abs() < 1000) & (flow[1].abs() < 1000)

        return img1 * 255, img2 * 255, flow, valid.float()


class HD1K(FlowDataset):
    def __init__(self, aug_params=None, root="datasets/HD1k"):
        super(HD1K, self).__init__(aug_params, sparse=True)

        root = "s3://"
        self.image_list = []
        with open("./flow_dataset/HD1K/HD1K_image.txt") as f:
            images = f.readlines()
            for img1, img2 in zip(images[0::2], images[1::2]):
                self.image_list.append([root + img1.strip(), root + img2.strip()])
        self.flow_list = []
        with open("./flow_dataset/HD1K/HD1K_flow.txt") as f:
            flows = f.readlines()
            for flow in flows:
                self.flow_list.append(root + flow.strip())

        # seq_ix = 0
        # while 1:
        #     flows = sorted(glob(os.path.join(root, 'hd1k_flow_gt', 'flow_occ/%06d_*.png' % seq_ix)))
        #     images = sorted(glob(os.path.join(root, 'hd1k_input', 'image_2/%06d_*.png' % seq_ix)))

        #     if len(flows) == 0:
        #         break

        #     for i in range(len(flows)-1):
        #         self.flow_list += [flows[i]]
        #         self.image_list += [ [images[i], images[i+1]] ]

        #     seq_ix += 1


def fetch_dataloader(args, TRAIN_DS="C+T+K+S+H"):
    """Create the data loader for the corresponding trainign set"""

    if args.stage == "chairs":
        if hasattr(args.percostformer, "pwc_aug") and args.percostformer.pwc_aug:
            aug_params = {
                "crop_size": args.image_size,
                "min_scale": -0.1,
                "max_scale": 1.0,
                "do_flip": True,
                "pwc_aug": True,
            }
        else:
            aug_params = {
                "crop_size": args.image_size,
                "min_scale": -0.1,
                "max_scale": 1.0,
                "do_flip": True,
            }
        train_dataset = FlyingChairs(aug_params, split="training")

    elif args.stage == "things":
        aug_params = {
            "crop_size": args.image_size,
            "min_scale": -0.4,
            "max_scale": 0.8,
            "do_flip": True,
        }
        clean_dataset = FlyingThings3D(aug_params, dstype="frames_cleanpass")
        final_dataset = FlyingThings3D(aug_params, dstype="frames_finalpass")
        train_dataset = clean_dataset + final_dataset

    elif args.stage == "sintel":
        aug_params = {
            "crop_size": args.image_size,
            "min_scale": -0.2,
            "max_scale": 0.6,
            "do_flip": True,
        }
        things = FlyingThings3D(aug_params, dstype="frames_cleanpass")
        sintel_clean = MpiSintel(aug_params, split="training", dstype="clean")
        sintel_final = MpiSintel(aug_params, split="training", dstype="final")

        if TRAIN_DS == "C+T+K+S+H":
            kitti = KITTI(
                {
                    "crop_size": args.image_size,
                    "min_scale": -0.3,
                    "max_scale": 0.5,
                    "do_flip": True,
                }
            )
            hd1k = HD1K(
                {
                    "crop_size": args.image_size,
                    "min_scale": -0.5,
                    "max_scale": 0.2,
                    "do_flip": True,
                }
            )
            train_dataset = (
                100 * sintel_clean
                + 100 * sintel_final
                + 200 * kitti
                + 5 * hd1k
                + things
            )

        elif TRAIN_DS == "C+T+K/S":
            train_dataset = 100 * sintel_clean + 100 * sintel_final + things

    elif args.stage == "kitti":
        aug_params = {
            "crop_size": args.image_size,
            "min_scale": -0.2,
            "max_scale": 0.4,
            "do_flip": False,
        }
        train_dataset = KITTI(aug_params, split="training")

    elif args.stage == "autoflow-pwcaug":
        aug_params = {
            "num_steps": args.trainer.num_steps,
            "crop_size": args.image_size,
            "log_dir": args.log_dir,
        }
        train_dataset = AutoFlow(**aug_params)

    train_loader = data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        pin_memory=False,
        shuffle=True,
        num_workers=args.batch_size,
        drop_last=True,
    )

    print("Training with %d image pairs" % len(train_dataset))
    return train_loader


if __name__ == "__main__":
    aug_params = {
        "crop_size": [400, 720],
        "min_scale": -0.2,
        "max_scale": 0,
        "do_flip": True,
    }
    aug_params["min_scale"] = -0.2
    aug_params["min_stretch"] = -0.2
    sintel_clean = MpiSintel(aug_params, split="training", dstype="clean")

    train_loader = data.DataLoader(
        sintel_clean,
        batch_size=1,
        pin_memory=False,
        shuffle=True,
        num_workers=1,
        drop_last=True,
    )

    for i_batch, data_blob in enumerate(train_loader):
        image1, image2, flow, valid = [x for x in data_blob]
        print(i_batch, image1.shape)

        # if i_batch==5:
        #     exit()
