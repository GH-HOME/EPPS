import math
import os
import matplotlib.colors as colors
import numpy as np
import torch
from torch.utils.data import Dataset


def get_mgrid(sidelen, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.'''
    if isinstance(sidelen, int):
        sidelen = dim * (sidelen,)

    if dim == 2:
        pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1]], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[0, :, :, 0] = pixel_coords[0, :, :, 0] / (sidelen[0] - 1)
        pixel_coords[0, :, :, 1] = pixel_coords[0, :, :, 1] / (sidelen[1] - 1)
    elif dim == 3:
        pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1], :sidelen[2]], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[..., 0] = pixel_coords[..., 0] / max(sidelen[0] - 1, 1)
        pixel_coords[..., 1] = pixel_coords[..., 1] / (sidelen[1] - 1)
        pixel_coords[..., 2] = pixel_coords[..., 2] / (sidelen[2] - 1)
    else:
        raise NotImplementedError('Not implemented for dim=%d' % dim)

    pixel_coords -= 0.5
    pixel_coords *= 2.
    pixel_coords = torch.Tensor(pixel_coords).view(-1, dim)
    return pixel_coords


def get_mgrid_fxx_fyy(sidelen, dim=2, isflip = True):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.'''
    # x [+1 -> -1] in column y [+1 -> -1] in row

    if isinstance(sidelen, int):
        sidelen = dim * (sidelen,)

    if dim == 2:
        row, column = sidelen[0], sidelen[1]
        yy, xx = np.mgrid[:row, :column]
        if isflip:
            yy = np.flip(yy, axis=0)
            xx = np.flip(xx, axis=1)
        pixel_coords = np.stack([xx, yy], axis=-1)[None, ...].astype(np.float32) ## -yy, xx
        pixel_coords[0, :, :, 0] = pixel_coords[0, :, :, 0] / (column - 1) #xx
        pixel_coords[0, :, :, 1] = pixel_coords[0, :, :, 1] / (row - 1)
    else:
        raise NotImplementedError('Not implemented for dim=%d' % dim)

    pixel_coords -= 0.5
    pixel_coords *= 2.
    pixel_coords = torch.Tensor(pixel_coords).view(-1, dim)
    return pixel_coords


def get_mgrid_xx_yy(sidelen, dim=2, mask = None):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.'''
    if isinstance(sidelen, int):
        sidelen = dim * (sidelen,)

    if dim == 2:
        row, column = sidelen[0], sidelen[1]
        yy, xx = np.mgrid[:row, :column]

        pixel_coords = np.stack([xx, yy], axis=-1)[None, ...].astype(np.float32) ## -yy, xx
        pixel_coords[0, :, :, 0] = pixel_coords[0, :, :, 0] / (sidelen[0] - 1) #xx
        pixel_coords[0, :, :, 1] = pixel_coords[0, :, :, 1] / (sidelen[1] - 1)
    else:
        raise NotImplementedError('Not implemented for dim=%d' % dim)

    pixel_coords -= 0.5
    pixel_coords *= 2.
    if mask is not None:
        pixel_coords = pixel_coords[:, mask, :]
    pixel_coords = torch.Tensor(pixel_coords).view(-1, dim)
    return pixel_coords



def lin2img(tensor, image_resolution=None):
    batch_size, num_samples, channels = tensor.shape
    if image_resolution is None:
        width = np.sqrt(num_samples).astype(int)
        height = width
    else:
        height = image_resolution[0]
        width = image_resolution[1]

    return tensor.permute(0, 2, 1).view(batch_size, channels, height, width)


def grads2img(gradients):
    mG = gradients.detach().squeeze(0).permute(-2, -1, -3).cpu()

    # assumes mG is [row,cols,2]
    nRows = mG.shape[0]
    nCols = mG.shape[1]
    mGr = mG[:, :, 0]
    mGc = mG[:, :, 1]
    mGa = np.arctan2(mGc, mGr)
    mGm = np.hypot(mGc, mGr)
    mGhsv = np.zeros((nRows, nCols, 3), dtype=np.float32)
    mGhsv[:, :, 0] = (mGa + math.pi) / (2. * math.pi)
    mGhsv[:, :, 1] = 1.

    nPerMin = np.percentile(mGm, 5)
    nPerMax = np.percentile(mGm, 95)
    mGm = (mGm - nPerMin) / (nPerMax - nPerMin)
    mGm = np.clip(mGm, 0, 1)

    mGhsv[:, :, 2] = mGm
    mGrgb = colors.hsv_to_rgb(mGhsv)
    return torch.from_numpy(mGrgb).permute(2, 0, 1)


def rescale_img(x, mode='scale', perc=None, tmax=1.0, tmin=0.0):
    if (mode == 'scale'):
        if perc is None:
            xmax = torch.max(x)
            xmin = torch.min(x)
        else:
            xmin = np.percentile(x.detach().cpu().numpy(), perc)
            xmax = np.percentile(x.detach().cpu().numpy(), 100 - perc)
            x = torch.clamp(x, xmin, xmax)
        if xmin == xmax:
            return 0.5 * torch.ones_like(x) * (tmax - tmin) + tmin
        x = ((x - xmin) / (xmax - xmin)) * (tmax - tmin) + tmin
    elif (mode == 'clamp'):
        x = torch.clamp(x, 0, 1)
    return x


def to_uint8(x):
    return (255. * x).astype(np.uint8)


def to_numpy(x):
    return x.detach().cpu().numpy()



class Shading_LEDNPY(Dataset):
    def __init__(self, img_paths, LED_path, mask_path, normal_path, depth_path,
                 camera_para = None, custom_albedo = None, custom_mu = None, custom_LED_PDIR = None,
                 use_color_channel = False, cast_shadow_ratio = 0.05):
        super().__init__()

        self.LED_set = np.load(LED_path)
        self.numFrames =  len(self.LED_set)
        self.imgs = np.load(img_paths)
        self.numFrames, h, w = self.imgs.shape[0], self.imgs.shape[1], self.imgs.shape[2]
        assert len(self.LED_set) == self.numFrames

        if depth_path is None:
            self.depth = np.zeros([h, w])
        else:
            self.depth = np.load(depth_path)

        if normal_path is None:
            self.normal = np.zeros([h, w, 3])
        else:
            self.normal = np.load(normal_path)

        self.mask = np.load(mask_path)
        self.camera_para = camera_para
        self.albedo = None

        if os.path.exists(custom_mu) and  os.path.exists(custom_LED_PDIR):
            self.LED_mu = np.load(custom_mu)
            self.LED_PDIR = np.load(custom_LED_PDIR)
        else:
            self.LED_mu = np.zeros(self.numFrames)
            self.LED_PDIR = np.zeros([self.numFrames, 3])
            self.LED_PDIR[:, 2] = 1


        if len(self.imgs.shape) == 4 and not use_color_channel: # RGB
            self.imgs = np.mean(self.imgs, axis=3, keepdims=True)

        if custom_albedo is not None:
            self.albedo = np.load(custom_albedo)
        else:
            h, w = self.mask.shape
            self.albedo = np.ones([h, w, 3])

        if len(self.albedo.shape) == 3 and not use_color_channel:
            self.albedo = np.mean(self.albedo, axis=2, keepdims=True)
        self.imgs = self.imgs * self.albedo[np.newaxis]
        self.color_channel = self.imgs.shape[-1]

        img_roi = np.min(self.imgs[:, self.mask], axis=2)
        cast_shadow_thres = np.median(img_roi, axis=1) * cast_shadow_ratio
        self.cast_shadow_mask = np.min(self.imgs, axis=3, keepdims=True) < cast_shadow_thres[:, np.newaxis, np.newaxis, np.newaxis]
        self.cast_shadow_mask[:, ~self.mask] = True
        self.cast_shadow_mask = np.repeat(self.cast_shadow_mask, self.color_channel, axis=3)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return {'img': self.imgs, 'LED_loc': self.LED_set, 'cam_para': self.camera_para,
                'cast_shadow_mask': self.cast_shadow_mask, 'mask':self.mask,
                'LED_mu': self.LED_mu, 'LED_PDIR': self.LED_PDIR,
                'depth_gt':self.depth, 'normal_gt':self.normal, 'albedo_gt': self.albedo}


class Implicit2DWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset, sidelength=None, is_flip=True):

        if isinstance(sidelength, int):
            sidelength = (sidelength, sidelength)
        self.sidelength = sidelength

        self.dataset = dataset


        self.mgrid = get_mgrid_fxx_fyy(sidelength, dim = 2, isflip = is_flip)


        data = self.dataset[0]
        # 2D to 1D
        self.mgrid = self.mgrid.reshape(sidelength[0], sidelength[1], 2)[data['mask']]

        img = data['img'][:, data['mask']].transpose([1, 2, 0])
        img = torch.from_numpy(img)
        self.img = img.view(-1, self.dataset.color_channel, self.dataset.numFrames)

        self.LED_loc = torch.from_numpy(data['LED_loc'])

        cast_shadow = data['cast_shadow_mask'][:, data['mask']].transpose([1, 2, 0])
        cast_shadow_mask = torch.from_numpy(cast_shadow)
        self.cast_shadow_mask = cast_shadow_mask.view(-1, self.dataset.color_channel, self.dataset.numFrames)


        depth_gt, normal_gt = data['depth_gt'][data['mask']], data['normal_gt'][data['mask']]
        if depth_gt is not None:
            depth_gt = torch.from_numpy(depth_gt)
            self.depth_gt = depth_gt.view(-1, 1)
        else:
            self.depth_gt = None

        if normal_gt is not None:
            normal_gt = torch.from_numpy(normal_gt)
            self.normal_gt = normal_gt.view(-1, 3)
        else:
            self.normal_gt = None

        self.camera_para = torch.from_numpy(data['cam_para'])

        if data['LED_mu'] is not None:
            self.LED_mu = torch.from_numpy(data['LED_mu'])
            self.LED_PDIR = torch.from_numpy(data['LED_PDIR'])

    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, idx):

        in_dict = {'idx': idx, 'coords': self.mgrid}
        if self.camera_para is not None:
            gt_dict = {'img': self.img, 'LED_loc': self.LED_loc, 'cam_para': self.camera_para,
                        'cast_shadow_mask':self.cast_shadow_mask}
        else:
            gt_dict = {'img': self.img, 'LED_loc': self.LED_loc}

        if self.depth_gt is not None:
            gt_dict['depth_gt'] = self.depth_gt

        if self.normal_gt is not None:
            gt_dict['normal_gt'] = self.normal_gt

        if self.LED_mu is not None:
            gt_dict['LED_mu'] = self.LED_mu
            gt_dict['LED_PDIR'] = self.LED_PDIR



        return in_dict, gt_dict


