import re
import os
import numpy as np
import imageio
import logging
import cv2
import OpenEXR
import Imath
from matplotlib import pyplot as plt
import json

imageio.plugins.freeimage.download()

def atoi(text):
    return float(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [atoi(c) for c in re.split('(\d+)', text) ]

def readList(list_path,ignore_head=False):
    lists = []
    with open(list_path) as f:
        lists = f.read().splitlines()
    if ignore_head:
        lists = lists[1:]
    return lists


def readHDR(hdr_path):
    img = imageio.imread(hdr_path, format='HDR-FI')
    return img

def writeHDR(hdr_path, img):
    imageio.imwrite(hdr_path, format='HDR', im = img)
    # cv2.imwrite(hdr_path,img)

def saveNormalRGB(Normmal, savepath, mask=None):
    N_show = Normmal /2 + 0.5
    N_show = np.array(N_show * 255).astype(np.uint8)
    if mask is not None:
        N_show[mask] = 0
    N_show = cv2.cvtColor(N_show, cv2.COLOR_BGR2RGB)
    cv2.imwrite(savepath, N_show)

def readNormalRGB(datapath):
    N_show = cv2.imread(datapath)
    Normmal = cv2.cvtColor(N_show, cv2.COLOR_BGR2RGB)
    Normmal = Normmal/128.0 - 1
    return Normmal


def readNormal16bitRGB(datapath):
    N_show = cv2.imread(datapath, cv2.IMREAD_UNCHANGED)
    N_show = cv2.cvtColor(N_show, cv2.COLOR_BGR2RGB)
    N_show = N_show / 65535.0
    Normal = 2 * N_show - 1
    N_norm = np.linalg.norm(Normal, axis=2)
    mask = N_norm > 1e-2
    Normal[mask] = Normal[mask] / N_norm[mask][:, np.newaxis]
    Normal[~mask] = 0
    return Normal, mask

def save_all_txt(dir, dict):
    assert dict is not None
    if not os.path.exists(dir):
        os.makedirs(dir)

    for key, value in dict.items():
        path = os.path.join(dir, key)
        np.savetxt(path, value)
        logging.info('=>save file: {}'.format(path))



def split_channel(f, channel, float_flag=True):
    dw = f.header()['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
    if float_flag:
        pt = Imath.PixelType(Imath.PixelType.FLOAT)
    else:
        pt = Imath.PixelType(Imath.PixelType.HALF)
    channel_str = f.channel(channel, pt)
    img = np.frombuffer(channel_str, dtype=np.float32)
    img.shape = (size[1], size[0])
    return img


def EXR2NPY(exr_name, output_folder, showflag=False):
    f = OpenEXR.InputFile(exr_name)

    channels = dict()
    for channel_name in f.header()["channels"]:
        print(channel_name)
        split_channel(f, channel_name)
        channels[channel_name] = split_channel(f, channel_name)

    try:
        normal = np.concatenate(
            (channels["normal.R"][:, :, None], channels["normal.G"][:, :, None], channels["normal.B"][:, :, None]), axis=-1)
        np.save(os.path.join(output_folder, "normal.npy"), normal)

        if showflag:
            N_show = normal
            N_show[:,:,2] = -normal[:,:,2]
            N_show[:, :, 0] = -normal[:, :, 0]
            plt.imshow((normal+1)/2)
            plt.show()
    except:
        print("EXR has no normal channel")

    try:
        position =  np.concatenate(
            (channels["position.R"][:, :, None], channels["position.G"][:, :, None], channels["position.B"][:, :, None]), axis=-1)
        np.save(os.path.join(output_folder, "position.npy"), position)

        mask = np.ones([position.shape[0], position.shape[1]]).astype(np.bool)
        mask[np.isinf(position[:,:,2])] = False
        # mask[position[:,:,2] < 0] = False
        # import cv2
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        # mask = cv2.erode(mask.astype(np.uint8), kernel).astype(np.bool)
        np.save(os.path.join(output_folder, "mask.npy"), mask)

        if showflag:
            plt.imshow(mask)
            plt.show()
            depth_show = position[:,:,2]
            depth_show[~mask] = np.mean(depth_show[mask])
            plt.imshow(depth_show, "gray")
            plt.show()
    except:
        print("EXR has no depth channel")


    try:
        image = np.concatenate(
            (channels["color.R"][:, :, None], channels["color.G"][:, :, None], channels["color.B"][:, :, None]),
            axis=-1)
        np.save(os.path.join(output_folder, "image.npy"), image)
        if showflag:
            plt.imshow(image)
            plt.show()
    except:
        print("EXR has no image channel")


def readEXR(exr_name):
    f = OpenEXR.InputFile(exr_name)

    channels = dict()
    for channel_name in f.header()["channels"]:
        split_channel(f, channel_name)
        channels[channel_name] = split_channel(f, channel_name)
    image = np.concatenate(
        (channels["R"][:, :, None], channels["G"][:, :, None], channels["B"][:, :, None]),
        axis=-1)

    return image



def createDir(dirpath):
    import os
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

    return dirpath


def read_json(json_path):
    with open(json_path) as json_file:
        data_param = json.load(json_file)

    return data_param

def write_json(data_param, json_path):
    with open(json_path, 'w') as outfile:
        json.dump(data_param, outfile, sort_keys=True, indent=4)


def write_obj(filename, d, mask = None):
    f = open(filename, "w")

    if mask is None:
        mask = np.ones_like(d)
    mask = mask.astype(np.bool)
    ind = np.zeros_like(mask, dtype=np.int32)
    ind[mask] = range(1, np.sum(mask.astype(np.int32)) + 1)

    h, w = mask.shape

    for i in range(h):
        for j in range(w):
            if ind[i, j]:
                f.write("v {0} {1} {2}\n".format(j - 0.5, h - (i - 0.5), d[i, j]))
    for i in range(h):
        for j in range(w):
            if ind[i, j] and j + 1 < w and i + 1 < h:
                if ind[i, j + 1] and ind[i + 1, j + 1]:
                    f.write("f {0} {1} {2}\n".format(ind[i, j], ind[i + 1, j + 1], ind[i, j + 1]))
                if ind[i + 1, j] and ind[i + 1, j + 1]:
                    f.write("f {0} {1} {2}\n".format(ind[i, j], ind[i + 1, j], ind[i + 1, j + 1]))
    f.close()



if __name__ == '__main__':

    s1 = readEXR(r'F:\Project\blender_2.8_rendering\output_dir\Sphere\orthographic\lambertian\scale_512\albedo_0\wo_globalIllumin\normals_alt.exr')
    s1[:,:,2]*= (-1)
    s1[:, :, 0] *= (-1)
    plt.imshow(s1/2 + 0.5)
    plt.show()
    # plt.imshow(s1[:,:,0])
    # plt.show()

