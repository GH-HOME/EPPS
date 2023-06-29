import cv2
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import os
from PIL import Image


def scatter_3d(PointSet):
    """
    Visualize the scattered 3D points
    PointSet: [N, 3] point coordinates
    """

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(PointSet[:, 0], PointSet[:, 1], PointSet[:, 2])
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    # ax.set_xlim([-1, 1])
    # ax.set_ylim([-1, 1])
    # ax.set_zlim([-1, 1])
    plt.show()


def create_gif(imgs, save_path, mask=None, fps = 10):
    """
    create gif from images
    :param imgs: [N, H, W]
    :param save_path:
    :param mask:
    :return:
    """

    fig = plt.figure()
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    import imageio
    with imageio.get_writer(save_path, mode='I', fps=fps) as writer:
        for i in range(len(imgs)):
            if mask is not None:
                import cv2
                idx = mask != 0
                imgs[i][~idx] = np.NaN
                imgs[i] = imgs[i] / imgs[i][idx].max()
            else:
                imgs[i] = imgs[i] / imgs[i].max()

            writer.append_data(imgs[i])


def save_fig_no_margin(img, file_name):
    plt.imshow(img)
    plt.axis('off')

    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                        hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig(file_name, bbox_inches='tight',
                pad_inches=0)


def save_normal_no_margin(N, mask, file_name, white = False):
    img = N_2_N_show(N, mask, white)
    img = img[:,:,::-1] * 255
    img = np.uint8(img)
    cv2.imwrite(file_name, img)


def save_plt_fig_with_title(file_name, title, dpi=300, transparent = False):
    plt.title(title)
    plt.axis('off')
    plt.savefig(file_name, dpi=dpi, bbox_inches='tight',
                pad_inches=0, transparent = transparent)


def save_plt_fig_no_margin(file_name, dpi=100, transparent = False):
    plt.savefig(file_name, dpi=dpi, bbox_inches='tight',
                pad_inches=0, transparent = transparent)

def set_zero_margin():
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                        hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.axis('off')
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())


def plt_normal(normal, mask=None, savepath=None):
    N_show = normal/2 + 0.5
    if mask is not None:
        N_show[~mask] = 0
    im = plt.imshow(N_show)
    plt.axis('off')
    set_zero_margin()
    if savepath is None:
        plt.show()
    else:
        plt.savefig(savepath)
    return im


def plt_error_map(err_map, mask, vmin=0, vmax=40, withbar=False, title=None, img_path=None):
    err_map[~mask] = 0
    fig, axes = plt.subplots(1, 1)
    im = axes.imshow(err_map, vmin=vmin, vmax = vmax, cmap=plt.cm.jet)
    plt.axis('off')
    if title is not None:
        plt.title(title)

    if withbar:
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(axes)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cbar = plt.colorbar(im, cax=cax, orientation='vertical')
        # cbar.formatter.set_powerlimits((0, -4))
        cbar.update_ticks()

    if img_path is not None:
        plt.savefig(img_path, transparent=True, dpi = 300, bbox_inches='tight')
        plt.close('all')

    return im

def plt_error_map_cv2(err_map, mask, vmin=0, vmax=40):
    err_map = np.maximum(err_map, vmin)
    err_map = np.minimum(err_map, vmax)
    err_map = err_map / (vmax - vmin) * 255
    err_map = err_map.astype(np.uint8)
    im_color = cv2.applyColorMap(err_map, cv2.COLORMAP_JET)
    im_color[~mask] = 255
    return im_color

def N_2_N_show(N, mask, white=False):
    N_show = N/2 + 0.5
    N_show[~mask] = 0
    if white:
        N_show[~mask] = 1
    return N_show

def save_transparent_img(input_img_path, mask, output_img_path=None):
    """
    convert an image to a transparent image by setting the region outof mask as transparent
    """
    assert input_img_path is not None
    assert mask is not None
    if output_img_path is None:
        output_img_path = input_img_path

    img = Image.open(input_img_path)
    img = img.convert("RGBA")

    pixdata = img.load()
    width, height = img.size
    for y in range(height):
        for x in range(width):
            if not mask[y, x]:
                pixdata[x, y] = (255, 255, 255, 0)

    img.save(output_img_path, "PNG")

def createImgGIF(gif_path, img_filenames, fps = 10):
    import imageio, tqdm
    print('creating GIF...')
    with imageio.get_writer(gif_path, mode='I', fps=fps) as writer:
        for i, filename in enumerate(tqdm.tqdm(img_filenames)):
            image = imageio.imread(filename)
            writer.append_data(image)
