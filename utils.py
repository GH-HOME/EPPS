import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from sutils import diff_operators
from hutils.PhotometricStereoUtil import evalsurfaceNormal, evaldepth
from hutils.visualization import N_2_N_show, plt_error_map, save_plt_fig_with_title
from hutils.draw_3D import generate_mesh

def cond_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def write_image_summary_read_data_no_gt(image_resolution, model, model_input, gt,
                        model_output, writer, total_steps, prefix='train_', loss_val = 0, kwargs = None):

    save_folder = kwargs['save_folder']
    vmaxN, vmaxD, vmaxA = kwargs['vmaxNDA']
    mask = kwargs['mask']

    N_gt = None; depth_gt= None
    if 'N_gt' in kwargs:
        N_gt = kwargs['N_gt']
    if 'depth_gt' in kwargs:
        depth_gt = kwargs['depth_gt']

    cond_mkdir(save_folder)

    h, w = image_resolution
    batch_size, _, _ = model_output['model_out'].shape
    depth_est = np.zeros(image_resolution)
    depth_est[mask] =  model_output['model_out'].detach().cpu().numpy().squeeze()

    img_gradient = diff_operators.gradient(model_output['model_out'], model_output['model_in'])

    dx, dy = img_gradient[:, :, 0], img_gradient[:, :, 1]
    xx, yy = model_output['model_in'][:, :, 0], model_output['model_in'][:, :, 1]
    zz = model_output['model_out']
    du = dx.unsqueeze(2)
    dv = dy.unsqueeze(2)
    dz = torch.ones_like(du)

    focal_len, fx, fy, cx, cy, img_h, img_w = gt['cam_para'][0]  # cx = crop_center - camera_matrix_x in ordinary image coordinates
    m2pix_x = fx / focal_len * 1e3
    m2pix_y = fy / focal_len * 1e3
    focal_len = focal_len / 1000  # mm -> m

    sensor_xx, sensor_yy = -(img_w / 2 * xx.unsqueeze(2) + cx) / m2pix_x, \
                           -(img_h / 2 * yy.unsqueeze(2) + cy) / m2pix_y  # the coordinate in mm in the sensor
    dZ_sensor_x, dZ_sensor_y = -du * 2 * m2pix_x / img_w, -dv * m2pix_y * 2 / img_h  # dz/d_sensor_x = dz / d(sensor_width / 2 * xx) = 2 / sensor_width * dz / dxx
    nxp = dZ_sensor_x * focal_len
    nyp = dZ_sensor_y * focal_len
    nzp = - (zz + sensor_xx * dZ_sensor_x + sensor_yy * dZ_sensor_y)
    normal_set = torch.stack([nxp, nyp, nzp], dim=2).squeeze(3)

    N_norm = torch.norm(normal_set, p=2, dim=2)
    normal_dir = normal_set / N_norm.unsqueeze(2)

    normal_dir = normal_dir.detach().cpu().numpy()
    normal_map = np.zeros([h, w, 3])
    normal_map[mask] = normal_dir.squeeze()

    if N_gt is not None:
        error_map, mae, _ = evalsurfaceNormal(normal_map, N_gt, mask = mask)
        img_path = os.path.join(save_folder, 'iter_{:0>5d}_ang_err_{:.2f}.png'.format(total_steps, mae))
        plt_error_map(error_map, mask, vmax = vmaxN, withbar=True,
                      title = 'Iter: {:0>5d} MAE: {:.2f}'.format(total_steps, mae), img_path = img_path)
        print('==> Surface normal error: Iter_{:0>5d} MAngE: {:.2f}'.format(total_steps, mae))

    normal_show = normal_map.copy()
    normal_show[:, :, 0] *= (-1)
    normal_show[:, :, 2] *= (-1)
    plt.imshow(N_2_N_show(normal_show, mask))
    save_plt_fig_with_title(os.path.join(save_folder, 'iter_{:0>5d}_N_est.png'.format(total_steps)), 'Iter: {:0>5d} \n loss: {:.2e}'.format(total_steps, loss_val))
    np.save(os.path.join(save_folder, 'iter_{:0>5d}_N_est_w.npy'.format(total_steps)), normal_map)

    if depth_gt is not None:
        error_map, mabse, _ = evaldepth(depth_est, depth_gt, mask = mask)
        img_path = os.path.join(save_folder, 'iter_{:0>5d}_abs_err_{:.2e}.png'.format(total_steps, mabse))
        plt_error_map(error_map, mask, vmax=vmaxD, withbar=True,
                      title='Iter: {:0>5d} MAbsE: {:.2e}'.format(total_steps, mabse), img_path=img_path)
        print('==> Depth error: Iter_{:0>5d} MAbsE: {:.2e}'.format(total_steps, mabse))

    shape_file_name = os.path.join(save_folder, 'iter_{:0>5d}_Z_est.png'.format(total_steps))

    x_map = np.zeros([batch_size, h, w])
    y_map = np.zeros([batch_size, h, w])
    x_map[:, mask] = sensor_xx.detach().cpu().numpy().squeeze()
    y_map[:, mask] = sensor_yy.detach().cpu().numpy().squeeze()

    point_cloud = np.dstack([x_map[0]  * depth_est / focal_len.cpu().numpy(),
                           y_map[0]  * depth_est / focal_len.cpu().numpy(),
                           depth_est])


    point_cloud = point_cloud.reshape(h, w, 3)

    generate_mesh(point_cloud, mask, shape_file_name, window_size = (1024, 768), title = 'Iter: {:0>5d} \n \nloss: {:.2e}'.format(total_steps, loss_val))
    np.save(os.path.join(save_folder, 'iter_{:0>5d}_depth_est_w.npy'.format(total_steps)), depth_est)




def loadCheckpoint(path, model, cuda=True):
    if cuda:
        checkpoint = torch.load(path)
    else:
        checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint)