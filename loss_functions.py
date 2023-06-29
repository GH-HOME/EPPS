import torch
import torch.nn.functional as F
from sutils import diff_operators

L1_loss = torch.nn.L1Loss()
L2_loss = torch.nn.MSELoss()

def analytical_L1(mask, model_output, gt, total_steps, device):
    gradients = diff_operators.gradient(model_output['model_out'], model_output['model_in'])
    dx, dy = gradients[:, :, 0], gradients[:, :, 1]

    xx, yy = model_output['model_in'][:, :, 0], model_output['model_in'][:, :, 1]
    zz = model_output['model_out']
    du = dx.unsqueeze(2)
    dv = dy.unsqueeze(2)
    dz = torch.ones_like(du)

    if 'cam_para' in gt:
        # perspective projection
        focal_len, fx, fy, cx, cy, img_h, img_w  = gt['cam_para'][0]  # cx = crop_center - camera_matrix_x in ordinary image coordinates

        m2pix_x = (fx / focal_len) * 1e3  # number of pixel per meter
        m2pix_y = (fy / focal_len) * 1e3
        focal_len = focal_len / 1000 # mm -> m


        sensor_xx, sensor_yy = -(img_w / 2 * xx.unsqueeze(2) + cx) / m2pix_x,  -(img_h / 2 * yy.unsqueeze(2) + cy) / m2pix_y  # the coordinate in mm in the sensor
        dZ_sensor_x, dZ_sensor_y = -du * 2 * m2pix_x / img_w, -dv * m2pix_y * 2 / img_h # dz/d_sensor_x = dz / d(sensor_width / 2 * xx) = 2 / sensor_width * dz / dxx
        nxp = dZ_sensor_x * focal_len
        nyp = dZ_sensor_y * focal_len
        nzp = - (zz + sensor_xx*dZ_sensor_x + sensor_yy * dZ_sensor_y)
        normal_set = torch.stack([nxp, nyp, nzp], dim=2).squeeze(3)

        point_set = torch.stack([sensor_xx * zz / focal_len,
                                 sensor_yy * zz / focal_len,
                                 zz], dim=2).squeeze(3)

    else:
        # orthographic projection
        normal_set = torch.stack([du, dv, -dz], dim=2).squeeze(3)
        point_set = torch.stack([xx.unsqueeze(2), yy.unsqueeze(2), zz], dim=2).squeeze(3)
    N_norm = torch.norm(normal_set, p=2, dim=2)
    normal_dir = normal_set / N_norm.unsqueeze(2)


    # now we test use the rendering error for all image sequence
    batch_size, numPixel, numChannel, numLEDs = gt['img'].shape
    shading_list = []
    attach_shadow = torch.nn.ReLU()

    for i in range(numLEDs):
        LED_loc = gt['LED_loc'][:, i].unsqueeze(1)
        lights = LED_loc - point_set
        L_norm = torch.norm(lights, p=2, dim=2).unsqueeze(2)
        light_dir = lights / L_norm
        light_falloff = torch.pow(L_norm, -2)
        shading = torch.sum(light_dir * normal_dir, dim=2, keepdims=True)

        LED_pdir_product = (- light_dir * gt['LED_PDIR'][:, i]).sum(dim=2, keepdims=True)
        LED_pdir_product = attach_shadow(LED_pdir_product)
        light_aniso_ins = torch.pow(LED_pdir_product, gt['LED_mu'][:, i])

        img = light_falloff * shading * light_aniso_ins
        shading_list.append(img.unsqueeze(3))

    if torch.sum(torch.isnan(normal_dir)) >0:
        print('Nan Occurs')
    shading_set = torch.cat(shading_list, 3).repeat(1, 1, numChannel, 1)

    shading_set = attach_shadow(shading_set)
    shading_set[torch.isnan(shading_set)] = 0
    mask_cast_shadow = gt['cast_shadow_mask']
    shading_set[mask_cast_shadow] = 0

    #################################################################################
    # Calculate dependent albedo from current depth and surface normal
    ################################################################################
    shading_sum = (shading_set * shading_set).sum(dim = 3)
    albedo = (gt['img'] * shading_set).sum(dim = 3) / shading_sum
    albedo[torch.isnan(albedo)] = 0
    img_recons = shading_set * albedo.unsqueeze(3)


    img_loss_all = L1_loss(gt['img'][~mask_cast_shadow], img_recons[~mask_cast_shadow])
    if img_loss_all < 1e-5:
        print('Convergent to {} at iteration {}'.format(img_loss_all, total_steps))
        img_loss_all = torch.tensor(-1.0)
        return { 'img_loss': img_loss_all}
    else:
        return { 'img_loss': img_loss_all}




def finite_L1(mask, model_output, gt, total_steps, device):

    xx, yy = model_output['model_in'][:, :, 0], model_output['model_in'][:, :, 1]
    zz = model_output['model_out']


    focal_len, fx, fy, cx, cy, img_h, img_w  = gt['cam_para'][0]
    scale_u = img_h / 2
    scale_v = img_w / 2
    filter_x = torch.tensor([[0, 0., 0], [1., 0., -1.], [0., 0., 0.]])
    filter_y = torch.tensor([[0, 1., 0.], [0., 0., 0], [0., -1, 0.]])
    fx = filter_x.expand(1, 1, 3, 3).to(device)
    fy = filter_y.expand(1, 1, 3, 3).to(device)

    zz_map = torch.zeros([1, int(img_h), int(img_w)]).to(device)
    mask = mask > 0
    mask = mask.squeeze().reshape(int(img_h), int(img_w))
    zz_map[:, mask] = zz.squeeze(2)
    p = -torch.nn.functional.conv2d(zz_map.reshape(1, 1, int(img_h), int(img_w)), fx, stride=1, padding=1)[0]  * scale_u / 2 # 2 is for the sum of the convolution
    q = -torch.nn.functional.conv2d(zz_map.reshape(1, 1, int(img_h), int(img_w)), fy, stride=1, padding=1)[0] * scale_v / 2 # 2 is for the sum of the convolution
    dx = p.reshape(1, int(img_h), int(img_w))[:, mask]
    dy = q.reshape(1, int(img_h), int(img_w))[:, mask]

    du = dx.unsqueeze(2)
    dv = dy.unsqueeze(2)
    dz = torch.ones_like(du)


    if 'cam_para' in gt:
        # perspective projection
        focal_len, fx, fy, cx, cy, img_h, img_w  = gt['cam_para'][0]  # cx = crop_center - camera_matrix_x in ordinary image coordinates

        m2pix_x = (fx / focal_len) * 1e3  # number of pixel per meter
        m2pix_y = (fy / focal_len) * 1e3
        focal_len = focal_len / 1000 # mm -> m

        sensor_xx, sensor_yy = -(img_w / 2 * xx.unsqueeze(2) + cx) / m2pix_x,  -(img_h / 2 * yy.unsqueeze(2) + cy) / m2pix_y  # the coordinate in mm in the sensor
        dZ_sensor_x, dZ_sensor_y = -du * 2 * m2pix_x / img_w, -dv * m2pix_y * 2 / img_h # dz/d_sensor_x = dz / d(sensor_width / 2 * xx) = 2 / sensor_width * dz / dxx
        nxp = dZ_sensor_x * focal_len
        nyp = dZ_sensor_y * focal_len
        nzp = - (zz + sensor_xx*dZ_sensor_x + sensor_yy * dZ_sensor_y)
        normal_set = torch.stack([nxp, nyp, nzp], dim=2).squeeze(3)

        point_set = torch.stack([sensor_xx * zz / focal_len,
                                 sensor_yy * zz / focal_len,
                                 zz], dim=2).squeeze(3)

    else:
        # orthographic projection
        normal_set = torch.stack([du, dv, -dz], dim=2).squeeze(3)
        point_set = torch.stack([xx.unsqueeze(2), yy.unsqueeze(2), zz], dim=2).squeeze(3)
    N_norm = torch.norm(normal_set, p=2, dim=2)
    normal_dir = normal_set / N_norm.unsqueeze(2)


    # now we test use the rendering error for all image sequence
    batch_size, numPixel, numChannel, numLEDs = gt['img'].shape
    shading_list = []
    attach_shadow = torch.nn.ReLU()

    for i in range(numLEDs):
        LED_loc = gt['LED_loc'][:, i].unsqueeze(1)
        lights = LED_loc - point_set
        L_norm = torch.norm(lights, p=2, dim=2).unsqueeze(2)
        light_dir = lights / L_norm
        light_falloff = torch.pow(L_norm, -2)
        shading = torch.sum(light_dir * normal_dir, dim=2, keepdims=True)

        LED_pdir_product = (- light_dir * gt['LED_PDIR'][:, i]).sum(dim=2, keepdims=True)
        LED_pdir_product = attach_shadow(LED_pdir_product)
        light_aniso_ins = torch.pow(LED_pdir_product, gt['LED_mu'][:, i])

        img = light_falloff * shading * light_aniso_ins
        shading_list.append(img.unsqueeze(3))


    shading_set = torch.cat(shading_list, 3).repeat(1, 1, numChannel, 1)

    shading_set = attach_shadow(shading_set)
    shading_set[torch.isnan(shading_set)] = 0
    mask_cast_shadow = gt['cast_shadow_mask']
    shading_set[mask_cast_shadow] = 0

    shading_sum = (shading_set * shading_set).sum(dim = 3)
    albedo = (gt['img'] * shading_set).sum(dim = 3) / shading_sum
    albedo[torch.isnan(albedo)] = 0
    img_recons = shading_set * albedo.unsqueeze(3)

    img_loss_all = L1_loss(gt['img'][~mask_cast_shadow], img_recons[~mask_cast_shadow])

    return {
            'img_loss': img_loss_all,
            }

