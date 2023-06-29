import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"]= "TRUE"
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )
from datetime import datetime
import dataio, utils, training, loss_functions, modules

from torch.utils.data import DataLoader
import configargparse
from functools import partial
import torch
from torchvision.transforms import ToTensor
import numpy as np
import configparser

p = configargparse.ArgumentParser()

p.add_argument('--code_id', type=str, default='TPAMI_submit', help='git commid id for the running')
p.add_argument('--experiment_name', type=str, default='Result_Ours', required=False,
               help='Name of subdirectory to save result.')

# General training options
p.add_argument('--batch_size', type=int, default=1)
p.add_argument('--lr', type=float, default=1e-4, help='learning rate. default=1e-4')
p.add_argument('--num_epochs', type=int, default=100,
               help='Number of epochs to train for.')

p.add_argument('--epochs_til_ckpt', type=int, default=1000,
               help='Time interval in seconds until checkpoint is saved.')
p.add_argument('--steps_til_summary', type=int, default=1000,
               help='Time interval in seconds until internal result is saved.')
p.add_argument('--net_type', type=str, default='FCRes',
               help='The network structure, FC or FCRes ')

p.add_argument('--color_channel', type=bool, default=True, help='whether to use color channels')
p.add_argument('--is_flip', type=bool, default=False, help='whether to flip image coordinates')
p.add_argument('--cast_shadow_ratio', type=float, default=0.05, help='threshold for determine the cast shadow')

p.add_argument('--checkpoint_path', default=None, help='Checkpoint path of trained model.')
p.add_argument('--data_folder', type=str, default=None, help='Path to data')
p.add_argument('--img_name', type=str, default='img_sv_albedo.npy', help='image name')
p.add_argument('--custom_depth_offset', type=float, default=3, help='initial depth from the LED position')
p.add_argument('--difference', type=str, default='analytical', help='whether to use finite difference or the analytical difference')
p.add_argument('--hidden_features', type=int, default=256, help='number of hidden features')
p.add_argument('--sv_albedo', type=bool, default=True, help='whether to use SV albedo')
p.add_argument('--gpu_id', type=int, default=1, help='GPU ID')
p.add_argument('--env', type=str, default='win32', help='system environment')


opt = p.parse_args()
print('Now process {}'.format(opt.data_folder))

if opt.env == 'linux':
    from matplotlib import pyplot as plt
    plt.switch_backend('agg')
    import pyvista as pv
    pv.start_xvfb()


device = torch.device("cuda:{gpu}".format(gpu=opt.gpu_id))

# load data
custom_mask = os.path.join(opt.data_folder, 'render_para/mask.npy')
custom_image = os.path.join(opt.data_folder, 'render_img/{}'.format(opt.img_name))
custom_LEDs = os.path.join(opt.data_folder, 'render_para/LED_locs.npy')
custom_mu = os.path.join(opt.data_folder, 'render_para/mu.npy')
custom_LED_PDIR = os.path.join(opt.data_folder, 'render_para/LED_principle_dir.npy')
custom_camera_para = os.path.join(opt.data_folder, 'render_para/save.ini')
camera_para_config = configparser.ConfigParser()
camera_para_config.optionxform = str
camera_para_config.read(custom_camera_para)

camera_para = np.array([float(camera_para_config['configInfo']['focal_len']), # camera lens in mm
                        float(camera_para_config['configInfo']['fx']), # fx in mm
                        float(camera_para_config['configInfo']['fy']), # fy in mm
                        float(camera_para_config['configInfo']['cx']),
                        float(camera_para_config['configInfo']['cy']),
                        float(camera_para_config['configInfo']['img_h']),
                        float(camera_para_config['configInfo']['img_w'])
                        ])

# load GT
custom_depth = os.path.join(opt.data_folder, 'render_para/depth.npy')
custom_normal = os.path.join(opt.data_folder, 'render_para/normal_world.npy')
custom_albedo = os.path.join(opt.data_folder, 'render_para/albedo.npy')
if not os.path.exists(custom_depth):
    custom_depth = None
if not os.path.exists(custom_normal):
    custom_normal = None
if not os.path.exists(custom_albedo):
    custom_albedo = None


img_dataset = dataio.Shading_LEDNPY(custom_image, custom_LEDs, custom_mask, custom_normal,
                                    custom_depth, camera_para, custom_albedo, custom_mu, custom_LED_PDIR,
                                    opt.color_channel, opt.cast_shadow_ratio)

if len(img_dataset[0]['img'].shape) == 3:
    numImg, h, w = img_dataset[0]['img'].shape
elif len(img_dataset[0]['img'].shape) == 4:
    numImg, h, w, numChannel = img_dataset[0]['img'].shape
elif len(img_dataset[0]['img'].shape) == 2:
    h, w = img_dataset[0]['img'].shape
else:
    raise Exception('Image channel is not fit.')

image_resolution = (h, w)
coord_dataset = dataio.Implicit2DWrapper(img_dataset, image_resolution, is_flip=opt.is_flip)

dataloader = DataLoader(coord_dataset, shuffle=False, batch_size=opt.batch_size, pin_memory=True, num_workers=0)

if custom_mask:
    mask = np.load(custom_mask)
    mask = ToTensor()(mask)
    mask = mask.float().to(device)
else:
    mask = torch.ones(image_resolution)
    mask = mask.float().to(device)

# Define the model.
model = modules.SingleBVPNet(type='sine', mode='mlp', out_features=1,
                                 sidelength=image_resolution, num_hidden_layers = 5, hidden_features = opt.hidden_features,
                                 net_type = opt.net_type,
                                 last_layer_offset = opt.custom_depth_offset)

model.to(device)
now = datetime.now() # current date and time
date_time = now.strftime("%Y_%m_%d_%H_%M_%S")
root_path = os.path.join(opt.data_folder, opt.experiment_name, '{}_{}'.format(date_time, opt.code_id))

if opt.difference == 'analytical':
    print("Loss function: L1 analytical derivative re-rendering loss")
    loss_fn = partial(loss_functions.analytical_L1, mask.view(-1,1), device = device)
elif opt.difference == 'finite':
    print("Loss function: L1 finite difference re-rendering loss")
    loss_fn = partial(loss_functions.finite_L1, mask.view(-1,1), device = device)
else:
    raise Exception('Unknown loss type')

summary_fn = partial(utils.write_image_summary_read_data_no_gt, image_resolution)

kwargs = {'save_folder': os.path.join(root_path, 'Recoverd_Shapes'),
          'vmaxNDA': [10, 0.1, 0.1],
          'mask': np.load(custom_mask)}

if custom_albedo is not None:
    kwargs['albedo_gt'] = np.load(custom_albedo)
    kwargs['imgs'] = np.load(custom_image)
    kwargs['LED_loc'] = np.load(custom_LEDs)

if custom_depth is not None:
    kwargs['depth_gt'] = np.load(custom_depth)
if custom_normal is not None:
    kwargs['N_gt'] = np.load(custom_normal)

training.train(model=model, train_dataloader=dataloader, epochs=opt.num_epochs, lr=opt.lr,
               steps_til_summary=opt.steps_til_summary, epochs_til_checkpoint=opt.epochs_til_ckpt,
               model_dir=root_path, loss_fn=loss_fn, summary_fn=summary_fn, use_lbfgs = False, kwargs = kwargs,
               save_state_path = opt.checkpoint_path, clip_grad = False, device = device)
