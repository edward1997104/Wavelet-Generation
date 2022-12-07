import pywt
import torch

debug_base_folder = r"../debug"


## training
# data
data_files = [('wavelet_data/03001627_0.1_bior6.8_3_zero.npy', 3), ('wavelet_data/03001627_0.1_bior6.8_2_zero.npy', 2)]
interval = 1
first_k = None
loss_function = torch.nn.MSELoss()
mix_precision = True

batch_size = 4
lr = 5e-5
lr_decay = False
lr_decay_feq = 500
lr_decay_rate = 0.998
data_worker = 24
beta1 = 0.9
beta2 = 0.999
optimizer = torch.optim.Adam

## network
resolution = 256
padding_mode = 'zero'
wavelet_type = 'bior6.8'
wavelet = pywt.Wavelet(wavelet_type)
max_depth = pywt.dwt_max_level(data_len = resolution, filter_len=wavelet.dec_len)
activation = torch.nn.LeakyReLU(0.02)
use_dense_conv = True
use_gradient_clip = False
gradient_clip_value = 1.0
use_instance_norm = True
use_instance_affine = True
use_layer_norm = False
use_layer_affine = False
training_stage = max_depth - 1
train_with_gt_coeff = True

### diffusion setting
from models.module.gaussian_diffusion import  ModelMeanType, ModelVarType, LossType
diffusion_step = 1000
diffusion_model_var_type = ModelVarType.FIXED_SMALL
diffusion_learn_sigma = False
diffusion_sampler = 'second-order'
diffusion_model_mean_type = ModelMeanType.EPSILON
diffusion_rescale_timestep = False
diffusion_loss_type = LossType.MSE
diffusion_beta_schedule = 'linear'
diffusion_scale_ratio = 1.0
unet_model_channels = 64
unet_num_res_blocks = 3
unet_channel_mult = (1, 1, 2, 4)
unet_channel_mult_low = (1, 2, 2, 2)
unet_activation = None
attention_resolutions = []
if diffusion_learn_sigma:
    diffusion_model_var_type = ModelVarType.LEARNED_RANGE
    diffusion_loss_type = LossType.RESCALED_MSE




## resume
starting_epoch = 0
training_epochs = 3000
saving_intervals = 10
special_symbol = ''
network_resume_path = None
optimizer_resume_path = None
discriminator_resume_path = None
discriminator_opt_resume_path = None
exp_idx = 0
