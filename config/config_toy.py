# # coding=utf-8
# # Copyright 2020 The Google Research Authors.
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.
#
# # Lint as: python3
# """Config file for reproducing the results of DDPM on cifar-10."""
# import torch
# # from config.default_toy_configs import get_default_configs
#
#
# def get_config():
#   # config = get_default_configs()
#   #
#   # training
#   training = get_default_configs.config_training
#   training.sde = 'vesde'
#   training.continuous = True
#   training.reduce_mean = True
#
#   # sampling
#   # sampling = config.sampling
#   sampling.method = 'pc'
#   sampling.predictor = 'reverse_diffusion'
#   sampling.corrector = 'langevin'
#
#   # data
#   # data = config.data
#   data.centered = True
#
#   # model
#   # model = config.model
#   model.name = 'csdi'
#   model.scale_by_sigma = False
#   model.ema_rate = 0.9999
#   model.normalization = 'GroupNorm'
#   model.nonlinearity = 'swish'
#   model.nf = 128
#   model.ch_mult = (1,  2)
#   # model.ch_mult = (1, 2, 2)
#   #model.ch_mult = (1, 2, 2, 2)
#   model.num_res_blocks = 2
#   # model.attn_resolutions = (16,)
#   model.attn_resolutions = (16,)
#   model.resamp_with_conv = True
#   model.conditional = True
#
#
# def get_default_configs():
#   # config = ml_collections.ConfigDict()
#   # training
#   # config.training = training = ml_collections.ConfigDict()
#   config_training_batch_size = 16
#   training_n_iters = 178
#   # training.snapshot_freq = 50000
#   training_snapshot_freq = 50
#   training_log_freq = 50
#   training_eval_freq = 100
#   ## store additional checkpoints for preemption in cloud computing environments
#   training_snapshot_freq_for_preemption = 10000
#   ## produce samples at each snapshot.
#   training_snapshot_sampling = False
#   training_likelihood_weighting = False
#   training_continuous = True
#   training_reduce_mean = True
#
#   # sampling
#   # config.sampling = sampling = ml_collections.ConfigDict()
#   sampling_n_steps_each = 1
#   sampling_noise_removal = True
#   sampling_probability_flow = False
#   sampling_snr = 0.16
#   sampling_method = 'pc'
#
#   # evaluation
#   # config.eval = evaluate = ml_collections.ConfigDict()
#   evaluate_begin_ckpt = 1
#   evaluate_end_ckpt = 3
#   evaluate_batch_size = 16
#   evaluate_enable_sampling = True
#   evaluate_num_samples = 50000
#   evaluate_enable_loss = True
#   evaluate_enable_bpd = False
#   evaluate_bpd_dataset = 'test'
#
#   # data
#   # config.data = data = ml_collections.ConfigDict()
#   data_dataset = 'TOY'
#   data_time_series = 51
#   data_variable_size = 2
#   data_random_flip = False
#   data_centered = False
#   data_uniform_dequantization = False
#   data_num_channels = 2
#
#   # model
#   # config.model = model = ml_collections.ConfigDict()
#   model_name = 'ddpm'
#   model_sigma_min = 0.01
#   model_sigma_max = 50
#   model_num_scales = 1000
#   model_beta_min = 0.1
#   model_beta_max = 20.
#   model_dropout = 0.1
#   model_embedding_type = 'fourier'
#
#   # optimization
#   # config.optim = optim = ml_collections.ConfigDict()
#   optim_weight_decay = 0
#   optim_optimizer = 'Adam'
#   optim_lr = 2e-4
#   optim_beta1 = 0.9
#   optim_eps = 1e-8
#   optim_warmup = 5000
#   optim_grad_clip = 1.
#
#   config_seed = 42
#   config_device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
#   # config.device = 'cpu'
#
#   return