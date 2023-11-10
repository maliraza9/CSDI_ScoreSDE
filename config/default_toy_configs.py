# import ml_collections
import torch





#
#
# # import ml_collections
# import torch
#
#
# def get_default_configs():
#   config = ml_collections.ConfigDict()
#   # training
#   config.training = training = ml_collections.ConfigDict()
#   config.training.batch_size = 16
#   training.n_iters = 178
#   # training.snapshot_freq = 50000
#   training.snapshot_freq = 50
#   training.log_freq = 50
#   training.eval_freq = 100
#   ## store additional checkpoints for preemption in cloud computing environments
#   training.snapshot_freq_for_preemption = 10000
#   ## produce samples at each snapshot.
#   training.snapshot_sampling = False
#   training.likelihood_weighting = False
#   training.continuous = True
#   training.reduce_mean = True
#
#   # sampling
#   config.sampling = sampling = ml_collections.ConfigDict()
#   sampling.n_steps_each = 1
#   sampling.noise_removal = True
#   sampling.probability_flow = False
#   sampling.snr = 0.16
#   sampling.method = 'pc'
#
#   # evaluation
#   config.eval = evaluate = ml_collections.ConfigDict()
#   evaluate.begin_ckpt = 1
#   evaluate.end_ckpt = 3
#   evaluate.batch_size = 16
#   evaluate.enable_sampling = True
#   evaluate.num_samples = 50000
#   evaluate.enable_loss = True
#   evaluate.enable_bpd = False
#   evaluate.bpd_dataset = 'test'
#
#   # data
#   config.data = data = ml_collections.ConfigDict()
#   data.dataset = 'TOY'
#   data.time_series = 51
#   data.variable_size = 2
#   data.random_flip = False
#   data.centered = False
#   data.uniform_dequantization = False
#   data.num_channels = 2
#
#   # model
#   config.model = model = ml_collections.ConfigDict()
#   model.name = 'ddpm'
#   model.sigma_min = 0.01
#   model.sigma_max = 50
#   model.num_scales = 1000
#   model.beta_min = 0.1
#   model.beta_max = 20.
#   model.dropout = 0.1
#   model.embedding_type = 'fourier'
#
#   # optimization
#   config.optim = optim = ml_collections.ConfigDict()
#   optim.weight_decay = 0
#   optim.optimizer = 'Adam'
#   optim.lr = 2e-4
#   optim.beta1 = 0.9
#   optim.eps = 1e-8
#   optim.warmup = 5000
#   optim.grad_clip = 1.
#
#   config.seed = 42
#   config.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
#   # config.device = 'cpu'
#
#   return config