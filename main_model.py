import numpy as np
import torch
import torch.nn as nn
from diff_models import diff_CSDI
import sde_lib
import dataset_toy
import sampling
import utils_sde
# from config.config_toy import get_config



# config = get_config()
centered = True
import functools

inverse_scalar = dataset_toy.get_data_inverse_scaler(centered=centered)

from sampling import NonePredictor
from sampling import NoneCorrector




eps=1e-3
num_step = 2
sigma_min = 0.1
sigma_max = 20
num_scales = 100
dropout = 0.1
sde_T = 1
sde_N = 3
rsde_N = 3
reduce_mean = True
reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)
denoise = True
beta_high = 0.1
beta_low = 20

beta_max = beta_high
beta_min = beta_low


def new_marginal_prob(x, t):
    log_mean_coeff = -0.25 * t ** 2 * (beta_high - beta_low) - 0.5 * t * beta_low
    mean = torch.exp(log_mean_coeff[:, None, None]) * x
    std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
    return mean, std


def new_sde(x, t):
    beta_t = beta_low + t * (beta_high - beta_low)
    drift = -0.5 * beta_t[:, None, None] * x
    diffusion = torch.sqrt(beta_t)
    return drift, diffusion



# emb_time_dimensions = 128
# emb_feature_dimensions = 16
# embeddingDimensions = emb_time_dimensions + emb_feature_dimensions
#
# isit_unconditional = False
# input_dimensions = 1 if isit_unconditional == True else 2
# diffmodelGlobal = diff_CSDI(embeddingDimensions, input_dimensions)




def shared_predictor_update_fn(diff_input, inp, vec_t, side_info, sde, model, predictor, probability_flow, continuous):
  """A wrapper that configures and returns the update function of predictors."""
  score_fn = utils_sde.get_score_fn(sde, model, train=False, continuous=continuous)

  tmp_side_info = side_info
  side_info = vec_t
  vec_t = tmp_side_info
  # inp = diff_input
  if predictor is None:
    # diff_input = diff_input.to(x.device)  # Move diff_input to the same device as x

    # Corrector-only sampler
    predictor_obj = NonePredictor(sde, score_fn, probability_flow)
  else:
    # diff_input = diff_input.to(x.device)  # Move diff_input to the same device as x

    predictor_obj = predictor(sde, score_fn, probability_flow)
  return predictor_obj.update_fn(diff_input, inp, side_info, vec_t)

def shared_corrector_update_fn(diff_input, x, vec_t, side_info, sde, model, corrector, continuous, snr, n_steps):
  """A wrapper tha configures and returns the update function of correctors."""
  tmp_side_info = side_info
  side_info = vec_t
  vec_t = tmp_side_info
  score_fn = utils_sde.get_score_fn(sde, model, train=False, continuous=continuous)

  if corrector is None:
    # Predictor-only sampler
    corrector_obj = NoneCorrector(sde, score_fn, snr, n_steps)
  else:
    corrector_obj = corrector(sde, score_fn, snr, n_steps)
  return corrector_obj.update_fn(diff_input, x, side_info, vec_t)



use_sde = 'vesde'  # or 'vesde' or 'subvpsde'

if use_sde == 'vpsde':
    sde = sde_lib.VPSDE(beta_min=beta_min, beta_max=beta_max, N=num_scales)
    sampling_eps = 1e-3
elif use_sde == 'subvpsde':
    sde = sde_lib.subVPSDE(beta_min=beta_min, beta_max=beta_max, N=num_scales)
    sampling_eps = 1e-3
elif use_sde == 'vesde':
    sde = sde_lib.VESDE(sigma_min=sigma_min, sigma_max=sigma_max, N=num_scales)
    sampling_eps = 1e-5
else:
    raise NotImplementedError(f"SDE {use_sde} unknown.")


predictor_update_fn = functools.partial(shared_predictor_update_fn,
                                          sde=sde,
                                          predictor=sampling.EulerMaruyamaPredictor,
                                          # predictor=sampling.ReverseDiffusionPredictor,
                                          probability_flow=False,
                                          continuous=True)

corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                        sde=sde,
                                        # corrector=sampling.AnnealedLangevinDynamics,
                                        corrector=None,
                                        continuous=True,
                                        snr=0.16,
                                        n_steps=1)
class CSDI_base(nn.Module):
    def __init__(self, target_dim, config, device):
        super().__init__()
        self.device = device
        self.target_dim = target_dim

        self.emb_time_dim = config["model"]["timeemb"]
        self.emb_feature_dim = config["model"]["featureemb"]
        self.is_unconditional = config["model"]["is_unconditional"]
        self.target_strategy = config["model"]["target_strategy"]

        self.emb_total_dim = self.emb_time_dim + self.emb_feature_dim
        if self.is_unconditional == False:
            self.emb_total_dim += 1  # for conditional mask
        self.embed_layer = nn.Embedding(
            num_embeddings=self.target_dim, embedding_dim=self.emb_feature_dim
        )

        config_diff = config["diffusion"]
        config_diff["side_dim"] = self.emb_total_dim

        input_dim = 1 if self.is_unconditional == True else 2
        self.diffmodel = diff_CSDI(config_diff, input_dim)

        # parameters for diffusion models
        self.num_steps = config_diff["num_steps"]
        if config_diff["schedule"] == "quad":
            self.beta = np.linspace(
                config_diff["beta_start"] ** 0.5, config_diff["beta_end"] ** 0.5, self.num_steps
            ) ** 2
        elif config_diff["schedule"] == "linear":
            self.beta = np.linspace(
                config_diff["beta_start"], config_diff["beta_end"], self.num_steps
            )

        self.alpha_hat = 1 - self.beta
        self.alpha = np.cumprod(self.alpha_hat)
        self.alpha_torch = torch.tensor(self.alpha).float().to(self.device).unsqueeze(1).unsqueeze(1)

    def time_embedding(self, pos, d_model=128):
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(self.device)
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow(
            10000.0, torch.arange(0, d_model, 2).to(self.device) / d_model
        )
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    def get_randmask(self, observed_mask):
        rand_for_mask = torch.rand_like(observed_mask) * observed_mask
        rand_for_mask = rand_for_mask.reshape(len(rand_for_mask), -1)
        for i in range(len(observed_mask)):
            sample_ratio = np.random.rand()  # missing ratio
            num_observed = observed_mask[i].sum().item()
            num_masked = round(num_observed * sample_ratio)
            rand_for_mask[i][rand_for_mask[i].topk(num_masked).indices] = -1
        cond_mask = (rand_for_mask > 0).reshape(observed_mask.shape).float()
        return cond_mask

    def get_hist_mask(self, observed_mask, for_pattern_mask=None):
        if for_pattern_mask is None:
            for_pattern_mask = observed_mask
        if self.target_strategy == "mix":
            rand_mask = self.get_randmask(observed_mask)

        cond_mask = observed_mask.clone()
        for i in range(len(cond_mask)):
            mask_choice = np.random.rand()
            if self.target_strategy == "mix" and mask_choice > 0.5:
                cond_mask[i] = rand_mask[i]
            else:  # draw another sample for histmask (i-1 corresponds to another sample)
                cond_mask[i] = cond_mask[i] * for_pattern_mask[i - 1]
        return cond_mask

    def get_side_info(self, observed_tp, cond_mask):
        B, K, L = cond_mask.shape

        time_embed = self.time_embedding(observed_tp, self.emb_time_dim)  # (B,L,emb)
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, K, -1)
        feature_embed = self.embed_layer(
            torch.arange(self.target_dim).to(self.device)
        )  # (K,emb)
        feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)

        side_info = torch.cat([time_embed, feature_embed], dim=-1)  # (B,L,K,*)
        side_info = side_info.permute(0, 3, 2, 1)  # (B,*,K,L)

        if self.is_unconditional == False:
            side_mask = cond_mask.unsqueeze(1)  # (B,1,K,L)
            side_info = torch.cat([side_info, side_mask], dim=1)

        return side_info

    def calc_loss_valid(
        self, observed_data, cond_mask, observed_mask, side_info, is_train
    ):
        loss_sum = 0
        for t in range(self.num_steps):  # calculate loss for all t
            loss = self.calc_loss(
                observed_data, cond_mask, observed_mask, side_info, is_train, set_t=t
            )
            loss_sum += loss.detach()
        return loss_sum / self.num_steps




    def calc_loss(
        self, observed_data, cond_mask, observed_mask, side_info, is_train, set_t=-1
    ):
        B, K, L = observed_data.shape

        #1. initial t setting
        if is_train != 1:  # for validation
            # t = (torch.ones(B) * set_t).long().to(self.device)

            t = torch.rand(observed_data.shape[0], device=observed_data.device) * (sde_T - eps) + eps

        else:
            # originla CSDI
            # t = torch.randint(0, self.num_steps, [B]).to(self.device)
            # Score SDE:
            # t = torch.rand(batch.shape[0], device=batch.device) * (sde.T - eps) + eps
            t = torch.rand(observed_data.shape[0], device=observed_data.device) * (sde_T - eps) + eps


        #2. Get noise scheduling

        z = torch.randn_like(observed_data)
        mean, std = new_marginal_prob(observed_data, t)
        perturbed_data = mean + std[:, None, None] * z

        #3. Score


        total_input = self.set_input_to_diffmodel(perturbed_data, observed_data, cond_mask)
        labels = t * num_step
        score = self.diffmodel(total_input, side_info, t.long()) # (B,K,L)
        # score = self.diffmodel(total_input, side_info, labels.long())
        std = new_marginal_prob(torch.zeros_like(perturbed_data), t)[1]
        # std = sde.marginal_prob(torch.zeros_like(observed_data), t)[1]
        # std = std.to(t.device)
        # score = -score / std[:, None, None]

        #4. Losses
        #
        likelihood_weighting = True
        if not likelihood_weighting:
            losses = torch.square(score * std[:, None, None] + z)
            losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
        else:
            g2 = new_sde(torch.zeros_like(observed_data), t)[1] ** 2
            losses = torch.square(score + z / std[:, None, None])
            losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1) * g2

        loss = torch.mean(losses)
        #
        # # #include std new
        # #
        target_mask = observed_mask - cond_mask
        residual = (z - score) * target_mask
        num_eval = target_mask.sum()
        loss = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)

        return loss


    def set_input_to_diffmodel(self, noisy_data, observed_data, cond_mask):
        if self.is_unconditional == True:
            total_input = noisy_data.unsqueeze(1)  # (B,1,K,L)
        else:

            cond_obs = (cond_mask * observed_data).unsqueeze(1)
                                                             #choose
            noisy_target = ((1 - cond_mask) * noisy_data).unsqueeze(1)

            total_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B,2,K,L)

        return total_input


    def impute(self, observed_data, cond_mask, side_info, n_samples):
        B, K, L = observed_data.shape

        imputed_samples = torch.zeros(B, n_samples, K, L).to(self.device)

        def impute_marginal_prob(x, t):
            log_mean_coeff = -0.25 * t ** 2 * (beta_high - beta_low) - 0.5 * t * beta_low
            mean = torch.exp(log_mean_coeff[:, None, None]) * x
            std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
            return mean, std

        def impute_score_fn(diff_input, x, side_info, t):
            labels = t * num_step
            labels = labels.to(x.device)
            # score = self.diffmodel(diff_input, side_info, labels.long())
            score = self.diffmodel(diff_input, side_info, t.long())
            std = impute_marginal_prob(torch.zeros_like(x), t)[1]
            # score = -score / std[:, None, None]
            return score

        def impute_sde_fn(x, t):
            beta_t = beta_low + t * (beta_high - beta_low)
            drift = -0.5 * beta_t[:, None, None] * x
            diffusion = torch.sqrt(beta_t)
            return drift, diffusion

        probability_flow = False

        def impute_rsde_sde(diff_input, x, side_info, t):
            """Create the drift and diffusion functions for the reverse SDE/ODE."""
            drift, diffusion = impute_sde_fn(x, t)
            score = impute_score_fn(diff_input, x, side_info, t)
            # score = score_fn(x, t)
            drift = drift - diffusion[:, None, None] ** 2 * score * (
                0.5 if probability_flow else 1.)
            # drift = drift - diffusion[:, None, None, None] ** 2 * score * (0.5 if self.probability_flow else 1.)
            # Set the diffusion function to zero for ODEs.
            diffusion = 0. if probability_flow else diffusion
            return drift, diffusion

        def EulerMaruyamaPredictor_update_fn(diff_input, x, side_info, t):
            dt = -1. / rsde_N
            z = torch.randn_like(x)
            drift, diffusion = impute_rsde_sde(diff_input, x, side_info, t)
            x_mean = x + drift * dt
            x = x_mean + diffusion[:, None, None] * np.sqrt(-dt) * z
            return x, x_mean

        def impute_prior_sampling(shape):
            return torch.randn(*shape)

        # n_samples = 1
        for i in range(n_samples):
            # generate noisy observation for unconditional model
            if self.is_unconditional == True:
                noisy_obs = observed_data
                noisy_cond_history = []
                for t in range(self.num_steps):
                    noise = torch.randn_like(noisy_obs)
                    noisy_obs = (self.alpha_hat[t] ** 0.5) * noisy_obs + self.beta[t] ** 0.5 * noise
                    noisy_cond_history.append(noisy_obs * cond_mask)

            current_sample = torch.randn_like(observed_data)

            for n in range(self.num_steps - 1, -1, -1):
                # if self.is_unconditional == True:
                #     diff_input = cond_mask * noisy_cond_history[t] + (1.0 - cond_mask) * current_sample
                #     diff_input = diff_input.unsqueeze(1)  # (B,1,K,L)
                # else:
                    # cond_obs = (cond_mask * observed_data).unsqueeze(1)
                    # noisy_target = ((1 - cond_mask) * current_sample).unsqueeze(1)
                    # diff_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B,2,K,L) ## 2 for cond input, noisy target

                updated_observed_data = observed_data

                timesteps = torch.linspace(sde_T, eps, sde_N, device=self.device)
                for i in range(sde_N):
                    t = timesteps[i]
                    # observed_data = x
                    cond_obs = (cond_mask * updated_observed_data).unsqueeze(1)
                    noisy_target = ((1 - cond_mask) * current_sample).unsqueeze(1)
                    diff_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B,2,K,L) ## 2 for cond input, noisy target
                    vec_t = torch.ones(updated_observed_data.shape[0], device=self.device) * t
                    # x, x_mean = corrector_update_fn(x, vec_t, model=model)
                    # updated_observed_data, updated_observed_data_mean = EulerMaruyamaPredictor_update_fn(diff_input, updated_observed_data, side_info, vec_t)
                    x, x_mean = EulerMaruyamaPredictor_update_fn(diff_input, updated_observed_data, side_info, vec_t)
                    updated_observed_data = x
                # current_sample = updated_observed_data_mean if denoise else updated_observed_data
                current_sample = x_mean if denoise else x
            imputed_samples[:, n] = current_sample.detach()

        return imputed_samples




    def forward(self, batch, is_train=1):
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            _,
        ) = self.process_data(batch)
        if is_train == 1:   # check thisss
            cond_mask = gt_mask
        elif self.target_strategy != "random":
            cond_mask = self.get_hist_mask(
                observed_mask, for_pattern_mask=for_pattern_mask
            )
        else:
            cond_mask = self.get_randmask(observed_mask)

        side_info = self.get_side_info(observed_tp, cond_mask)

        loss_func = self.calc_loss if is_train == 1 else self.calc_loss_valid

        return loss_func(observed_data, cond_mask, observed_mask, side_info, is_train)

    def evaluate(self, batch, n_samples):
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            _,
            cut_length,
        ) = self.process_data(batch)

        with torch.no_grad():
            cond_mask = gt_mask
            target_mask = observed_mask - cond_mask

            side_info = self.get_side_info(observed_tp, cond_mask)

            samples = self.impute(observed_data, cond_mask, side_info, n_samples)

            for i in range(len(cut_length)):  # to avoid double evaluation
                target_mask[i, ..., 0 : cut_length[i].item()] = 0
        return samples, observed_data, target_mask, observed_mask, observed_tp


class CSDI_PM25(CSDI_base):
    def __init__(self, config, device, target_dim=36):
        super(CSDI_PM25, self).__init__(target_dim, config, device)

    def process_data(self, batch):
        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch["observed_mask"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        gt_mask = batch["gt_mask"].to(self.device).float()
        cut_length = batch["cut_length"].to(self.device).long()
        for_pattern_mask = batch["hist_mask"].to(self.device).float()

        observed_data = observed_data.permute(0, 2, 1)
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)
        for_pattern_mask = for_pattern_mask.permute(0, 2, 1)

        return (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
        )


class CSDI_Physio(CSDI_base):
    def __init__(self, config, device, target_dim=35):
        super(CSDI_Physio, self).__init__(target_dim, config, device)

    def process_data(self, batch):
        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch["observed_mask"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        gt_mask = batch["gt_mask"].to(self.device).float()

        observed_data = observed_data.permute(0, 2, 1)
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)

        cut_length = torch.zeros(len(observed_data)).long().to(self.device)
        for_pattern_mask = observed_mask

        return (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
        )

class CSDI_Toy(CSDI_base):
    def __init__(self, config, device, target_dim=2):
        super(CSDI_Toy, self).__init__(target_dim, config, device)

    def process_data(self, batch):
        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch["observed_mask"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        gt_mask = batch["gt_mask"].to(self.device).float()

        observed_data = observed_data.permute(0, 2, 1)
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)

        cut_length = torch.zeros(len(observed_data)).long().to(self.device)
        for_pattern_mask = observed_mask

        return (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
        )
