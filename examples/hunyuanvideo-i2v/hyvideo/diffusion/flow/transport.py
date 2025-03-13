import enum
import math
from typing import Callable

import numpy as np
from hyvideo.constants import PRECISION_TO_TYPE

import mindspore as ms
from mindspore import ops

from . import path
from .integrators import ode, sde
from .utils import mean_flat

__all__ = ["ModelType", "PathType", "WeightType", "Transport", "Sampler", "SNRType"]


class ModelType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    NOISE = enum.auto()  # the model predicts epsilon
    SCORE = enum.auto()  # the model predicts \nabla \log p(x)
    VELOCITY = enum.auto()  # the model predicts v(x)


class PathType(enum.Enum):
    """
    Which type of path to use.
    """

    LINEAR = enum.auto()
    GVP = enum.auto()
    VP = enum.auto()


class WeightType(enum.Enum):
    """
    Which type of weighting to use.
    """

    NONE = enum.auto()
    VELOCITY = enum.auto()
    LIKELIHOOD = enum.auto()


class SNRType(enum.Enum):
    UNIFORM = enum.auto()
    LOGNORM = enum.auto()


def get_lin_function(x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15) -> Callable[[float], float]:
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


def time_shift(mu: float, sigma: float, t: ms.Tensor):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


class Transport:
    def __init__(
        self,
        *,
        model_type,
        path_type,
        loss_type,
        train_eps,
        sample_eps,
        snr_type,
        training_timesteps=1000,
        reverse_time_schedule=False,
        shift=1.0,
        video_shift=None,
        reverse=False,
    ):
        path_options = {
            PathType.LINEAR: path.ICPlan,
            PathType.GVP: path.GVPCPlan,
            PathType.VP: path.VPCPlan,
        }

        self.loss_type = loss_type
        self.model_type = model_type
        self.path_sampler = path_options[path_type](reverse=reverse)
        self.train_eps = train_eps
        self.sample_eps = sample_eps

        self.snr_type = snr_type
        # timestep shift: http://arxiv.org/abs/2403.03206
        self.shift = shift  # flow matching shift factor, =sqrt(m/n)
        if video_shift is None:
            video_shift = shift  # if video shift is not given, set it to be the same as flow shift
        self.video_shift = video_shift
        self.reverse = reverse

        self.training_timesteps = training_timesteps
        self.reverse_time_schedule = reverse_time_schedule

    def prior_logp(self, z):
        """
        Standard multivariate normal prior
        Assume z is batched
        """
        shape = ms.Tensor(z.shape)
        N = ops.prod(shape[1:])
        _fn = lambda x: -N / 2.0 * np.log(2 * np.pi) - ops.sum(x**2) / 2.0
        return ops.vmap(_fn)(z)

    def check_interval(
        self,
        train_eps,
        sample_eps,
        *,
        diffusion_form="SBDM",
        sde=False,
        reverse=False,
        eval=False,
        last_step_size=0.0,
    ):
        t0 = 0
        t1 = 1
        eps = train_eps if not eval else sample_eps
        if type(self.path_sampler) in [path.VPCPlan]:
            t1 = 1 - eps if (not sde or last_step_size == 0) else 1 - last_step_size

        elif (type(self.path_sampler) in [path.ICPlan, path.GVPCPlan]) and (
            self.model_type != ModelType.VELOCITY or sde
        ):  # avoid numerical issue by taking a first semi-implicit step
            t0 = eps if (diffusion_form == "SBDM" and sde) or self.model_type != ModelType.VELOCITY else 0
            t1 = 1 - eps if (not sde or last_step_size == 0) else 1 - last_step_size

        if reverse:
            t0, t1 = 1 - t0, 1 - t1

        return t0, t1

    def sample(self, x1, n_tokens=None):
        """Sampling x0 & t based on shape of x1 (if needed)
        Args:
          x1 - data point; [batch, *dim]
        """
        if isinstance(x1, (list, tuple)):
            x0 = [ms.Tensor(np.random.randn(*img_start.shape), dtype=ms.float32) for img_start in x1]
        else:
            x0 = ms.Tensor(np.random.randn(*x1.shape), dtype=ms.float32)
        t0, t1 = self.check_interval(self.train_eps, self.sample_eps)

        if self.snr_type == SNRType.UNIFORM:
            t = ms.Tensor(np.random.rand(x1[0].shape[0])) * (t1 - t0) + t0
        elif self.snr_type == SNRType.LOGNORM:
            u = ms.Tensor(np.random.normal(loc=0.0, scale=1.0, size=(x1[0].shape[0])), dtype=ms.float32)
            t = 1 / (1 + ops.exp(-u)) * (t1 - t0) + t0
        else:
            raise ValueError(f"Unknown snr type: {self.snr_type}")

        if self.shift != 1.0:
            if self.reverse:
                # xt = (1 - t) * x1 + t * x0
                t = (self.shift * t) / (1 + (self.shift - 1) * t)
            else:
                # xt = t * x1 + (1 - t) * x0
                t = t / (self.shift - (self.shift - 1) * t)

        t = t.to(x1[0].dtype)
        return t, x0, x1

    def get_model_t(self, t):
        if self.reverse_time_schedule:
            return (1 - t) * self.training_timesteps
        else:
            return t * self.training_timesteps

    def training_losses(
        self,
        model,
        x1,
        model_kwargs=None,
        timestep=None,
        n_tokens=None,
        data_type="video",
        i2v_mode=False,
        cond_latents=None,
        args=None,
    ):
        """Loss for training the score model
        Args:
            model: backbone model; could be score, noise, or velocity
            x1: datapoint
            model_kwargs: additional arguments for the model
            timestep: the timestep at which to evaluate loss.
            n_tokens: number of tokens for shift
        """
        if data_type == "image":
            self.shift = self.image_shift
        elif data_type == "video":
            self.shift = self.video_shift
        if model_kwargs is None:
            model_kwargs = {}

        t, x0, x1 = self.sample(x1, n_tokens)
        if timestep is not None:
            t = ops.ones_like(t) * timestep
        t, xt, ut = self.path_sampler.plan(t, x0, x1)
        input_t = self.get_model_t(t)

        if i2v_mode:
            if cond_latents is not None:
                x1_concat = ops.repeat_elements(cond_latents, x1.shape[2], axis=2)
                x1_concat = ops.stack(
                    [
                        x1_concat[:, :, i, :, :] if i == 0 else ops.zeros_like(x1_concat[:, :, i, :, :])
                        for i in range(x1_concat.shape[2])
                    ],
                    axis=2,
                )
            else:
                x1_concat = x1.copy()
                x1_concat = ops.stack(
                    [
                        x1_concat[:, :, i, :, :] if i == 0 else ops.zeros_like(x1_concat[:, :, i, :, :])
                        for i in range(x1_concat.shape[2])
                    ],
                    axis=2,
                )

            mask_concat = ops.ones((x1.shape[0], 1, x1.shape[2], x1.shape[3], x1.shape[4]), ms.float32)
            mask_concat = ops.stack(
                [
                    mask_concat[:, :, i, :, :] if i == 0 else ops.zeros_like(mask_concat[:, :, i, :, :])
                    for i in range(mask_concat.shape[2])
                ],
                axis=2,
            )

            xt = ops.concat([xt, x1_concat, mask_concat], axis=1)

        if args is not None and args.embedded_cfg_scale is not None:
            guidance_expand = ops.ones(x1.shape[0], ms.float32) * args.embedded_cfg_scale * 1000.0
            guidance_expand = guidance_expand.to(PRECISION_TO_TYPE[args.precision])
        else:
            guidance_expand = None

        model_kwargs["guidance"] = guidance_expand

        model_output = model(xt, input_t, **model_kwargs)["x"]

        if not i2v_mode:
            assert model_output.shape == xt.shape, (
                f"Output shape from model does not match input shape: " f"{model_output.shape} != {xt.shape}"
            )

        terms = {}
        if self.model_type == ModelType.VELOCITY:
            terms["loss"] = mean_flat(((model_output - ut) ** 2))
        else:
            _, drift_var = self.path_sampler.compute_drift(xt, t)
            sigma_t, _ = self.path_sampler.compute_sigma_t(path.expand_t_like_x(t, xt))
            if self.loss_type in [WeightType.VELOCITY]:
                weight = (drift_var / sigma_t) ** 2
            elif self.loss_type in [WeightType.LIKELIHOOD]:
                weight = drift_var / (sigma_t**2)
            elif self.loss_type in [WeightType.NONE]:
                weight = 1
            else:
                raise NotImplementedError()

            if self.model_type == ModelType.NOISE:
                terms["loss"] = mean_flat(weight * ((model_output - x0) ** 2))
            else:
                terms["loss"] = mean_flat(weight * ((model_output * sigma_t + x0) ** 2))

        return model_output, terms

    def get_drift(self):
        """member function for obtaining the drift of the probability flow ODE"""

        def score_ode(x, t, model, **model_kwargs):
            drift_mean, drift_var = self.path_sampler.compute_drift(x, t)
            model_output = model(x, t, **model_kwargs)
            return -drift_mean + drift_var * model_output  # by change of variable

        def noise_ode(x, t, model, **model_kwargs):
            drift_mean, drift_var = self.path_sampler.compute_drift(x, t)
            sigma_t, _ = self.path_sampler.compute_sigma_t(path.expand_t_like_x(t, x))
            model_output = model(x, t, **model_kwargs)
            score = model_output / -sigma_t
            return -drift_mean + drift_var * score

        def velocity_ode(x, t, model, **model_kwargs):
            model_output = model(x, t, **model_kwargs)
            return model_output

        if self.model_type == ModelType.NOISE:
            drift_fn = noise_ode
        elif self.model_type == ModelType.SCORE:
            drift_fn = score_ode
        else:
            drift_fn = velocity_ode

        def body_fn(x, t, model, **model_kwargs):
            model_output = drift_fn(x, t, model, **model_kwargs)
            assert model_output.shape == x.shape, "Output shape from ODE solver must match input shape"
            return model_output

        return body_fn

    def get_score(
        self,
    ):
        """member function for obtaining score of
        x_t = alpha_t * x + sigma_t * eps"""
        if self.model_type == ModelType.NOISE:
            score_fn = (
                lambda x, t, model, **kwargs: model(x, t, **kwargs)
                / -self.path_sampler.compute_sigma_t(path.expand_t_like_x(t, x))[0]
            )
        elif self.model_type == ModelType.SCORE:
            score_fn = lambda x, t, model, **kwagrs: model(x, t, **kwagrs)
        elif self.model_type == ModelType.VELOCITY:
            score_fn = lambda x, t, model, **kwargs: self.path_sampler.get_score_from_velocity(
                model(x, t, **kwargs), x, t
            )
        else:
            raise NotImplementedError()

        return score_fn


class Sampler:
    """Sampler class for the transport model"""

    def __init__(
        self,
        transport,
    ):
        """Constructor for a general sampler; supporting different sampling methods
        Args:
        - transport: an tranport object specify model prediction & interpolant type
        """

        self.transport = transport
        self.drift = self.transport.get_drift()
        self.score = self.transport.get_score()

    def __get_sde_diffusion_and_drift(
        self,
        *,
        diffusion_form="SBDM",
        diffusion_norm=1.0,
    ):
        def diffusion_fn(x, t):
            diffusion = self.transport.path_sampler.compute_diffusion(x, t, form=diffusion_form, norm=diffusion_norm)
            return diffusion

        sde_drift = lambda x, t, model, **kwargs: self.drift(x, t, model, **kwargs) + diffusion_fn(x, t) * self.score(
            x, t, model, **kwargs
        )

        sde_diffusion = diffusion_fn

        return sde_drift, sde_diffusion

    def __get_last_step(
        self,
        sde_drift,
        *,
        last_step,
        last_step_size,
    ):
        """Get the last step function of the SDE solver"""

        if last_step is None:
            last_step_fn = lambda x, t, model, **model_kwargs: x
        elif last_step == "Mean":
            last_step_fn = (
                lambda x, t, model, **model_kwargs: x + sde_drift(x, t, model, **model_kwargs) * last_step_size
            )
        elif last_step == "Tweedie":
            alpha = self.transport.path_sampler.compute_alpha_t  # simple aliasing; the original name was too long
            sigma = self.transport.path_sampler.compute_sigma_t
            last_step_fn = lambda x, t, model, **model_kwargs: x / alpha(t)[0][0] + (sigma(t)[0][0] ** 2) / alpha(t)[0][
                0
            ] * self.score(x, t, model, **model_kwargs)
        elif last_step == "Euler":
            last_step_fn = (
                lambda x, t, model, **model_kwargs: x + self.drift(x, t, model, **model_kwargs) * last_step_size
            )
        else:
            raise NotImplementedError()

        return last_step_fn

    def sample_sde(
        self,
        *,
        sampling_method="Euler",
        diffusion_form="SBDM",
        diffusion_norm=1.0,
        last_step="Mean",
        last_step_size=0.04,
        num_steps=250,
    ):
        """returns a sampling function with given SDE settings
        Args:
        - sampling_method: type of sampler used in solving the SDE; default to be Euler-Maruyama
        - diffusion_form: function form of diffusion coefficient; default to be matching SBDM
        - diffusion_norm: function magnitude of diffusion coefficient; default to 1
        - last_step: type of the last step; default to identity
        - last_step_size: size of the last step; default to match the stride of 250 steps over [0,1]
        - num_steps: total integration step of SDE
        """

        if last_step is None:
            last_step_size = 0.0

        sde_drift, sde_diffusion = self.__get_sde_diffusion_and_drift(
            diffusion_form=diffusion_form,
            diffusion_norm=diffusion_norm,
        )

        t0, t1 = self.transport.check_interval(
            self.transport.train_eps,
            self.transport.sample_eps,
            diffusion_form=diffusion_form,
            sde=True,
            eval=True,
            reverse=False,
            last_step_size=last_step_size,
        )

        _sde = sde(
            sde_drift,
            sde_diffusion,
            t0=t0,
            t1=t1,
            num_steps=num_steps,
            sampler_type=sampling_method,
        )

        last_step_fn = self.__get_last_step(sde_drift, last_step=last_step, last_step_size=last_step_size)

        def _sample(init, model, **model_kwargs):
            xs = _sde.sample(init, model, **model_kwargs)
            ts = ops.ones(init.shape[0], ms.float32) * t1
            x = last_step_fn(xs[-1], ts, model, **model_kwargs)
            xs.append(x)

            assert len(xs) == num_steps, "Samples does not match the number of steps"

            return xs

        return _sample

    def sample_ode(
        self,
        *,
        sampling_method="dopri5",
        num_steps=50,
        atol=1e-6,
        rtol=1e-3,
        reverse=False,
        time_shifting_factor=None,
    ):
        """returns a sampling function with given ODE settings
        Args:
        - sampling_method: type of sampler used in solving the ODE; default to be Dopri5
        - num_steps:
            - fixed solver (Euler, Heun): the actual number of integration steps performed
            - adaptive solver (Dopri5): the number of datapoints saved during integration; produced by interpolation
        - atol: absolute error tolerance for the solver
        - rtol: relative error tolerance for the solver
        - reverse: whether solving the ODE in reverse (data to noise); default to False
        """
        if reverse:
            drift = lambda x, t, model, **kwargs: self.drift(x, ops.ones_like(t) * (1 - t), model, **kwargs)
        else:
            drift = self.drift

        t0, t1 = self.transport.check_interval(
            self.transport.train_eps,
            self.transport.sample_eps,
            sde=False,
            eval=True,
            reverse=reverse,
            last_step_size=0.0,
        )

        _ode = ode(
            drift=drift,
            t0=t0,
            t1=t1,
            sampler_type=sampling_method,
            num_steps=num_steps,
            atol=atol,
            rtol=rtol,
            time_shifting_factor=time_shifting_factor,
        )
        self.ode = _ode
        return _ode.sample

    def sample_ode_likelihood(
        self,
        *,
        sampling_method="dopri5",
        num_steps=50,
        atol=1e-6,
        rtol=1e-3,
    ):
        """returns a sampling function for calculating likelihood with given ODE settings
        Args:
        - sampling_method: type of sampler used in solving the ODE; default to be Dopri5
        - num_steps:
            - fixed solver (Euler, Heun): the actual number of integration steps performed
            - adaptive solver (Dopri5): the number of datapoints saved during integration; produced by interpolation
        - atol: absolute error tolerance for the solver
        - rtol: relative error tolerance for the solver
        """

        def _likelihood_drift(x, t, model, **model_kwargs):
            x, _ = x
            eps = ms.Tensor(np.random.randint(2, size=x.shape).astype(np.float32)) * 2 - 1
            t = ops.ones_like(t) * (1 - t)
            x.requires_grad = True
            grad = ops.grad(lambda x: ops.sum(self.drift(x, t, model, **model_kwargs) * eps))(x)
            logp_grad = ops.sum(grad * eps, axis=tuple(range(1, len(x.shape))))
            drift = self.drift(x, t, model, **model_kwargs)
            return (-drift, logp_grad)

        t0, t1 = self.transport.check_interval(
            self.transport.train_eps,
            self.transport.sample_eps,
            sde=False,
            eval=True,
            reverse=False,
            last_step_size=0.0,
        )

        _ode = ode(
            drift=_likelihood_drift,
            t0=t0,
            t1=t1,
            sampler_type=sampling_method,
            num_steps=num_steps,
            atol=atol,
            rtol=rtol,
        )

        def _sample_fn(x, model, **model_kwargs):
            init_logp = ops.zeros(x.shape[0], ms.float32)
            input = (x, init_logp)
            drift, delta_logp = _ode.sample(input, model, **model_kwargs)
            drift, delta_logp = drift[-1], delta_logp[-1]
            prior_logp = self.transport.prior_logp(drift)
            logp = prior_logp - delta_logp
            return logp, drift

        return _sample_fn
