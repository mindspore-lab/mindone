import mindspore as ms
import mindspore.numpy as np
from mindspore import Tensor, ops

from .diffeq import odeint


class sde:
    """SDE solver class"""

    def __init__(
        self,
        drift,
        diffusion,
        *,
        t0,
        t1,
        num_steps,
        sampler_type,
    ):
        assert t0 < t1, "SDE sampler has to be in forward time"

        self.num_timesteps = num_steps
        self.t = np.linspace(t0, t1, num_steps)
        self.dt = self.t[1] - self.t[0]
        self.drift = drift
        self.diffusion = diffusion
        self.sampler_type = sampler_type

    def __Euler_Maruyama_step(self, x, mean_x, t, model, **model_kwargs):
        w_cur = Tensor(np.random.randn(*x.shape), x.dtype)
        t = ops.ones(x.shape[0], ms.float32) * t
        dw = w_cur * np.sqrt(self.dt)
        drift = self.drift(x, t, model, **model_kwargs)
        diffusion = self.diffusion(x, t)
        mean_x = x + drift * self.dt
        x = mean_x + np.sqrt(2 * diffusion) * dw
        return x, mean_x

    def __Heun_step(self, x, _, t, model, **model_kwargs):
        w_cur = Tensor(np.random.randn(*x.shape), x.dtype)
        dw = w_cur * np.sqrt(self.dt)
        t_cur = ops.ones(x.shape[0], ms.float32) * t
        diffusion = self.diffusion(x, t_cur)
        xhat = x + np.sqrt(2 * diffusion) * dw
        K1 = self.drift(xhat, t_cur, model, **model_kwargs)
        xp = xhat + self.dt * K1
        K2 = self.drift(xp, t_cur + self.dt, model, **model_kwargs)
        return (
            xhat + 0.5 * self.dt * (K1 + K2),
            xhat,
        )  # at last time point we do not perform the heun step

    def __forward_fn(self):
        """TODO: generalize here by adding all private functions ending with steps to it"""
        sampler_dict = {
            "Euler": self.__Euler_Maruyama_step,
            "Heun": self.__Heun_step,
        }

        try:
            sampler = sampler_dict[self.sampler_type]
        except KeyError:
            raise NotImplementedError("Smapler type not implemented.")

        return sampler

    def sample(self, init, model, **model_kwargs):
        """forward loop of sde"""
        x = init
        mean_x = init
        samples = []
        sampler = self.__forward_fn()
        for ti in self.t[:-1]:
            x, mean_x = sampler(x, mean_x, ti, model, **model_kwargs)
            samples.append(x)

        return samples


class ode:
    """ODE solver class"""

    def __init__(
        self,
        drift,
        *,
        t0,
        t1,
        sampler_type,
        num_steps,
        atol,
        rtol,
        time_shifting_factor=None,
    ):
        assert t0 < t1, "ODE sampler has to be in forward time"

        self.drift = drift
        self.t = np.linspace(t0, t1, num_steps)
        if time_shifting_factor:
            self.t = self.t / (self.t + time_shifting_factor - time_shifting_factor * self.t)
        self.atol = atol
        self.rtol = rtol
        self.sampler_type = sampler_type

    def sample(self, x, model, **model_kwargs):
        def _fn(t, x):
            t = (
                ops.ones(x[0].shape[0], ms.float32) * t
                if isinstance(x, tuple)
                else ops.ones(x.shape[0], ms.float32) * t
            )
            model_output = self.drift(x, t, model, **model_kwargs)
            return model_output

        t = self.t
        atol = [self.atol] * len(x) if isinstance(x, tuple) else [self.atol]
        rtol = [self.rtol] * len(x) if isinstance(x, tuple) else [self.rtol]
        samples = odeint(_fn, x, t, method=self.sampler_type, atol=atol, rtol=rtol)
        return samples

    def sample_with_step_fn(self, x, step_fn):
        t = self.t
        atol = [self.atol] * len(x) if isinstance(x, tuple) else [self.atol]
        rtol = [self.rtol] * len(x) if isinstance(x, tuple) else [self.rtol]
        samples = odeint(step_fn, x, t, method=self.sampler_type, atol=atol, rtol=rtol)
        return samples
