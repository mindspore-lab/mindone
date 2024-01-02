# Developer Guide: How to add a new scheduler

The scheduler class should inherit from mindspore.nn.Cell, and provide two key methods:
    1. `add_noise`: adding noise to previous (latent) sample for diffusion forward
    2. `construct`: predict the previous sample based on model output (typically, predicted latent noise) and input sample, i.e. diffusion backward


Example Template:

```python
class NewScheduler(nn.Cell):
    def add_noise(self, original_samples, noise, timesteps):
	...
        return noisy_samples

    def construct(self, model_output, timestep, sample):
	...
	return  prev_sample

```

See [DDIMSchedulera](./ddim_scheduler.py) for detailed implementation reference.
