# Developer Guide: How to add a new scheduler

The scheduler class should inherit from mindspore.nn.Cell, and provide two key methods:
    1. `add_noise`: adding noise to previous (latent) sample for diffusion forward
    2. `construct`: predict the previous sample based on model output (typically, predicted latent noise) and input sample, i.e. diffusion backward
    3. `set_timesteps`: set the discrete timesteps used for the diffusion chain (to be run before inference).


Example Template:

```python
class NewScheduler(nn.Cell):
    def add_noise(self, original_samples: ms.Tensor, noise: ms.Tensor, timestep: int) -> ms.Tensor:
	    ...
        return noisy_samples

    def construct(self, model_output: ms.Tensor, timestep: ms.Tensor, sample: ms.Tensor) -> ms.Tensor:
	    ...
	    return  prev_sample

    def set_timesteps(self, num_inference_steps: int) -> List[int]:

```

See [DDIMSchedulera](./ddim_scheduler.py) for detailed implementation reference.
