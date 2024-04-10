import numpy as np
import torch

import mindspore as ms


def compare_gn(npy_fp=None):
    bs, cin, T, H, W = 1, 3, 8, 256, 256  # noqa: F841
    cout = 128
    # x = np.random.normal(size=(bs, cout, T, H, W ))
    x = np.random.uniform(size=(bs, cout, H, W), low=-1, high=1)
    # x = np.random.normal(size=(bs, cout, H, W ))

    npt = torch.nn.GroupNorm(32, cout, eps=1e-6, affine=True)
    npt.eval()
    outpt = npt(torch.Tensor(x))
    # print(outpt.sum().detach().numpy())
    outpt = outpt.detach().numpy()

    def _convert_ckpt(net=npt, name="gn"):
        torch.save(
            {
                "model_state_dict": net.state_dict(),
            },
            f"tests/{name}.pth",
        )

        target_data = []
        for k in net.state_dict():
            ms_name = k.replace("weight", "gamma").replace("bias", "beta")
            target_data.append({"name": ms_name, "data": ms.Tensor(net.state_dict()[k].detach().numpy())})
        save_fn = f"tests/{name}.ckpt"
        ms.save_checkpoint(target_data, save_fn)
        return save_fn

    ms_ckpt = _convert_ckpt(npt, "gn")
    # ms
    nms = ms.nn.GroupNorm(32, cout, eps=1e-6, affine=True)
    nms.set_train(False)
    sd = ms.load_checkpoint(ms_ckpt)
    ms.load_param_into_net(nms, sd)

    outms = nms(ms.Tensor(x, dtype=ms.float32))
    # print(outms.sum().asnumpy())
    outms = outms.asnumpy()

    abs_diff = np.fabs(outms - outpt)
    print("mae: ", abs_diff.mean())
    print("max ae: ", abs_diff.max())


compare_gn()
