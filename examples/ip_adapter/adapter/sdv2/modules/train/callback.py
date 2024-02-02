from ldm.modules.train.callback import EvalSaveCallback


class IPAdapterEvalSaveCallback(EvalSaveCallback):
    def __init__(
        self,
        network,
        save_ip_only=True,
        **kwargs,
    ):
        super().__init__(network, **kwargs)
        if save_ip_only:
            ckpt_ip = []
            ip_names = ["to_k_ip", "to_v_ip", "image_proj"]
            for name, param in network.parameters_and_names():
                if any([x in name for x in ip_names]):
                    ckpt_ip.append({"name": name, "data": param})
            self.net_to_save = ckpt_ip
        else:
            self.net_to_save = network
