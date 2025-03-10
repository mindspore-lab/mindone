# -*- coding: utf-8 -*-


def deepspeed_config_from_args(args, global_batch_size):
    deepspeed_config = {
        "optimizer": {
            "type": "AdamW",
            "params": {
                "learning_rate": args.lr,
                "beta1": 0.9,
                "beta2": 0.999,
                "eps": 1e-08,
                "weight_decay": args.weight_decay,
            },
        },
        "gradient_clipping": 1.0,
        "loss_scaler": {
            "scale_value": 2**15,
            "scale_window": 500,
            "scale_factor": 2,
        },
    }

    return deepspeed_config
