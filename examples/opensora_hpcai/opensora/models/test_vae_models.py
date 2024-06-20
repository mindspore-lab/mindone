from vae.vae import OpenSoraVAE_V1_2


def get_pnames():
    model = OpenSoraVAE_V1_2(ckpt_path=None)
    for param in model.get_parameters():
        # print(f"{param.name}#{tuple(param.shape)}")
        print("{}#{}".format(param.name, tuple(param.shape)))


if __name__ == "__main__":
    get_pnames()
