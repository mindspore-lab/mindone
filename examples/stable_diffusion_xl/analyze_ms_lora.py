import mindspore as ms
from fire import Fire

def check(ms_path):
    sd = ms.load_checkpoint(ms_path)
    for pname in sd:
        print("{}#{}#{}".format(pname, sd[pname].shape, sd[pname].dtype))


if __name__ == '__main__':
    Fire(check)
