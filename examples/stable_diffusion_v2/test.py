import time

import mindspore as ms
from mindspore import nn
import numpy as np
import time

ms.set_context(device_id=2, mode=ms.GRAPH_MODE)


def get_word_inds(text: str, word_place: int):
    split_text = text.split(" ")
    if type(word_place) is str:
        word_place = [i for i, word in enumerate(split_text) if word_place == word]
    elif type(word_place) is int:
        word_place = [word_place]
    out = []
    if len(word_place) > 0:
        # words_encode = [tokenizer.decode([item]).strip("#") for item in tokenizer.encode(text)][1:-1]
        words_encode = split_text
        cur_len, ptr = 0, 0

        for i in range(len(words_encode)):
            cur_len += len(words_encode[i])
            if ptr in word_place:
                out.append(i + 1)
            if cur_len >= len(split_text[ptr]):
                ptr += 1
                cur_len = 0
    return np.array(out)


def get_replacement_mapper_(x: str, y: str, max_len=77):
    words_x = x.split(' ')
    words_y = y.split(' ')
    if len(words_x) != len(words_y):
        raise ValueError(f"attention replacement edit can only be applied on prompts with the same length"
                         f" but prompt A has {len(words_x)} words and prompt B has {len(words_y)} words.")
    inds_replace = [i for i in range(len(words_y)) if words_y[i] != words_x[i]]
    inds_source = [get_word_inds(x, i) for i in inds_replace]
    inds_target = [get_word_inds(y, i) for i in inds_replace]
    mapper = np.zeros((max_len, max_len))
    i = j = 0
    cur_inds = 0
    while i < max_len and j < max_len:
        if cur_inds < len(inds_source) and inds_source[cur_inds][0] == i:
            inds_source_, inds_target_ = inds_source[cur_inds], inds_target[cur_inds]
            if len(inds_source_) == len(inds_target_):
                mapper[inds_source_, inds_target_] = 1
            else:
                ratio = 1 / len(inds_target_)
                for i_t in inds_target_:
                    mapper[inds_source_, i_t] = ratio
            cur_inds += 1
            i += len(inds_source_)
            j += len(inds_target_)
        elif cur_inds < len(inds_source):
            mapper[i, j] = 1
            i += 1
            j += 1
        else:
            mapper[j, j] = 1
            i += 1
            j += 1

    return mapper


get_replacement_mapper_(
    "a silver jeep driving down a curvy road in the countryside",
    "a Porsche car driving down a curvy road in the countryside"
)

exit()

start = time.time()


def c(a, b):
    return a * b[0]


class NetChild(nn.Cell):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 4, 3, has_bias=True, weight_init='normal')

    def construct(self, x, d):
        out = self.conv(x)
        out = c(out, b=d['num_self_replace'])
        # out = out.reshape((-1,) + list_)
        return out


class Net(nn.Cell):
    def __init__(self, d=None):
        super().__init__()
        self.blocks = nn.CellList([NetChild()])
        self.dict_ = d

    def construct(self, x):
        for c in self.blocks:
            x = c(x, d=self.dict_)
        return x


controller = {
    "num_self_replace": (10, 20)
}

k = {
    "d": controller
}
net = Net(**k)
input = ms.Tensor(np.ones([1, 2, 4, 1]), ms.float32)

out = net(input)

print(F"total time: {(time.time() - start):.2f}")
