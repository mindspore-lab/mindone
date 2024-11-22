import os

from .simple_tokenizer import SimpleTokenizer as _Tokenizer
from .viclip import ViCLIP


def get_viclip(
    size="l", pretrain=os.path.join(os.path.dirname(os.path.abspath(__file__)), "ViClip-InternVid-10M-FLT.pth")
):
    tokenizer = _Tokenizer()
    vclip = ViCLIP(tokenizer=tokenizer, size=size, pretrain=pretrain)
    m = {"viclip": vclip, "tokenizer": tokenizer}

    return m
