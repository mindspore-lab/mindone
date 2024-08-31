import datetime
import types
import numpy as np
from collections import OrderedDict
from tqdm import tqdm
import os
import re
import math
import random
import json
import time
import logging
from omegaconf import OmegaConf
from copy import deepcopy
import argparse
import data
from mindnlp.transformers import LlamaTokenizer,LlamaConfig,LlamaForCausalLM
from mindnlp.peft import LoraConfig,TaskType,get_peft_model
from mindone.diffusers.utils import export_to_video
import scipy

logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
