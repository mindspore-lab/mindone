# This code is adapted from https://github.com/huggingface/transformers
# with modifications to run transformers on mindspore.

from .squad import SquadExample, SquadFeatures, SquadV1Processor, SquadV2Processor, squad_convert_examples_to_features
from .utils import DataProcessor, InputExample, InputFeatures, SingleSentenceClassificationProcessor
