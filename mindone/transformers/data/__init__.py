# This code is adapted from https://github.com/huggingface/transformers
# with modifications to run transformers on mindspore.

from .processors import (
    DataProcessor,
    InputExample,
    InputFeatures,
    SingleSentenceClassificationProcessor,
    SquadExample,
    SquadFeatures,
    SquadV1Processor,
    SquadV2Processor,
    squad_convert_examples_to_features,
)
