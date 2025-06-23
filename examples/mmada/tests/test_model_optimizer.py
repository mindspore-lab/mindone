import os
import sys
from typing import NamedTuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import logging

from models import MAGVITv2, MMadaConfig, MMadaModelLM
from transformers import AutoConfig, AutoTokenizer

from mindspore.experimental import optim

logger = logging.getLogger(__name__)


# Placeholder for config object used in testing
class MockConfig:
    """Mock configuration class for testing."""

    def __init__(self):
        self.model = MockModelConfig()
        self.optimizer = MockOptimizerConfig()
        self.lr_scheduler = MockLRSchedulerConfig()


class MockModelConfig:
    """Mock model configuration class for testing."""

    def __init__(self):
        self.vq_model = MockVQModelConfig()
        self.mmada = MockMMadaConfig()
        self.gradient_checkpointing = True


class MockVQModelConfig:
    """Mock VQ model configuration class for testing."""

    def __init__(self):
        self.type = "magvitv2"
        self.vq_model_name = "showlab/magvitv2"


class MockMMadaConfig(NamedTuple):
    """Mock MMada configuration class for testing."""

    pretrained_model_path: str = "GSAI-ML/LLaDA-8B-Instruct"
    w_clip_vit: bool = False
    new_vocab_size: int = 134656
    llm_vocab_size: int = 126464
    codebook_size: int = 8192
    num_vq_tokens: int = 256
    num_new_special_tokens: int = 0
    tie_word_embeddings: bool = False


class MockOptimizerConfig:
    """Mock optimizer configuration class for testing."""

    def __init__(self):
        self.name = "adamw"
        self.params = MockOptimizerParamsConfig()


class MockOptimizerParamsConfig:
    """Mock optimizer parameter configuration class for testing."""

    def __init__(self):
        self.learning_rate = 1e-4
        self.scale_lr = False
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.weight_decay = 0.01
        self.epsilon = 1e-8


class MockLRSchedulerConfig:
    """Mock learning rate scheduler configuration class for testing."""

    def __init__(self):
        self.scheduler = "cosine"
        self.params = MockLRSchedulerParamsConfig()


class MockLRSchedulerParamsConfig:
    """Mock learning rate scheduler parameter configuration class for testing."""

    def __init__(self):
        self.learning_rate = 1e-4
        self.warmup_steps = 5000


def get_vq_model_class(model_type):
    if model_type == "magvitv2":
        return MAGVITv2
    else:
        raise ValueError(f"model_type {model_type} not supported.")


def test_model_and_optimizer_initialization():
    config = MockConfig()

    try:
        tokenizer = AutoTokenizer.from_pretrained(config.model.mmada.pretrained_model_path, padding_side="left")
        vq_model = get_vq_model_class(config.model.vq_model.type)
        vq_model = vq_model.from_pretrained(config.model.vq_model.vq_model_name, use_safetensors=True)
        vq_model.set_train(False)
        for p in vq_model.get_parameters():
            p.requires_grad = False

        base_config = AutoConfig.from_pretrained(config.model.mmada.pretrained_model_path).to_dict()
        mmada_config_dict = {k: v for k, v in config.model.mmada._asdict().items()}
        merged_config = {**base_config, **mmada_config_dict}
        mmada_config = MMadaConfig(**merged_config)
        mmada_config.num_hidden_layers = 1  # set to 1 layer for verifcation
        model = MMadaModelLM(config=mmada_config)
        model.resize_token_embeddings(mmada_config.new_vocab_size)
        model.config.embedding_size = model.config.vocab_size
        model.set_train(True)
        logger.info("Model loaded and initialized successfully")
    except Exception as e:
        logger.error(f"Model loading and initialization failed: {e}")
        raise

    try:
        # no decay on bias and layernorm and embedding
        no_decay = ["bias", "layer_norm.weight", "mlm_ln.weight", "embeddings.weight", "embeddings.embedding_table"]
        trainable_params = model.trainable_params()
        optimizer_grouped_parameters = [
            {
                "params": [p for p in trainable_params if not any(nd in p.name for nd in no_decay)],
                "weight_decay": config.optimizer.params.weight_decay,
            },
            {
                "params": [p for p in trainable_params if any(nd in p.name for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        # filter empty params
        optimizer_grouped_parameters = [d for d in optimizer_grouped_parameters if len(d["params"])]

        optimizer_type = config.optimizer.name
        if optimizer_type == "adamw":
            optimizer = optim.AdamW(
                optimizer_grouped_parameters,
                lr=config.optimizer.params.learning_rate,
                betas=(config.optimizer.params.beta1, config.optimizer.params.beta2),
                weight_decay=config.optimizer.params.weight_decay,
                eps=config.optimizer.params.epsilon,
            )
        else:
            raise ValueError(f"Optimizer {optimizer_type} not supported")

        logger.info("Optimizer initialized successfully")
    except Exception as e:
        logger.error(f"Optimizer initialization failed: {e}")
        raise


if __name__ == "__main__":
    test_model_and_optimizer_initialization()
