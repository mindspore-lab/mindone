import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import logging

from models import MAGVITv2, MMadaConfig, MMadaModelLM
from omegaconf import OmegaConf
from transformers import AutoConfig, AutoTokenizer

from mindone.trainers.utils import create_optimizer

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


class MockMMadaConfig:
    """Mock MMada configuration class for testing."""

    def __init__(self):
        self.pretrained_model_path = "GSAI-ML/LLaDA-8B-Instruct"
        self.w_clip_vit = False
        self.new_vocab_size = 134656
        self.llm_vocab_size = 126464
        self.codebook_size = 8192
        self.num_vq_tokens = 256
        self.num_new_special_tokens = 0
        self.tie_word_embeddings = False


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
        vq_model = vq_model.from_pretrained(config.model.vq_model.vq_model_name)
        vq_model.eval()
        vq_model.requires_grad = False

        base_config = AutoConfig.from_pretrained(config.model.mmada.pretrained_model_path).to_dict()
        mmada_config_dict = {k: v for k, v in config.model.mmada.items()}
        merged_config = {**base_config, **mmada_config_dict}
        mmada_config = MMadaConfig(**merged_config)
        model = MMadaModelLM.from_pretrained(
            config.model.mmada.pretrained_model_path, mindspore_dtype=None, config=mmada_config
        )
        model.resize_token_embeddings(mmada_config.new_vocab_size)
        model.config.embedding_size = model.config.vocab_size

        logger.info("Model loaded and initialized successfully")
    except Exception as e:
        logger.error(f"Model loading and initialization failed: {e}")
        raise

    try:
        # no decay on bias and layernorm and embedding
        no_decay = ["bias", "layer_norm.weight", "mlm_ln.weight", "embeddings.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in model.name_cells().items() if p.requires_grad and not any(nd in n for nd in no_decay)
                ],
                "weight_decay": config.optimizer.params.weight_decay,
            },
            {
                "params": [
                    p for n, p in model.name_cells().items() if p.requires_grad and any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer_type = config.optimizer.name
        if optimizer_type == "adamw":
            optimizer = create_optimizer(
                optimizer_grouped_parameters,
                name="adamw",
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
