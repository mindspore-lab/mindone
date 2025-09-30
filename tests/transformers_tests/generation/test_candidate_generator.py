import gc
import unittest
import weakref
from unittest.mock import MagicMock

from transformers import AutoTokenizer, GenerationConfig

import mindspore as ms
from mindspore import mint

from mindone.transformers import AutoConfig, AutoModelForCausalLM
from mindone.transformers.generation.candidate_generator import (
    AssistantToTargetTranslator,
    AssistantVocabTranslatorCache,
    UniversalSpeculativeDecodingGenerator,
)
from mindone.transformers.testing_utils import require_mindspore


@require_mindspore
class TestAssistantToTargetTranslator(unittest.TestCase):
    def setUp(self):
        # Create mock tokenizers with predefined vocabularies
        self.target_tokenizer = MagicMock()
        self.assistant_tokenizer = MagicMock()

        # Define mock vocabularies for the tokenizers
        self.target_vocab = {"hello": 0, "world": 1, "foo": 2, "bar": 3}
        self.assistant_vocab = {"hello": 0, "world": 1, "foo": 2, "baz": 4}

        self.target_tokenizer.get_vocab.return_value = self.target_vocab
        self.assistant_tokenizer.get_vocab.return_value = self.assistant_vocab
        self.target_vocab_size = 6

        # Instantiate the class under test
        self.translator = AssistantToTargetTranslator(
            target_tokenizer=self.target_tokenizer,
            assistant_tokenizer=self.assistant_tokenizer,
            target_vocab_size=self.target_vocab_size,
        )

    def test_get_assistant_to_target_input_ids(self):
        """Test the mapping from assistant tokens to target tokens."""
        expected_mapping = [0, 1, 2, self.translator.SUPPRESS_TOKEN_ID, self.translator.SUPPRESS_TOKEN_ID]
        actual_mapping = self.translator._assistant_to_target_input_ids.tolist()
        self.assertEqual(actual_mapping, expected_mapping)

    def test_get_suppress_input_ids(self):
        """Test the suppression of assistant input IDs not present in the target vocabulary."""
        expected_suppress_ids = [3, 4]
        actual_suppress_ids = self.translator._get_suppress_input_ids().tolist()
        self.assertEqual(actual_suppress_ids, expected_suppress_ids)


class MockTokenizer:
    """A simple mock tokenizer class that supports weak references."""

    def __init__(self, vocab=None):
        self._vocab = vocab or {}

    def get_vocab(self):
        return self._vocab

    def __call__(self, text, add_special_tokens=True):
        # Mock implementation of the __call__ method
        tokens = text.split()
        input_ids = [self._vocab.get(token, 0) for token in tokens]
        return {"input_ids": input_ids}


@require_mindspore
class TestAssistantVocabTranslatorCache(unittest.TestCase):
    def setUp(self):
        # Clear the cache before each test
        AssistantVocabTranslatorCache._cache.clear()
        # Create mock tokenizers with different vocabularies
        self.target_tokenizer = MockTokenizer({"hello": 0, "world": 1})
        self.assistant_tokenizer = MockTokenizer({"hello": 0, "world": 1, "foo": 2})
        self.other_target_tokenizer = MockTokenizer({"foo": 2, "bar": 3})
        self.other_assistant_tokenizer = MockTokenizer({"baz": 4, "qux": 5})
        self.target_vocab_size = 6

    def test_same_instance_for_same_tokenizers(self):
        """Test that the same translator is returned for the same tokenizers."""
        translator1 = AssistantVocabTranslatorCache.get_translator(
            self.target_tokenizer,
            self.assistant_tokenizer,
            target_vocab_size=self.target_vocab_size,
        )
        translator2 = AssistantVocabTranslatorCache.get_translator(
            self.target_tokenizer,
            self.assistant_tokenizer,
            target_vocab_size=self.target_vocab_size,
        )
        self.assertIs(translator1, translator2, "Translators should be cached and identical")

    def test_different_instances_for_different_tokenizers(self):
        """Test that different tokenizers produce different translators."""
        translator1 = AssistantVocabTranslatorCache.get_translator(
            self.target_tokenizer,
            self.assistant_tokenizer,
            target_vocab_size=self.target_vocab_size,
        )
        translator2 = AssistantVocabTranslatorCache.get_translator(
            self.other_target_tokenizer,
            self.other_assistant_tokenizer,
            target_vocab_size=self.target_vocab_size,
        )
        self.assertIsNot(translator1, translator2, "Translators should differ for different tokenizers")

    def test_cache_with_weakref_key(self):
        """Ensure that the cache uses weak references as keys."""
        initial_cache_size = len(AssistantVocabTranslatorCache._cache)
        target_tokenizer = MockTokenizer({"hello": 0})
        assistant_tokenizer = MockTokenizer({"hello": 0})

        # Store translator in a local variable to avoid it being kept alive
        translator = AssistantVocabTranslatorCache.get_translator(
            target_tokenizer,
            assistant_tokenizer,
            target_vocab_size=self.target_vocab_size,
        )
        self.assertEqual(len(AssistantVocabTranslatorCache._cache), initial_cache_size + 1)

        # Delete all strong references
        del target_tokenizer
        del assistant_tokenizer
        del translator

        # Force garbage collection
        gc.collect()

        # Call cleanup to remove dead entries
        AssistantVocabTranslatorCache.cleanup()

        # The cache size remains increased due to strong references
        self.assertEqual(len(AssistantVocabTranslatorCache._cache), initial_cache_size + 1)

    def test_weakref_cache_cleanup(self):
        """Test that the cache cleans up translators when tokenizers are garbage collected."""

        def create_translator():
            target_tokenizer = MockTokenizer({"hello": 0})
            assistant_tokenizer = MockTokenizer({"hello": 0})
            translator = AssistantVocabTranslatorCache.get_translator(
                target_tokenizer,
                assistant_tokenizer,
                target_vocab_size=self.target_vocab_size,
            )
            # Create weak references before returning
            refs = (weakref.ref(translator), weakref.ref(target_tokenizer), weakref.ref(assistant_tokenizer))
            # Remove strong references inside the function
            del target_tokenizer
            del assistant_tokenizer
            del translator
            return refs

        translator_ref, target_ref, assistant_ref = create_translator()

        # Force garbage collection
        gc.collect()

        # Call cleanup to remove dead entries
        AssistantVocabTranslatorCache.cleanup()

        # The tokenizers and translator are not garbage collected due to strong references
        self.assertIsNotNone(target_ref(), "Target tokenizer should still be alive due to strong references")
        self.assertIsNotNone(assistant_ref(), "Assistant tokenizer should still be alive due to strong references")
        self.assertIsNotNone(translator_ref(), "Translator should still be alive due to strong references")


@require_mindspore
class TestUniversalSpeculativeDecoding(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.target_name = "hf-internal-testing/tiny-random-LlamaForCausalLM"
        cls.assistant_name = "hf-internal-testing/tiny-random-PhiForCausalLM"

    def setUp(self):
        self.target_tokenizer = AutoTokenizer.from_pretrained(self.target_name)
        self.target_config = AutoConfig.from_pretrained(self.target_name)
        self.assistant_model = AutoModelForCausalLM.from_pretrained(self.assistant_name)
        self.assistant_tokenizer = AutoTokenizer.from_pretrained(self.assistant_name)

        self.generation_config = GenerationConfig()

        # Ensure required tokens exist
        if self.target_tokenizer.pad_token_id is None:
            self.target_tokenizer.pad_token_id = self.target_tokenizer.eos_token_id
        if self.target_tokenizer.bos_token_id is None:
            self.target_tokenizer.bos_token_id = self.target_tokenizer.eos_token_id
        if self.assistant_tokenizer.pad_token_id is None:
            self.assistant_tokenizer.pad_token_id = self.assistant_tokenizer.eos_token_id
        if self.target_tokenizer.bos_token_id is None:
            self.assistant_tokenizer.bos_token_id = self.assistant_tokenizer.eos_token_id

        self.input_ids = ms.tensor([[1, 2, 3]])
        self.model_kwargs = {
            "attention_mask": mint.ones_like(self.input_ids),
        }

        atm_translator = AssistantVocabTranslatorCache.get_translator(
            self.target_tokenizer,
            self.assistant_tokenizer,
            self.target_config.vocab_size,
        )
        self.generator = UniversalSpeculativeDecodingGenerator(
            input_ids=self.input_ids,
            assistant_model=self.assistant_model,
            target_tokenizer=self.target_tokenizer,
            assistant_tokenizer=self.assistant_tokenizer,
            generation_config=self.generation_config,
            model_kwargs=self.model_kwargs,
            atm_translator=atm_translator,
        )
