import numpy as np
import pytest
import torch
from transformers import LiltConfig

import mindspore as ms

from tests.modeling_test_utils import compute_diffs, generalized_parse_args, get_modules
from tests.transformers_tests.models.modeling_common import ids_numpy

DTYPE_AND_THRESHOLDS = {"fp32": 5e-4, "fp16": 5e-2, "bf16": 5e-2}
MODES = [1]


class LiltModelTester:
    def __init__(
        self,
        batch_size=2,
        seq_length=16,
        is_training=False,
        use_attention_mask=True,
        vocab_size=101,
        hidden_size=72,
        num_hidden_layers=2,
        num_attention_heads=6,
        intermediate_size=288,
        hidden_act="gelu",
        max_position_embeddings=64,
        type_vocab_size=2,
        max_2d_position_embeddings=128,
        channel_shrink_ratio=4,
        torch_dtype="float32",
        # heads
        num_labels_seq=3,
        num_labels_tok=7,
        num_labels_qa=2,
    ):
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_attention_mask = use_attention_mask

        # text & encoder shape
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.max_2d_position_embeddings = max_2d_position_embeddings
        self.channel_shrink_ratio = channel_shrink_ratio
        self.torch_dtype = torch_dtype

        # task heads
        self.num_labels_seq = num_labels_seq
        self.num_labels_tok = num_labels_tok
        self.num_labels_qa = num_labels_qa

    def get_base_config(self):
        return LiltConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            max_position_embeddings=self.max_position_embeddings,
            type_vocab_size=self.type_vocab_size,
            max_2d_position_embeddings=self.max_2d_position_embeddings,
            channel_shrink_ratio=self.channel_shrink_ratio,
            torch_dtype=self.torch_dtype,
        )

    def get_seqcls_config(self):
        cfg = self.get_base_config()
        cfg.num_labels = self.num_labels_seq
        return cfg

    def get_tokcls_config(self):
        cfg = self.get_base_config()
        cfg.num_labels = self.num_labels_tok
        return cfg

    def get_qa_config(self):
        cfg = self.get_base_config()
        cfg.num_labels = self.num_labels_qa  # 2
        return cfg

    def _make_bbox(self, batch, seq, vmax):
        """
        create valid (x0,y0,x1,y1) integer boxes in [0, vmax-1] with x1>=x0 and y1>=y0.
        shape: (B, S, 4)
        """
        x0 = ids_numpy([batch, seq], vocab_size=vmax)
        y0 = ids_numpy([batch, seq], vocab_size=vmax)

        w = ids_numpy([batch, seq], vocab_size=max(1, vmax // 4))
        h = ids_numpy([batch, seq], vocab_size=max(1, vmax // 4))
        x1 = np.clip(x0 + w, 0, vmax - 1)
        y1 = np.clip(y0 + h, 0, vmax - 1)
        bbox = np.stack([x0, y0, x1, y1], axis=-1).astype(np.int64)
        return bbox

    def prepare_config_and_inputs(self):
        base_config = self.get_base_config()

        input_ids = ids_numpy([self.batch_size, self.seq_length], self.vocab_size)

        attention_mask = None
        if self.use_attention_mask:
            attention_mask = ids_numpy([self.batch_size, self.seq_length], vocab_size=2)

        bbox = self._make_bbox(
            batch=self.batch_size,
            seq=self.seq_length,
            vmax=self.max_2d_position_embeddings,
        )

        token_type_ids = ids_numpy([self.batch_size, self.seq_length], vocab_size=self.type_vocab_size)

        return base_config, input_ids, attention_mask, bbox, token_type_ids

    def prepare_task_configs(self):
        return self.get_seqcls_config(), self.get_tokcls_config(), self.get_qa_config()


model_tester = LiltModelTester()
base_config, input_ids, attention_mask, bbox, token_type_ids = model_tester.prepare_config_and_inputs()
cfg_seq, cfg_tok, cfg_qa = model_tester.prepare_task_configs()


TEST_CASES = [
    [  # base encoder (with pooling)
        "LiltModel",
        "transformers.LiltModel",
        "mindone.transformers.LiltModel",
        (base_config,),
        {},
        (),
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "bbox": bbox,
            "token_type_ids": token_type_ids,
        },
        {
            "last_hidden_state": 0,
            "pooler_output": 1,
        },
    ],
    [  # sequence classification head
        "LiltForSequenceClassification",
        "transformers.LiltForSequenceClassification",
        "mindone.transformers.LiltForSequenceClassification",
        (cfg_seq,),
        {},
        (),
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "bbox": bbox,
            "token_type_ids": token_type_ids,
        },
        {
            "logits": 0,
        },
    ],
    [  # token classification head
        "LiltForTokenClassification",
        "transformers.LiltForTokenClassification",
        "mindone.transformers.LiltForTokenClassification",
        (cfg_tok,),
        {},
        (),
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "bbox": bbox,
            "token_type_ids": token_type_ids,
        },
        {
            "logits": 0,
        },
    ],
    [  # QA head (start/end logits)
        "LiltForQuestionAnswering",
        "transformers.LiltForQuestionAnswering",
        "mindone.transformers.LiltForQuestionAnswering",
        (cfg_qa,),
        {},
        (),
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "bbox": bbox,
            "token_type_ids": token_type_ids,
        },
        {
            "start_logits": 0,
            "end_logits": 1,
        },
    ],
]


@pytest.mark.parametrize(
    "name,pt_module,ms_module,init_args,init_kwargs,inputs_args,inputs_kwargs,outputs_map,dtype,mode",
    [case + [dtype] + [mode] for case in TEST_CASES for dtype in DTYPE_AND_THRESHOLDS.keys() for mode in MODES],
)
def test_named_modules(
    name,
    pt_module,
    ms_module,
    init_args,
    init_kwargs,
    inputs_args,
    inputs_kwargs,
    outputs_map,
    dtype,
    mode,
):
    ms.set_context(mode=mode)

    (
        pt_model,
        ms_model,
        pt_dtype,
        ms_dtype,
    ) = get_modules(pt_module, ms_module, dtype, *init_args, **init_kwargs)

    pt_inputs_args, pt_inputs_kwargs, ms_inputs_args, ms_inputs_kwargs = generalized_parse_args(
        pt_dtype, ms_dtype, *inputs_args, **inputs_kwargs
    )

    with torch.no_grad():
        pt_outputs = pt_model(*pt_inputs_args, **pt_inputs_kwargs)
    ms_outputs = ms_model(*ms_inputs_args, **ms_inputs_kwargs)

    if outputs_map:
        pt_outputs_n = []
        ms_outputs_n = []
        for pt_key, ms_idx in outputs_map.items():
            pt_output = getattr(pt_outputs, pt_key)
            ms_output = ms_outputs[ms_idx]
            if isinstance(pt_output, (list, tuple)):
                pt_outputs_n += list(pt_output)
                ms_outputs_n += list(ms_output)
            else:
                pt_outputs_n.append(pt_output)
                ms_outputs_n.append(ms_output)
        diffs = compute_diffs(pt_outputs_n, ms_outputs_n)
    else:
        diffs = compute_diffs(pt_outputs, ms_outputs)

    THRESHOLD = DTYPE_AND_THRESHOLDS[ms_dtype]
    assert (np.array(diffs) < THRESHOLD).all(), (
        f"ms_dtype: {ms_dtype}, pt_type:{pt_dtype}, "
        f"Outputs({np.array(diffs).tolist()}) has diff bigger than {THRESHOLD}"
    )
