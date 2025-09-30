<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# MADLAD-400

## Overview

MADLAD-400 models were released in the paper [MADLAD-400: A Multilingual And Document-Level Large Audited Dataset](MADLAD-400: A Multilingual And Document-Level Large Audited Dataset).

The abstract from the paper is the following:

*We introduce MADLAD-400, a manually audited, general domain 3T token monolingual dataset based on CommonCrawl, spanning 419 languages. We discuss
the limitations revealed by self-auditing MADLAD-400, and the role data auditing
had in the dataset creation process. We then train and release a 10.7B-parameter
multilingual machine translation model on 250 billion tokens covering over 450
languages using publicly available data, and find that it is competitive with models
that are significantly larger, and report the results on different domains. In addition, we train a 8B-parameter language model, and assess the results on few-shot
translation. We make the baseline models 1
available to the research community.*

This model was added by [Juarez Bochi](https://huggingface.co/jbochi). The original checkpoints can be found [here](https://github.com/google-research/google-research/tree/master/madlad_400).

This is a machine translation model that supports many low-resource languages, and that is competitive with models that are significantly larger.

One can directly use MADLAD-400 weights without finetuning the model:

```python
import mindspore as ms
from transformers import AutoTokenizer
from mindone.transformers import AutoModelForSeq2SeqLM

model = AutoModelForSeq2SeqLM.from_pretrained("google/madlad400-3b-mt")
tokenizer = AutoTokenizer.from_pretrained("google/madlad400-3b-mt")

inputs = tokenizer("<2pt> I love pizza!", return_tensors="np")
for key in inputs.keys():
    inputs[key] = ms.tensor(inputs[key])

outputs = model.generate(**inputs)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
# Outputs:
# ['Eu amo pizza!']
```
