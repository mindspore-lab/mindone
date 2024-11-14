<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Fine-tune a pretrained model

There are significant benefits to using a pretrained model. It reduces computation costs, your carbon footprint, and allows you to use state-of-the-art models without having to train one from scratch. 🤗 Transformers provides access to thousands of pretrained models for a wide range of tasks. When you use a pretrained model, you train it on a dataset specific to your task. This is known as fine-tuning, an incredibly powerful training technique. In this tutorial, you will fine-tune a pretrained model with a deep learning framework of your choice:

- Fine-tune a pretrained model with 🤗 Transformers Trainer.
- Fine-tune a pretrained model in native MindSpore.

## Prepare a dataset

Before you can fine-tune a pretrained model, download a dataset and prepare it for training. The previous tutorial showed you how to process data for training, and now you get an opportunity to put those skills to the test!

Begin by loading the Yelp Reviews dataset:

```pycon
>>> from datasets import load_dataset

>>> dataset = load_dataset("yelp_review_full")
>>> dataset["train"][100]
{'label': 0,
 'text': 'My expectations for McDonalds are t rarely high. But for one to still fail so spectacularly...that takes something special!\\nThe cashier took my friends\'s order, then promptly ignored me. I had to force myself in front of a cashier who opened his register to wait on the person BEHIND me. I waited over five minutes for a gigantic order that included precisely one kid\'s meal. After watching two people who ordered after me be handed their food, I asked where mine was. The manager started yelling at the cashiers for \\"serving off their orders\\" when they didn\'t have their food. But neither cashier was anywhere near those controls, and the manager was the one serving food to customers and clearing the boards.\\nThe manager was rude when giving me my order. She didn\'t make sure that I had everything ON MY RECEIPT, and never even had the decency to apologize that I felt I was getting poor service.\\nI\'ve eaten at various McDonalds restaurants for over 30 years. I\'ve worked at more than one location. I expect bad days, bad moods, and the occasional mistake. But I have yet to have a decent experience at this store. It will remain a place I avoid unless someone in my party needs to avoid illness from low blood sugar. Perhaps I should go back to the racially biased service of Steak n Shake instead!'}
```

As you now know, you need a tokenizer to process the text and include a padding and truncation strategy to handle any variable sequence lengths. To process your dataset in one step, use 🤗 Datasets map method to apply a preprocessing function over the entire dataset:

```pycon
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")


>>> def tokenize_function(examples):
...     return tokenizer(examples["text"], padding="max_length", truncation=True)


>>> tokenized_datasets = dataset.map(tokenize_function, batched=True)
```

If you like, you can create a smaller subset of the full dataset to fine-tune on to reduce the time it takes:

```pycon
small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))
```

## Train

At this point, you should follow the section corresponding to the framework you want to use. You can use the links in the right sidebar to jump to the one you want - and if you want to hide all of the content for a given framework, just use the button at the top-right of that framework’s block!

### Train with MindSpore Trainer

<details open>

!!! Note

    Taking bert as an example, you can find the complete code in `examples/transformers/bert/finetune_with_mindspore_trainer.py`

🤗 Transformers provides a Trainer class optimized for training 🤗 Transformers models, making it easier to start training without manually writing your own training loop. The Trainer API supports a wide range of training options and features such as logging, gradient accumulation, and mixed precision.

Start by loading your model and specify the number of expected labels. From the Yelp Review dataset card, you know there are five labels:

```pycon
>>> from mindone.transformers.models.bert import BertForSequenceClassification

>>> model = BertForSequenceClassification.from_pretrained("google-bert/bert-base-cased", num_labels=5)
```

!!! Note

    You will see a warning about some of the pretrained weights not being used and some weights being randomly initialized. Don’t worry, this is completely normal! The pretrained head of the BERT model is discarded, and replaced with a randomly initialized classification head. You will fine-tune this new model head on your sequence classification task, transferring the knowledge of the pretrained model to it.

#### Training hyperparameters

Next, create a TrainingArguments class which contains all the hyperparameters you can tune as well as flags for activating different training options. For this tutorial you can start with the default training hyperparameters, but feel free to experiment with these to find your optimal settings.

Specify where to save the checkpoints from your training:

```pycon
>>> from mindone.transformers.training_args import TrainingArguments

>>> training_args = TrainingArguments(output_dir="test_trainer")
```

(optional but recommended) Init environment:

```pycon
>>> import mindspore as ms
>>> from mindone.transformers.mindspore_adapter import MindSporeArguments, init_environment

>>> env_args = MindSporeArguments(mode=ms.GRAPH_MODE, device_target="Ascend")
>>> init_environment(env_args)
```

#### Trainer

Create a Trainer object with your model, training arguments, training and test datasets, and evaluation function:

```pycon
>>> trainer = Trainer(
...     model=model,
...     args=training_args,
...     train_dataset=small_train_dataset,
...     eval_dataset=small_eval_dataset,
...     compute_metrics=compute_metrics,
... )
```

Then fine-tune your model by calling train():

```pycon
>>> trainer.train()
```

</details>

### Train in native MindSpore

<details open>

!!! Note

    Taking bert as an example, you can find the complete code in `examples/transformers/bert/finetune_in_native_mindspore.py`

Trainer takes care of the training loop and allows you to fine-tune a model in a single line of code. For users who prefer to write their own training loop, you can also fine-tune a 🤗 Transformers model in native MindSpore.

At this point, you may need to restart your notebook to free memory.

Next, manually postprocess `tokenized_dataset` to prepare it for training.

1. Remove the text column because the model does not accept raw text as an input:

```pycon
>>> tokenized_datasets = tokenized_datasets.remove_columns(["text"])
```

2. Rename the `label` column to `labels` because the model expects the argument to be named `labels`:

```pycon
>>> tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
```

#### DataLoader

Create a MindSpore DataLoader for your training datasets so you can iterate over batches of data:

```pycon
>>> import mindspore as ms
>>> from mindone.transformers.mindspore_adapter import HF2MSDataset

>>> def ms_data_collator(features, batch_info):
...    batch = {}
...     for k, v in features[0]:
...         batch[k] = np.stack([f[k] for f in features]) if isinstance(v, np.ndarray) else np.array([f[k] for f in features])
...     return batch

>>> batch_size, num_epochs = 1, 3
>>> train_dataloader = ms.dataset.GeneratorDataset(HF2MSDataset(small_train_dataset), column_names="item")
>>> train_dataloader = train_dataloader.batch(batch_size=batch_size, per_batch_map=ms_data_collator)
>>> train_dataloader = train_dataloader.repeat(1)
>>> train_dataloader = train_dataloader.create_dict_iterator(num_epochs=num_epochs, output_numpy=True)
```

Load your model with the number of expected labels:

```pycon
>>> from mindone.transformers.models.bert import BertForSequenceClassification

>>> model = BertForSequenceClassification.from_pretrained("google-bert/bert-base-cased", num_labels=5)
```

#### Optimizer

Create an optimizer to fine-tune the model. Let’s use the AdamWeightDecay optimizer from MindSpore:

```pycon
>>> from mindspore import nn

>>> optimizer = nn.AdamWeightDecay(model.trainable_params(), learning_rate=5e-6)
```

#### Train Network

Create an MindSpore train network

```pycon
>>> from mindone.transformers.mindspore_adapter import TrainOneStepWrapper

>>> class ReturnLoss(nn.Cell):
...     def __init__(self, model):
...         super(ReturnLoss, self).__init__(auto_prefix=False)
...         self.model = model
...
...     def construct(self, *args, **kwargs):
...         outputs = self.model(*args, **kwargs)
...         loss = outputs[0]
...         return loss

>>> train_model = TrainOneStepWrapper(ReturnLoss(model), optimizer)
```

Great, now you are ready to train! 🥳

#### Training loop

To keep track of your training progress, use the tqdm library to add a progress bar over the number of training steps:

```pycon
>>> from tqdm.auto import tqdm

>>> num_training_steps = len(small_train_dataset) * num_epochs // batch_size
>>> progress_bar = tqdm(range(num_training_steps))

>>> train_model.train()
>>> for step, batch in enumerate(train_dataloader):
...     batch = batch["item"]
...
...     tuple_inputs = (
...         ms.Tensor(batch["input_ids"], ms.int32),
...         ms.Tensor(batch["attention_mask"], ms.bool_),
...         None,
...         None,
...         None,
...         None,
...         ms.tensor(batch["labels"], ms.int32)
...     )
...
...     loss, _, overflow = train_model(*tuple_inputs)
...
...     progress_bar.update(1)
```

</details>
