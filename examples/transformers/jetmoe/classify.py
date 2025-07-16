# Adapted from https://huggingface.co/docs/transformers/model_doc/jetmoe#transformers.JetMoeForSequenceClassification.forward.example-2
from transformers import AutoTokenizer

import mindspore as ms
from mindspore import Tensor, mint

from mindone.transformers import JetMoeForSequenceClassification


def main():
    tokenizer = AutoTokenizer.from_pretrained("jetmoe/jetmoe-8b")
    model = JetMoeForSequenceClassification.from_pretrained(
        "jetmoe/jetmoe-8b", problem_type="multi_label_classification"
    )

    inputs = Tensor(tokenizer("Hello, my dog is cute", return_tensors="np").input_ids)

    with ms._no_grad():
        logits = model(input_ids=inputs).logits

    predicted_class_ids = mint.arange(0, logits.shape[-1])[mint.sigmoid(logits).squeeze(dim=0) > 0.5]
    print(predicted_class_ids)


if __name__ == "__main__":
    main()
