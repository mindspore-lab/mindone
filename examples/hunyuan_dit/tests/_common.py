import numpy as np
from transformers import BertTokenizer

np.random.seed(0)

prompt = "一只聪明的狐狸走在阔叶树林里, 旁边是一条小溪, 细节真实, 摄影"

tokenizer = BertTokenizer.from_pretrained("../ckpts/t2i/tokenizer/")
x = tokenizer(prompt, return_tensors="np")

inputs = [
    np.random.randn(2, 4, 128, 128).astype(np.float16),
    np.random.randn(
        2,
    ).astype(np.float16),
]

random_angles = np.random.uniform(0, 2 * np.pi, (4096, 88))
kwargs = {
    "encoder_hidden_states": np.random.randn(2, 77, 1024).astype(np.float16),
    "text_embedding_mask": np.random.randint(0, 1, size=(2, 77)),
    "encoder_hidden_states_t5": np.random.randn(2, 256, 2048).astype(np.float16),
    "text_embedding_mask_t5": np.random.randint(0, 1, size=(2, 256)),
    "image_meta_size": None,
    "style": None,
    "cos_cis_img": np.cos(random_angles).astype(np.float32),
    "sin_cis_img": np.sin(random_angles).astype(np.float32),
}
