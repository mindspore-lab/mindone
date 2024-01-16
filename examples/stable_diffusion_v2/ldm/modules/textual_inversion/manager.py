import mindspore as ms


class TextualInversionManager:
    def __init__(self, model, placeholder_token=None, num_vectors=None):
        self.model = model
        self.placeholder_token = placeholder_token
        self.num_vectors = num_vectors
        if self.placeholder_token is None or self.num_vectors is None:
            print(
                "TextualInversionManager is not READY. Please run `load_checkpoint_textual_inversion` first, "
                "or set placeholder_token and num_vectors."
            )

    @property
    def placeholder_tokens(self):
        return self._get_placeholder_tokens(self.placeholder_token, self.num_vectors)

    @staticmethod
    def _get_placeholder_tokens(placeholder_token, num_vectors):
        placeholder_tokens = [placeholder_token]

        if num_vectors < 1:
            raise ValueError(f"--num_vectors has to be larger or equal to 1, but is {num_vectors}")

        # add dummy tokens for multi-vector
        additional_tokens = []
        for i in range(1, num_vectors):
            additional_tokens.append(f"{placeholder_token}_{i}")
        placeholder_tokens += additional_tokens
        return placeholder_tokens

    @staticmethod
    def initiate_new_text_embeds(token_embeds, placeholder_token_ids, initializer_token_id=None):
        dtype = token_embeds.dtype
        for token_id in placeholder_token_ids:
            if initializer_token_id is not None:
                token_embeds[token_id] = token_embeds[initializer_token_id].astype(dtype)
            else:
                token_embeds[token_id] = ms.ops.randn_like(token_embeds[token_id]) * 0.01  # start with [-0.04,+0.04]

    @property
    def text_encoders(self):
        return [self.model.cond_stage_model]

    @property
    def tokenizers(self):
        return [self.model.cond_stage_model.tokenizer]

    def manage_prompt(self, prompt):
        placeholder_tokens = " ".join(self.placeholder_tokens)
        if placeholder_tokens not in prompt:
            if self.placeholder_token in prompt:
                prompt = prompt.replace(self.placeholder_token, placeholder_tokens)
            else:
                print(
                    f"WARNING: the placeholder token {self.placeholder_token} should be in the text prompt, but not found!"
                )
        return prompt

    def initiate_textual_inversion_params(self, initializer_token=None):
        model = self.model
        placeholder_tokens = self.placeholder_tokens
        tokenizers = self.tokenizers
        text_encoders = self.text_encoders
        placeholder_token, num_vectors = self.placeholder_token, self.num_vectors

        # add tokens to tokenizer, and initiate the newly added text embedding (random or use initializer token)
        for tokenizer, text_encoder in zip(tokenizers, text_encoders):
            num_added_tokens = tokenizer.add_tokens(placeholder_tokens)
            if num_added_tokens != num_vectors:
                raise ValueError(
                    f"The tokenizer already contains the token {placeholder_token}. Please pass a different"
                    " `placeholder_token` that is not already in the tokenizer."
                )
            placeholder_token_ids = tokenizer.convert_tokens_to_ids(placeholder_tokens)
            # initiate a new embedding which has the same name, dtype, requires_grad as the old embedding, but shape is different
            text_encoder.resize_token_embeddings(len(tokenizer))

            # Initialise the newly added placeholder token (random or use initializer token)
            if initializer_token is not None:
                # Convert the initializer_token, placeholder_token to ids
                token_ids = tokenizer.encode(initializer_token, add_special_tokens=False)
                # Check if initializer_token is a single token or a sequence of tokens
                if len(token_ids) > 1:
                    raise ValueError("The initializer token must be a single token.")
                initializer_token_id = token_ids[0]
            else:
                initializer_token_id = None

            token_embeds = text_encoder.get_input_embeddings()
            self.initiate_new_text_embeds(token_embeds, placeholder_token_ids, initializer_token_id)

        # update cell.children name prefix recursively since we have replaced the old embedding by the new embedding
        model.update_parameters_name()
        return model

    def get_textual_inversion_params(self):
        params = []
        text_encoders = self.text_encoders
        for text_encoder in text_encoders:
            params.append(text_encoder.get_input_embeddings())
        return params

    def set_train_textual_inversion(self, train_flag=True):
        text_encoders = self.text_encoders
        for text_encoder in text_encoders:
            embedding_table = text_encoder.get_input_embeddings()
            if hasattr(embedding_table, "set_train"):
                embedding_table.set_train(train_flag)

    def save_checkpoint_textual_inversion(
        self, path, num_vectors=None, only_save_ti=True, placeholder_token=None, suffix="ti"
    ):
        # save only the newly learned text embedding
        num_vectors = self.num_vectors if num_vectors is None else num_vectors
        ckpt, ckpt_ti = [], []
        model = self.model
        for n, p in model.parameters_and_names():
            if ".embedding_table" in n:
                new_params = p[-num_vectors:]
                ckpt_ti.append({"name": n, "data": new_params})
            else:
                ckpt.append({"name": n, "data": p})

        if not only_save_ti:
            ms.save_checkpoint(ckpt, path)
            print(f"save checkpoint to {path}")

        if len(ckpt_ti) > 0:
            append_dict = {}
            if placeholder_token is not None:
                assert isinstance(placeholder_token, str), "expect that placeholder_token is a string"
                append_dict["placeholder_token"] = placeholder_token
            else:
                append_dict["placeholder_token"] = self.placeholder_token
            append_dict["num_vectors"] = num_vectors

            path_ti = path[: -len(".ckpt")] + "_{}.ckpt".format(suffix)
            ms.save_checkpoint(ckpt_ti, path_ti, append_dict=append_dict)
            print(f"save textual inversion checkpoint to {path_ti}")

    def load_checkpoint_textual_inversion(self, path, num_vectors=None, placeholder_token=None, verbose=True):
        ti_checkpoint = ms.load_checkpoint(path)
        if "num_vectors" in ti_checkpoint:
            if num_vectors is not None and verbose:
                print(
                    f"Found `num_vectors={ti_checkpoint['ti_checkpoint']}` in the provided checkpoint. Overwrites the num_vectors value {num_vectors}"
                )
            num_vectors = int(ti_checkpoint["num_vectors"].value())
            del ti_checkpoint["num_vectors"]
        elif num_vectors is None:
            print("WARNING: num_vectors is not provided, nor found in the provided checkpoint. Use 1 instead.")
            num_vectors = 1

        if "placeholder_token" in ti_checkpoint:
            if placeholder_token is not None and verbose:
                print(
                    f"Found `num_vectors={ti_checkpoint['placeholder_token']}` in the provided checkpoint. Overwrites the num_vectors value {placeholder_token}"
                )
            placeholder_token = ti_checkpoint["placeholder_token"]
            del ti_checkpoint["placeholder_token"]
        elif placeholder_token is None:
            print(
                "WARNING: placeholder_token is not provided, nor found in the provided checkpoint. Use empty string instead."
            )
            placeholder_token = ""
        self.placeholder_token = placeholder_token
        self.num_vectors = num_vectors
        self.initiate_textual_inversion_params()
        params = ti_checkpoint
        assert len(params) > 0, "the checkpoint is empty!"
        # load the textual inversion params into network
        self.load_textual_inversion_params_into_network(params, verbose=verbose)

    def load_textual_inversion_params_into_network(self, ti_ckpt, verbose=True):
        ti_params = self.get_textual_inversion_params()
        for name in ti_ckpt:
            data = ti_ckpt[name].value()
            assert any([param.name == name for param in ti_params]), f"{name} not found in textual inversion params!"
            for param in ti_params:
                if param.name == name:
                    embedding_table = param
                    num_no_updates = embedding_table.shape[0] - data.shape[0]
                    data_to_copy = ms.ops.concat([embedding_table.value()[:num_no_updates], data], axis=0)
                    ms.ops.Assign()(embedding_table, data_to_copy)
                    if verbose:
                        print(f"Textual Inversion param {name} is loaded into network")
