def process_prompt_and_text_embed(prompt, text_embed_path):
    """
    Process the prompt and text embed path to ensure they are in the correct format and match in number.

    Args:
        prompt (str or None): A string prompt or path to a text file containing prompts.
        text_embed_path (str or None): A path to a text file containing text embeddings or a npz file.

    Returns:
        tuple: A tuple containing two lists, the first with prompts and the second with text embed paths.

    Raises:
        ValueError: If the prompt and text embed path combination is invalid or the number of prompts and text embed paths do not match.
    """
    if prompt is None and text_embed_path is None:
        raise ValueError("Either `prompt` or `text_embed_path` must be provided.")

    if prompt is not None and prompt.endswith(".txt"):
        with open(prompt, "r") as f:
            prompts = [line.strip() for line in f.readlines()]
        if text_embed_path is not None:
            assert text_embed_path.endswith(
                ".txt"
            ), "When `prompt` is a txt file, `text_embed_path` should be a txt file too."
            with open(text_embed_path, "r") as f:
                text_embed_paths = [line.strip() for line in f.readlines()]
        else:
            text_embed_paths = [None] * len(prompts)
    elif prompt is not None and isinstance(prompt, str):
        prompts = [prompt.strip()]
        if text_embed_path is not None:
            assert text_embed_path.endswith(
                ".npz"
            ), "When `prompt` is a string, `text_embed_path` should be a npz file."
            text_embed_paths = [text_embed_path.strip()]
        else:
            text_embed_paths = [None] * len(prompts)
    elif text_embed_path is not None:
        if text_embed_path.endswith(".txt"):
            with open(text_embed_path, "r") as f:
                text_embed_paths = [line.strip() for line in f.readlines()]
            prompts = [None] * len(text_embed_paths)
        else:
            text_embed_paths = [text_embed_path.strip()]
            prompts = [None]
    else:
        raise ValueError("Invalid combination of `prompt` and `text_embed_path`.")

    if len(prompts) != len(text_embed_paths):
        raise ValueError("The number of prompts and text embed paths must match.")

    return prompts, text_embed_paths
