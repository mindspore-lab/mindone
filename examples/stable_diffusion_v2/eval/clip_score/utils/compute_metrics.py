def compute_torchmetric_clip(images, texts, model_name, no_check_certificate=False):
    from functools import partial

    import torch
    import torchvision.transforms as transforms
    from torchmetrics.functional.multimodal import clip_score

    if no_check_certificate:
        import os

        os.environ["CURL_CA_BUNDLE"] = ""

    clip_score_fn = partial(clip_score, model_name_or_path=model_name)
    transform = transforms.Compose([transforms.PILToTensor()])

    images_ = []
    for image in images:
        img = transform(image)
        if img.shape[0] == 4:
            img = img[:3, :, :]
        images_.append(img)
    images = torch.stack(images_)
    score = clip_score_fn(images, texts).detach()
    return float(score)
