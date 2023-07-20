def compute_torchmetric_clip(images, texts, model_name):
    import torch
    from torchmetrics.functional.multimodal import clip_score
    from functools import partial
    import torchvision.transforms as transforms

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
