import os
import json
import argparse
from PIL import Image

import mindspore
from mindspore import ops

from clip_score import CLIPModel, CLIPImageProcessor, CLIPTokenizer, parse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        default= 'eval/clip_score/configs/clip_vit_b_16.yaml', type=str,
        help='YAML config files for ms backend'
             ' Default: ./configs/clip_vit_b_16.yaml')
    parser.add_argument(
        '--model_name',
        default='openai/clip-vit-base-patch16', type=str,
        help='the name of a (Open/)CLIP model as shown in HuggingFace for pt backend.'
             ' Default: openai/clip-vit-base-patch16'
    )
    parser.add_argument(
        '--image_path', default=None, type=str,
        help='input data for predict, it support real data path or data directory.'
             ' Default: None')
    parser.add_argument(
        '--prompt', default=None, type=str,
        help='prompt corresponding to the image from image path.'
             ' Default: None')
    parser.add_argument(
        '--backend', default='ms', type=str,
        help='backend to do CLIP model inference for CLIP score compute. Option: ms, pt.'
             ' Default: ms')
    parser.add_argument(
        '--load_checkpoint', default=None, type=str,
        help='load model checkpoint.'
             ' Default: None')
    parser.add_argument(
        '--tokenizer_path', default='bpe_simple_vocab_16e6.txt.gz', type=str,
        help='load model checkpoint.'
             ' Default: bpe_simple_vocab_16e6.txt.gz')
    parser.add_argument(
        '--save_result', default=True, type=str,
        help='save results or not, if set to True then save to result_path.'
        ' Default: True'
    )
    parser.add_argument(
        '--result_path', default='results.jsonl', type=str,
        help='the path for saving results if save_result is set to True.'
        ' Default: results.jsonl'
    )
    args = parser.parse_args()

    # load images
    assert args.image_path is not None
    images = []
    if os.path.isdir(args.image_path) and os.path.exists(args.image_path):
        image_path = [os.path.join(root, file)
                        for root, _, file_list in os.walk(os.path.join(args.image_path)) for file in file_list
                        if file.endswith('.jpg') or file.endswith('.png') or file.endswith('.jpeg')
                        or file.endswith('.JPEG') or file.endswith('bmp')]
        image_path.sort()
        images = [Image.open(p) for p in image_path]
        args.image_path = image_path
    else:
        images = [Image.open(args.image_path)]
        args.image_path = [args.image_path]

    # load prompts
    assert args.prompt is not None
    texts = []
    if os.path.exists(args.prompt):
        with open(args.prompt) as f:
            texts = [p.strip() for p in f.readlines()]
        args.prompt = texts
    else:
        texts = [args.prompt]
    assert len(images) == len(texts)
    print(f'{len(images)} image-text pairs are loaded')

    print(f'Backend: {args.backend}')
    if args.backend == 'pt':
        from clip_score import compute_torchmetric_clip
        score = compute_torchmetric_clip(images, texts, model_name=args.model_name)
    elif args.backend == 'ms':
        image_processor = CLIPImageProcessor()
        text_processor = CLIPTokenizer(args.tokenizer_path, pad_token='!')
        process_text = lambda p: mindspore.Tensor(text_processor(p, padding='max_length', max_length=77) \
                                                ['input_ids']).reshape(1, -1)
        images = [image_processor(image) for image in images]
        texts = [process_text(text) for text in texts]

        # parse config file
        config = parse(args.config, args.load_checkpoint)
        model = CLIPModel(config)

        inputs = [(images[i], texts[i]) for i in range(len(images))]
        results = []
        for i, (image, text) in enumerate(inputs):
            image_feature = model.get_image_features(image)
            image_feature = image_feature / image_feature.norm(1, keep_dims=True)
            text_feature = model.get_text_features(text)
            text_feature = text_feature / text_feature.norm(1, keep_dims=True)
            result = float(ops.matmul(image_feature, text_feature.T)[0][0] * 100)
            results.append(result)
            print(args.image_path[i], args.prompt[i], '->', round(result, 4))
        score = sum(results) / len(results)

        # save results
        if args.save_result:
            with open(args.result_path, 'w') as f:
                for i in range(len(inputs)):
                    line = {
                        'image_path': os.path.abspath(args.image_path[i]),
                        'prompt': args.prompt[i],
                        'clip_score': results[i]
                    }
                    f.write(json.dumps(line) + '\n')
    else:
        raise ValueError(f'Unknown backend: {args.backend}. Valid backend: [ms, pt]')

    print('Mean score =', score)

    