
from examples.nextgpt.core_code.model.anyToImageVideoAudio import NextGPTModel


def load_model(args):
    agent_name = args['models'][args['model']]['agent_name']
    model_name = args['models'][args['model']]['model_name']
    model = globals()[model_name](**args)

    # agent = globals()[agent_name](model, args)
    return model
