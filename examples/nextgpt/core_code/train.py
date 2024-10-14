from header import *
from dataset import load_dataset
from model import *
from config import *

def parser_args():
    parser = argparse.ArgumentParser(description='train parameters')
    parser.add_argument('--model', type=str, default='nextgpt')
    parser.add_argument('--mode', type=str, default='train', help='train or test or validation')
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--save_path', type=str, default='../ckpt/log/')
    parser.add_argument('--log_path', type=str, default='../ckpt/log/')
    parser.add_argument('--assets_path', type=str, default='./assets/')

    # model configurations
    parser.add_argument('--max_length', type=int, default=512)  # the maximum input sequence length for LLMs
    parser.add_argument('--stage', type=int, default=1)  # the training stage
    parser.add_argument('--modality', type=list, default=['image', 'video', 'audio', 'text'])
    return parser.parse_args()

def config_env(args):
    args['root_dir'] = '../'
    # args['mode'] = 'train'
    config = load_config(args)
    args.update(config)
    # initialize_distributed(args)
    # set_random_seed(args['seed'])

def main(**args):
    config_env(args)
    print(args)
    args['ds_config_path'] = f'dsconfig/stage_{args["stage"]}.json'


    if args['log_path']:
        logging.basicConfig(
            format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
            level=logging.DEBUG,
            filename=f'{args["log_path"]}/train_{time.asctime()}.log',
            filemode='w'
        )
    train_data = load_dataset(args, args['dataset_name_list'])

    agent = load_model(args)


if __name__ == "__main__":
    args = parser_args()
    args = vars(args)
    main(**args)