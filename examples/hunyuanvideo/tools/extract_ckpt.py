import torch

def extract(ckpt_path, mm_double_blocks_depth=None, mm_single_blocks_depth=None):
    state_dict = torch.load(ckpt_path)
    load_key = 'module'
    sd = state_dict[load_key]
    pnames = list(sd.keys())

    extract_ckpt = mm_double_blocks_depth is not None mm_single_blocks_depth is not None
    for pname in pnames:
        print(pname, tuple(sd[pname].shape), sd[pname].dtype)
        
        if extract_ckpt: 
            if pname.startswith('double_blocks'):
                idx = int(pname.split('.')[1])
                if idx >= mm_double_blocks_depth:
                    sd.pop(pname)
             if pname.startswith('single_blocks'):
                idx = int(pname.split('.')[1])
                if idx >= mm_single_blocks_depth:
                    sd.pop(pname)

    if extract_ckpt:
        torch.save(state_dict, 'ckpts/dit_small.pt')

if __name__ == '__main__':
    extract() 

