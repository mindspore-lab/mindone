from vc.models.droppath import DropPathWithControl, gen_zero_keep_mask
import mindspore as ms
from mindspore import ops
import numpy as np

def test_mask_gen():
    zero_mask, keep_mask = gen_zero_keep_mask(0.2, 0.2, 8)  
    print(zero_mask, keep_mask)

def test():
    bs = 4
    d = 2
    #one_mask = ms.Tensor(np.ones([bs, 1]), ms.float32) 
    
    x = ms.Tensor(np.ones([bs, d]))

    p_all_zero = p_all_keep = 0.2

    drop_prob = 0.5
    dp = DropPathWithControl(drop_prob, scale_by_keep=False )
    dp.set_train(True)

    ms.set_context(mode=0)
    
    bernoulli0 = ops.Dropout(keep_prob=p_all_zero) # used to generate zero_mask for droppath on conditions
    bernoulli1= ops.Dropout(keep_prob=p_all_keep)
    
    #zero_mask= ms.numpy.rand((bs, 1)) < p_all_zero
    #keep_mask= ms.numpy.rand((bs, 1)) < p_all_keep

    type_dist = ms.Tensor([p_all_zero, p_all_keep, 1 - (p_all_zero + p_all_keep)]) # used to control keep/drop all conditions for a sample
    sample_type = ops.multinomial(type_dist, bs)
    zero_mask = sample_type == 0
    keep_mask = sample_type == 1

    print("zero mask: ", zero_mask)
    print("keep mask: ", keep_mask)
    for i in range(20):
        #zero_mask = ops.Dropout(keep_prob=p_all_zero)(one_mask)[0] * p_all_zero
        #keep_mask = ops.Dropout(keep_prob=p_all_keep)(one_mask)[0] * p_all_keep
        #one_mask = ops.ones([bs, 1])
        #zero_mask = bernoulli0(one_mask)[0] * p_all_zero
        #keep_mask = bernoulli1(one_mask)[0] * p_all_keep
        #print("zero mask: ", zero_mask)
        #print("keep mask: ", keep_mask)
        print(x+dp(x*2, zero_mask=zero_mask, keep_mask=keep_mask))

if __name__ == '__main__':
    #test_mask_gen()
    test()
