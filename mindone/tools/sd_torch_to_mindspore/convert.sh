# export RANK_ID=0

source=v2-1_768-ema-pruned.ckpt
python pt2np.py $(source)
python np2ms.py $(source)
