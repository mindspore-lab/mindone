# export RANK_ID=0

source=v2-1_768-ema-pruned.ckpt
python pt2np_v2.py $source
python np2ms_v2.py $source

source=v1-5-pruned.ckpt
python pt2np_v1.py $source
python np2ms_v1.py $source
