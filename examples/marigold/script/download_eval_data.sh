#!/usr/bin/env bash
set -e
set -x

data_dir=${data_dir:-marigold-data}
mkdir -p $data_dir
cd $data_dir

mkdir -p kitti
mkdir -r nyuv2

wget -r -np -nH --cut-dirs=1 --show-progress -P kitti https://share.phys.ethz.ch/~pf/bingkedata/marigold/evaluation_dataset/kitti
wget -r -np -nH --cut-dirs=1 --show-progress -P nyuv2 https://share.phys.ethz.ch/~pf/bingkedata/marigold/evaluation_dataset/nyuv2
