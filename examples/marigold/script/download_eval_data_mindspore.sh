#!/usr/bin/env bash
set -e
set -x

data_dir=${data_dir:-marigold-data}
mkdir -p $data_dir
cd $data_dir

eval_url="https://source-xihe-mindspore.osinfra.cn/Braval/Marigold-Eval.git"
git clone "${eval_url}"

mv Marigold-Eval/nyuv2 .
mv Marigold-Eval/kitti .

rm -rf Marigold-Eval
