#!/usr/bin/env bash
set -e
set -x

ckpt_dir=${ckpt_dir:-marigold-checkpoint}
mkdir -p $ckpt_dir
cd $ckpt_dir

model_url="https://source-xihe-mindspore.osinfra.cn/Braval/Marigold-Model.git"
git clone "${model_url}"

cd Marigold-Model
git lfs install
git lfs pull
rm -rf .git
mv marigold-checkpoint/stable-diffusion-2 ../marigold-v1-0

cd ..
rm -rf Marigold-Model
