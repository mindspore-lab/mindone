#!/usr/bin/env bash
set -e
set -x

data_dir=${data_dir:-marigold-data}
mkdir -p $data_dir
cd $data_dir

train_url="https://source-xihe-mindspore.osinfra.cn/Braval/Marigold-Train.git"
git clone "${train_url}"

cd Marigold-Train
mkdir depth
mkdir rgb
rm -rf .git
tar -xf depth1.tar -C ./depth
rm depth1.tar
tar -xf depth2.tar -C ./depth
rm depth2.tar
tar -xf depth3.tar -C ./depth
rm depth3.tar
tar -xf rgb1.tar -C ./rgb
rm rgb1.tar
tar -xf rgb2.tar -C ./rgb
rm rgb2.tar
tar -xf rgb3.tar -C ./rgb
rm rgb3.tar
tar -cf vkitti.tar ./depth ./rgb
mkdir vkitti
mv vkitti.tar ./vkitti
rm -rf depth
rm -rf rgb
cd ..
mv Marigold-Train/vkitti .

rm -rf Marigold-Train
