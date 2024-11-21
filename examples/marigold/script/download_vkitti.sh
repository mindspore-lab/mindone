#!/usr/bin/env bash
set -e
set -x

data_dir=${data_dir:-marigold-data}
mkdir -p $data_dir
cd $data_dir

wget -nv --show-progress http://download.europe.naverlabs.com//virtual_kitti_2.0.3/vkitti_2.0.3_rgb.tar
tar -xf vkitti_2.0.3_rgb.tar
rm vkitti_2.0.3_rgb.tar

wget -nv --show-progress http://download.europe.naverlabs.com//virtual_kitti_2.0.3/vkitti_2.0.3_depth.tar
tar -xf vkitti_2.0.3_depth.tar
rm vkitti_2.0.3_depth.tar

mv vkitti_2.0.3_depth depth
mv vkitti_2.0.3_rgb rgb

tar -cf vkitti.tar ./depth ./rgb
mkdir vkitti
mv vkitti.tar ./vkitti
rm -rf depth
rm -rf rgb
