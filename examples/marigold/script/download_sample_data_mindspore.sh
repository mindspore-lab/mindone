#!/usr/bin/env bash
set -e
set -x

data_dir=input
mkdir -p $data_dir
cd $data_dir

if test -f "in-the-wild_example.tar" ; then
    echo "Tar file exists: in-the-wild_example.tar"
    exit 1
fi

example_url="https://source-xihe-mindspore.osinfra.cn/Braval/Marigold-Example.git"
git clone "${example_url}"

cd Marigold-Example
rm -rf .git
mv input/in-the-wild_example ..

cd ..
rm -rf Marigold-Example
