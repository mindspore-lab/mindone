#!/bin/bash

server=159.138.153.179  # Please make sure your local machine can access the remote server via ssh without password. (add local id_rsa.pub to remote ~/.ssh/authorized_keys)
root=data2
set=2M_train
partid=4

for tar_name in 22 24 26 27 28 31 34 35 37 42 46 47
do
        tar_path=/$root/webvid-10m/dataset/$set/part$partid/000$tar_name.tar
        echo Downloading $tar_path ...
        scp -r root@$server:$tar_path /e/webvid/dataset/$set/part$partid/ && echo Successfully downloaded!
done
