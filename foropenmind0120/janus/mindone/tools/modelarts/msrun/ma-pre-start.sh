WORK_DIR=mindone/examples/opensora_cai
RUN_DIR=C17
#mindspore_file=mindspore-2.3.0-cp37-cp37m-linux_aarch64.whl
mindspore_file=mindspore-2.3.0.20240423-cp37-cp37m-linux_aarch64.whl
#mindspore_file=mindspore-2.3.0rc1+20240419-cp37-cp37m-linux_aarch64.whl

LOCAL_DIR=/home/ma-user/modelarts/user-job-dir
LOCAL_DIR=${LOCAL_DIR}/../user-job-dir

echo "------------------------"uninstall "------------------------"
sudo bash /usr/local/Ascend/ascend-toolkit/7.0.RC1.1/aarch64-linux/script/uninstall.sh
sudo bash /usr/local/Ascend/ascend-toolkit/7.0/aarch64-linux/script/uninstall.sh
pip uninstall mindspore -y
pip uninstall mindformers -y
sudo rm -rf /usr/local/Ascend/ascend-toolkit/

/home/ma-user/anaconda3/envs/MindSpore/bin/python3.7 -m pip install --upgrade pip

#version=pkg
version=$RUN_DIR
if [ $version == $RUN_DIR ]
then
    echo "-------------------1start install new cann--------------------------"
    sudo bash ${LOCAL_DIR}/${WOatest/bin/setenv.bash
    ls -lrt ${LOCAL_DIR}RK_DIR}/${RUN_DIR}/Ascend-cann-toolkit_*.run --full --install-path=/usr/local/Ascend --quiet
    sudo bash ${LOCAL_DIR}/${WORK_DIR}/${RUN_DIR}/Ascend-cann-kernels-*.run --full --install-path=/usr/local/Ascend --quiet
    source /usr/local/Ascend/ascend-toolkit/l/${WORK_DIR}/${RUN_DIR}
    LIB_PATH=/usr/local/Ascend/ascend-toolkit/latest/fwkacllib/lib64
fi


if [ $version == $RUN_DIR ]
then
    echo "-------------------2start install new cann--------------------------"
    #mindspore_file=mindspore-2.3.0rc1-cp37-cp37m-linux_aarch64.whl
    sudo bash ${LOCAL_DIR}/${WORK_DIR}/${RUN_DIR}/../${version}/CANN-runtime-*-linux.aarch64.run --full --install-path=/usr/local/Ascend --quiet
    sudo bash ${LOCAL_DIR}/${WORK_DIR}/${RUN_DIR}/../${version}/CANN-compiler-*-linux.aarch64.run --full --install-path=/usr/local/Ascend --quiet
    sudo bash ${LOCAL_DIR}/${WORK_DIR}/${RUN_DIR}/../${version}/CANN-opp-*-linux.aarch64.run --full --install-path=/usr/local/Ascend --quiet
    sudo bash ${LOCAL_DIR}/${WORK_DIR}/${RUN_DIR}/../${version}/CANN-toolkit-*-linux.aarch64.run --full --install-path=/usr/local/Ascend --quiet
    sudo bash ${LOCAL_DIR}/${WORK_DIR}/${RUN_DIR}/../${version}/CANN-aoe-*-linux.aarch64.run --full --install-path=/usr/local/Ascend --quiet
    sudo bash ${LOCAL_DIR}/${WORK_DIR}/${RUN_DIR}/../${version}/CANN-hccl-*-linux.aarch64.run --full --install-path=/usr/local/Ascend --quiet
    sudo bash ${LOCAL_DIR}/${WORK_DIR}/${RUN_DIR}/../${version}/Ascend910B-opp_kernel-*.run --full --install-path=/usr/local/Ascend --quiet
    ls -lrt ${LOCAL_DIR}/${WORK_DIR}/${RUN_DIR}/../${version}
    source /usr/local/Ascend/latest/bin/setenv.bash
    LIB_PATH=/usr/local/Ascend/latest/fwkacllib/lib64
fi

echo "-------------------start install new mindspore--------------------------"
pip install ${LOCAL_DIR}/${WORK_DIR}/${RUN_DIR}/${mindspore_file}
echo "-------------------install te topi and hccl--------------------------"
pip install $LIB_PATH/te-*.whl --force-reinstall --user
pip install $LIB_PATH/hccl-*.whl --force-reinstall --user

pip install mindpet==1.0.3
pip install opencv-python==4.8.1.78
pip install opencv-python-headless==4.8.1.78

npu-smi info

unset RANK_TABLE_FILE

dryrun=0
