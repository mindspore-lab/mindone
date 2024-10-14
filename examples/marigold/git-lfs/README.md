# Install git-lfs in this directory

You may follow command below to install the specific version of git-lfs in this directory, please check the version make sure it's OK.

```bash
tar -zxvf git-lfs-linux-arm64-v3.5.1.tar.gz
cd git-lfs-3.5.1
mkdir -p YOUR_PATH_TO_INSTALL
export PREFIX=YOUR_PATH_TO_INSTALL
./install.sh
export PATH=$PATH:YOUR_PATH_TO_INSTALL/bin
```

Please notice that `PATH` should add **/bin** after your install path.
