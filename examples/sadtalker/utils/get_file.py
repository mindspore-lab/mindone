import os


def GetDirsFromThisRootDir(dir, ext=None):
    allfiles = []
    needExtFilter = ext != None
    for root, dirs, files in os.walk(dir):
        if len(files) > 0:
            allfiles.append(root)
    return allfiles


def GetFileFromThisRootDir(dir, ext=None):
    allfiles = []
    needExtFilter = ext != None
    for root, dirs, files in os.walk(dir):
        for filespath in files:
            filepath = os.path.join(root, filespath)
            extension = os.path.splitext(filepath)[1][1:]
            if needExtFilter and extension in ext:
                allfiles.append(filepath)
            elif not needExtFilter:
                allfiles.append(filepath)
    return allfiles


def get_img_paths(path, ext=None):
    if os.path.splitext(path)[-1] == ".txt":
        with open(path, "r") as f:
            img_paths = f.read().splitlines()
    else:
        img_paths = GetFileFromThisRootDir(path, ext=ext)
    return img_paths


def get_img_dirs(path, ext=None):
    img_dirs = GetDirsFromThisRootDir(path, ext=ext)
    return img_dirs
