import shutil


def read_video_list(fn, is_train=False):
    video_list = []
    with open(fn, "r") as f:
        lines = f.read().split("\n")
        for line in lines:
            if line == "":
                continue
            if is_train:
                v_name = line.split(" ")[0].strip()
            else:
                v_name = line.strip()
            video_list.append(v_name)

    return video_list


fn_train = "ucfTrainTestlist/trainlist01.txt"
fn_test = "ucfTrainTestlist/testlist01.txt"

# train
train_list = read_video_list(fn_train, True)
dir_src = "ucf101/fullset/UCF-101/"
dir_des = "ucf101/rec_train/"
for vname in train_list:
    path_src = dir_src + vname
    path_des = dir_des + vname.split("/")[-1]
    shutil.move(path_src, path_des)
    print(f"Moved {path_src} to {path_des}.")

# test
test_list = read_video_list(fn_test, False)
dir_src = "ucf101/fullset/UCF-101/"
dir_des = "ucf101/rec_test/"
for vname in test_list:
    path_src = dir_src + vname
    path_des = dir_des + vname.split("/")[-1]
    shutil.move(path_src, path_des)
    print(f"Moved {path_src} to {path_des}.")
