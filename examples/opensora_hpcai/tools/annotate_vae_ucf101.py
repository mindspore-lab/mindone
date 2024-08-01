import glob
import math
import os
import random

root_dir = "UCF-101/"
train_ratio = 0.8

all_files = glob.glob(os.path.join(root_dir, "*/*.avi"))
num_samples = len(all_files)
print("Num samples: ", num_samples)

# shuffle
# indices = list(range(num_samples))
random.shuffle(all_files)

# split
num_train = math.ceil(num_samples * train_ratio)
num_test = num_samples - num_train
train_set = sorted(all_files[:num_train])
test_set = sorted(all_files[num_train:])


# save csv
def save_csv(fns, save_path):
    with open(save_path, "w") as fp:
        fp.write("video\n")
        for i, fn in enumerate(fns):
            rel_path = fn.replace(root_dir, "")
            if i != len(fns) - 1:
                fp.write(f"{rel_path}\n")
            else:
                fp.write(f"{rel_path}")


save_csv(train_set, "ucf101_train.csv")
save_csv(test_set, "ucf101_test.csv")
print("Done. csv saved.")
