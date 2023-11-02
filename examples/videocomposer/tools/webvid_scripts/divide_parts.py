import os

meta_filename = "results_10M_train.csv"  # the splited metadata filename

root_dir = "/data1/webvid-10m/metadata/"
meta_filepath = os.path.join(root_dir, meta_filename)
output_dir = f"/data1/webvid-10m/metadata/{os.path.splitext(meta_filename)[0]}"
os.makedirs(output_dir, exist_ok=True)

linenum = int(os.popen(f"wc -l {meta_filepath}").read().split()[0])
stride = 500000
i = 2
partid = 0

print(f"Spliting the metadata file: {meta_filepath}. Total number of lines: {linenum}")

while i <= linenum:
    output_filename = os.path.join(output_dir, f"part{partid}.csv")
    os.system(f"head -n 1 {meta_filepath} > {output_filename}")
    os.system(f"sed -n '{i}, {i+stride}p' {meta_filepath} >> {output_filename}")
    i += stride + 1
    partid += 1

assert (
    partid == linenum // stride + 1
), f"The metadata file shoule be splited into {linenum//stride + 1} parts, but got {partid} parts."
print(f"Finished! The metadata file is splited into {partid} parts, saved in '{output_dir}'.")
