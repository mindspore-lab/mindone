import argparse
import math

import pandas as pd

# result_path = 'outputs/analyze_os1.2_stage2_vcg200_ms231/merged_result.log'


def analyze(result_path, save_path):
    warmup_steps = 50
    max_step_time = 100  # step time larger than this value will be dropped, considering checkpoint saving

    data = pd.read_csv(result_path, sep="\t")

    # filter warmup stage
    data = data.iloc[warmup_steps - 1 :]

    # filter out outliers
    data = data[data["train_time(s)"] < max_step_time]
    global_avg_step_time = data["train_time(s)"].mean()
    num_steps = data.shape[0]

    res = data.groupby("shape")["train_time(s)"].agg(["mean", "std", "count"]).reset_index()

    res.columns = ["shape", "mean_step_time", "std_step_time", "occurence"]

    res["std_step_time"].fillna(0, inplace=True)

    res_sorted = res.sort_values(by="mean_step_time", ascending=False)

    percent_col = []
    bs_col = []
    sug_bs_col = []
    rnd_bs_col = []
    tot_occur = res_sorted["occurence"].sum()
    # for shape_str in res_sorted['shape'].tolist():
    for idx, row in res_sorted.iterrows():
        percent_col.append(row["occurence"] / tot_occur)
        bs = int(row["shape"].split(",")[0][1:])
        bs_col.append(bs)

        suggest_bs = (global_avg_step_time / row["mean_step_time"]) * bs
        rounded_bs = max(1, math.floor(suggest_bs))
        sug_bs_col.append(suggest_bs)
        rnd_bs_col.append(rounded_bs)
    res_sorted["occ_percent"] = percent_col
    res_sorted["bs"] = bs_col
    res_sorted["suggested_bs"] = sug_bs_col
    res_sorted["rounded_bs"] = rnd_bs_col

    res_sorted.to_csv(save_path, index=False)

    print(res_sorted)
    print(f"\nAverage step time(s) (in {num_steps} steps starting from step {warmup_steps}): ", global_avg_step_time)
    print("Analysis csv saved in", save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str, default=None, help="path to result log")
    parser.add_argument("--output", "-o", type=str, default="shape_step_time.csv", help="path to save analysis output")

    args = parser.parse_args()
    analyze(args.input, args.output)
