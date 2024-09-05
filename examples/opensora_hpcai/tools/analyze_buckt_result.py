import pandas as pd
import argparse

# result_path = 'outputs/analyze_os1.2_stage2_vcg200_ms231/merged_result.log'

def analyze(result_path, save_path):
    warmup_steps = 50
    max_step_time = 100 # step time larger than this value will be dropped, considering checkpoint saving 

    data = pd.read_csv(result_path, sep='\t')

    # filter warmup stage
    data = data.iloc[warmup_steps-1:]

    # filter out outliers
    data = data[data['train_time(s)']<max_step_time]


    res = data.groupby('shape')['train_time(s)'].agg(['mean', 'std', 'count']).reset_index()

    res.columns = ['shape', 'mean_step_time', 'std_step_time', 'occurence']

    res['std_step_time'].fillna(0, inplace=True)

    res_sorted = res.sort_values(by='mean_step_time', ascending=False)

    res_sorted.to_csv(save_path, index=False)

    print(res_sorted)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str, default=None, help="path to result log")
    parser.add_argument(
        "--output", "-o", type=str, default="shape_step_time.csv", help="path to save analysis output"
    )

    args = parser.parse_args()
    analyze(args.input, args.output)
