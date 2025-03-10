"""
Usage:
python tools/plot.py --input path/to/exp1/result.log path/to/exp2/result.log  \
    --smooth --alpha 0.001 --y_max 0.6 --output loss_cmp.png
"""


import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot(inp, output, smooth=False, alpha=0.01, interval=1, duration=-1, linewidth=1, y_max=None):
    num_curve = len(inp)
    plt.figure()
    plt.title("loss")
    f, ax = plt.subplots(figsize=(14, 6))
    if y_max is not None:
        ax.set_ylim([0, y_max])

    for i in range(num_curve):
        log_path = inp[i]

        df = pd.read_csv(log_path, sep="\t")
        if smooth:
            print("curve soomthing enabled with alpha=", alpha)
            loss = df["loss"].ewm(alpha=alpha).mean().values
        else:
            loss = df["loss"].values
        step = df["step"].values

        ax.plot(step[:duration:interval], loss[:duration:interval], label=f"loss_{log_path}", linewidth=linewidth)

    ax.set_xlabel("steps")
    ax.set_ylabel("loss")
    ax.legend()
    ax.grid()

    plt.savefig(output)
    print("Figure saved in ", output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str, nargs="+", default=None, help="list of path to result log")
    parser.add_argument(
        "--output", "-o", type=str, default="loss_curve.png", help="target file path to save the output loss curve"
    )
    parser.add_argument(
        "--smooth", action="store_true", help="smooth curve by exponential weighted (ema). default: False"
    )
    parser.add_argument(
        "--alpha", default=0.01, type=float, help="smooth factor alpha, the smaller this value, the smoother the curve"
    )
    parser.add_argument("--linewidth", default=1.0, type=float, help="curve line width")
    parser.add_argument("--y_max", default=None, type=float, help="y max value")
    args = parser.parse_args()

    plot(args.input, args.output, args.smooth, args.alpha, linewidth=args.linewidth, y_max=args.y_max)
