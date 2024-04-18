import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot(inp, output, interval=10, duration=-1):
    # csv_path = 'outputs/stdit_256x256x16_align/2024-04-17T18-25-04/result.log'
    csv_path = inp 

    df = pd.read_csv(csv_path, sep='\t')
    loss = df['loss'].values
    step = df['step'].values
     
    plt.figure()
    plt.title('ms')
    plt.plot(step[:duration:interval], loss[:duration:interval], label="train_loss")
    plt.legend()
    plt.grid()

    plt.savefig(output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=None, help="path to result log")
    parser.add_argument(
        "--output", type=str, default="loss_curve.png", help="target file path to save the output loss curve"
    )
    args = parser.parse_args()

    plot(args.input, args.output)
