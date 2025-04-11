import os
from typing import List, Optional

import pandas as pd
import plotly.express as px
from jsonargparse import ArgumentParser
from jsonargparse.typing import Path_fc, Path_fr


def plot(log_files: List[Path_fr], out_path: Optional[Path_fc] = None, alpha: float = 0.01) -> None:
    """
    Generate an interactive plot from log files.
    When multiple log files are provided, the plots are combined into a single figure with different prefixes.

    Args:
      log_files: List of paths to log files containing loss information.
                 Each file should have a 'step' column and at least one data column.
      out_path: Path where the output HTML plot will be saved.
                If not provided, saves to 'plot.html' in the directory of the first log file.
      alpha: Smoothing factor for exponential weighted moving average applied to the 'loss' column.
             Values closer to 1 give more weight to recent data points. Default: 0.01.

    Returns:
      None: The function saves the plot as an HTML file and prints the save location.
    """
    if alpha is not None:
        print(f"Curve smoothing enabled with {alpha=}")

    dfs = []
    for log_file in log_files:
        df = pd.read_table(log_file)
        df = df.rename(columns=lambda x: x.strip())  # remove padding whitespaces
        df = df.set_index("step")  # set `step` as an index column
        if alpha is not None:
            df["loss"] = df["loss"].ewm(alpha=alpha).mean()
        dfs.append(df.loc[:, ~df.columns.str.contains("time", case=False)])  # drop the `time` column

    if len(dfs) > 1:  # merge multiple logs into a single dataframe by adding a unique prefix for each table
        dfs = [df.add_prefix(f"{i}: ") for i, df in enumerate(dfs)]
        dfs = pd.concat(dfs, axis=1)
        title = "<br>".join([f"{i}: {os.path.dirname(log_file.relative)}" for i, log_file in enumerate(log_files)])
    else:
        dfs = dfs[0]
        title = os.path.dirname(log_files[0].relative)

    fig = px.line(dfs, title=title).update_traces(connectgaps=True)
    fig.update_xaxes(gridcolor="lightgrey", fixedrange=True)  # prevent zooming on X-axis
    fig.update_yaxes(gridcolor="lightgrey", zerolinecolor="lightgrey")
    fig.update_layout(plot_bgcolor="white", dragmode="pan")

    out_path = os.path.join(os.path.dirname(log_files[0]), "plot.html") if out_path is None else out_path.absolute
    fig.write_html(out_path, config={"scrollZoom": True})
    print(f"Plot saved to {out_path}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Generate an interactive plot from log files.")
    parser.add_argument(
        "log_files", nargs="+", type=Path_fr, help="List of paths to log files containing loss information."
    )
    parser.add_function_arguments(plot, skip={"log_files"}, as_group=False)
    args = parser.parse_args()
    plot(**args)  # noqa
