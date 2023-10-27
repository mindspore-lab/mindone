#!/usr/bin/env python
import argparse
import os

from deploy.utils.export import LiteConverter
from deploy.utils.logger import setup_logger


def main():
    _logger = setup_logger()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--names", nargs="+", help="convert the model from MindIR to Mindspore Lite MindIR with the given name"
    )
    parser.add_argument("--root", default="models/mindir", help="Convert all MindIR in root folder")
    args = parser.parse_args()

    lite_converter = LiteConverter(target="ascend")

    if args.names:
        for name in args.names:
            lite_converter(name)
    elif args.root:
        names = sorted(os.listdir(args.root))
        names = [x.replace(".mindir", "") for x in names if x.endswith(".mindir")]
        names = [x.replace("_graph", "") for x in names]
        _logger.info(f"converting {names}")
        for name in names:
            lite_converter(name)


if __name__ == "__main__":
    main()
