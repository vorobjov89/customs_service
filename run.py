import argparse
from pathlib import Path

from predict import main


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to input data directory",
    )

    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Path to output directory",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    main(
        data_dir=args.data,
        out_dir=args.out,
    )
    