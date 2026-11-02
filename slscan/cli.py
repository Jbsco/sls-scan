"""The command line: one command per pipeline stage."""

import argparse
import sys

import numpy as np

from . import patterns


def _write_stack(stack: np.ndarray, prefix: str) -> None:
    import cv2

    for i, image in enumerate(stack):
        cv2.imwrite(f"{prefix}{i:03d}.png", image)


def cmd_patterns(args: argparse.Namespace) -> int:
    gray = patterns.gray_code_images(args.width, args.height)
    _write_stack(gray, f"{args.out}/gray_")
    phase = patterns.phase_shift_images(args.width, args.height, args.pitch, args.steps)
    _write_stack(phase, f"{args.out}/phase_")
    print(f"{gray.shape[0]} Gray code images and {phase.shape[0]} phase images written to {args.out}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="slscan", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("patterns", help="write the projector pattern stack as PNG files")
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--pitch", type=int, default=32, help="phase shift pitch in projector pixels")
    p.add_argument("--steps", type=int, default=4, help="phase shift steps")
    p.add_argument("--out", default="data/patterns", help="output directory, created by the caller")
    p.set_defaults(func=cmd_patterns)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
