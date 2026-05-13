"""CLI entry-point: python -m panorama_poc.cli --input capture/ --output output/"""
import argparse
import logging
import sys

from panorama_poc.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ghost-free panorama stitching with IMU-guided frame selection."
    )
    parser.add_argument("--input",   required=True, help="Path to capture/ directory")
    parser.add_argument("--output",  required=True, help="Path to output/ directory")
    parser.add_argument("--verbose", action="store_true", help="Enable DEBUG logging")
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    run_pipeline(args.input, args.output)


if __name__ == "__main__":
    main()
