"""Run one configuration or the full ablation using the original HW4 runner."""
import argparse
import subprocess
import sys
from pathlib import Path

import hw4


VARIANTS = ("B0", "V1", "V2", "V3", "V4", "V5", "V6")


def run_variant(variant, data_folder, output):
    mode = VARIANTS.index(variant)
    if variant != VARIANTS[-1]:
        hw4.chunk_document = lambda name, html: hw4._chunks(
            name, html, labelled_tables=mode >= 1, section_chunks=mode >= 2, register=True,
        )
        hw4.retrieve = lambda index, question: hw4._retrieve(index, question, mode=mode)
    if mode < 3:
        hw4.parse_query = lambda question: {}
    if mode == 0:
        hw4.table_to_text = lambda html: hw4.BeautifulSoup(html, "lxml").get_text("\n", strip=True)
    sys.argv = [sys.argv[0], "--data-folder", str(data_folder), "--output", str(output)]
    hw4.main()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-folder", type=Path, default=Path("public_test"))
    parser.add_argument("--output-dir", type=Path, default=Path("ablation"))
    parser.add_argument("--variant", choices=VARIANTS)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.variant:
        run_variant(args.variant, args.data_folder, args.output_dir / f"{args.variant}.csv")
        return
    for variant in VARIANTS:
        print(f"Running {variant}", flush=True)
        with (args.output_dir / f"{variant}.log").open("w", encoding="utf-8") as log:
            subprocess.run(
                [sys.executable, "-X", "utf8", __file__, "--data-folder", str(args.data_folder),
                 "--output-dir", str(args.output_dir), "--variant", variant],
                stdout=log, stderr=subprocess.STDOUT, check=True, timeout=hw4.TIME_LIMIT_SECONDS,
            )


if __name__ == "__main__":
    main()
