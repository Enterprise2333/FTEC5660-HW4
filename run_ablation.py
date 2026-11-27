"""Run one configuration or the full ablation using the original HW4 runner."""
import argparse
import subprocess
import sys
from functools import partial
from pathlib import Path

import hw4


VARIANTS = ("B0", "V1", "V2", "V3", "V4", "V5", "V6")
EARLY_VARIANTS = VARIANTS[:3]
INDEX_OPTIONS = ("original", "company", "normalized", "company-normalized")


def run_variant(variant, data_folder, output, index_option="original", evidence_rank=False):
    mode = VARIANTS.index(variant)
    hw4.build_index = partial(
        hw4.build_index,
        index_company=index_option in ("company", "company-normalized"),
        normalize_tokens=index_option in ("normalized", "company-normalized"),
    )
    if variant != VARIANTS[-1]:
        hw4.chunk_document = lambda name, html: hw4._chunks(
            name, html, labelled_tables=mode >= 1, section_chunks=mode >= 2, register=True,
        )
        parse_query = hw4.parse_query

        def retrieve(index, question):
            evidence_query = parse_query(question) if evidence_rank else None
            return hw4._retrieve(index, question, mode=mode, evidence_query=evidence_query)

        hw4.retrieve = retrieve
    if mode < 3:
        hw4.parse_query = lambda question: {}
    if mode == 0:
        hw4.table_to_text = lambda html: hw4.BeautifulSoup(html, "lxml").get_text("\n", strip=True)
    sys.argv = [sys.argv[0], "--data-folder", str(data_folder), "--output", str(output)]
    hw4.main()


def result_name(variant, index_option):
    if index_option == "original":
        return variant
    return f"{variant}-{index_option}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-folder", type=Path, default=Path("public_test"))
    parser.add_argument("--output-dir", type=Path, default=Path("ablation"))
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--index", choices=INDEX_OPTIONS, default="original",
                        help="BM25 index option for the early variants")
    parser.add_argument("--evidence-rank", action="store_true",
                        help="test V2 with metric and period ranking for the queried companies")
    args = parser.parse_args()
    if args.evidence_rank and (args.variant != "V2" or args.index not in ("company", "company-normalized")):
        parser.error("--evidence-rank requires --variant V2 and a company index")
    variants = VARIANTS if args.index == "original" else EARLY_VARIANTS
    if args.variant and args.variant not in variants:
        parser.error("experimental indexes are available for B0, V1 and V2")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.variant:
        name = result_name(args.variant, args.index)
        if args.evidence_rank:
            name += "-ranked"
        run_variant(args.variant, args.data_folder, args.output_dir / f"{name}.csv",
                    args.index, args.evidence_rank)
        return
    for variant in variants:
        name = result_name(variant, args.index)
        print(f"Running {name}", flush=True)
        with (args.output_dir / f"{name}.log").open("w", encoding="utf-8") as log:
            subprocess.run(
                [sys.executable, "-X", "utf8", __file__, "--data-folder", str(args.data_folder),
                 "--output-dir", str(args.output_dir), "--variant", variant, "--index", args.index],
                stdout=log, stderr=subprocess.STDOUT, check=True, timeout=hw4.TIME_LIMIT_SECONDS,
            )


if __name__ == "__main__":
    main()
