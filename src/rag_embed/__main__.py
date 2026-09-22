"""Dev/ops entrypoint: ``python -m rag_embed``.

Thin argparse wrapper over :func:`rag_embed.embed.run_embed`
— the library function remains the API that Airflow tasks import directly;
keep all stage logic out of this module.
"""

from __future__ import annotations

import argparse
import logging
import sys

from dotenv import load_dotenv

from rag_embed.embed import run_embed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m rag_embed",
        description="EMBED stage: embed every parsed document of one run's "
        "batches via TEI, write the points to Qdrant, mark the ledger.",
    )
    parser.add_argument(
        "--run-id",
        required=True,
        help="run whose batches to embed (the run_id plan was given)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    try:
        report = run_embed(args.run_id)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(
        f"run {report.run_id}: {report.counts} "
        f"({report.groups} groups, concurrency {report.concurrency})"
    )
    return 0


if __name__ == "__main__":
    # Only needed when running in development. In production the env variables are provided by Vault.
    load_dotenv()
    raise SystemExit(main())
