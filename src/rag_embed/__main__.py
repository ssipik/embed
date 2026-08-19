"""Dev/ops entrypoint: ``python -m rag_embed``.

Thin argparse wrapper over :func:`rag_embed.embed.run_embed_batch`
— the library function remains the API that Airflow tasks import directly;
keep all stage logic out of this module.
"""

from __future__ import annotations

import argparse
import logging
import sys

from dotenv import load_dotenv

from rag_embed.embed import run_embed_batch


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m rag_embed",
        description="EMBED stage: embed one parsed batch's staged "
        "nodes via TEI, upsert points to Qdrant, mark the ledger.",
    )
    parser.add_argument(
        "--batch-id",
        required=True,
        type=int,
        action="append",
        dest="batch_ids",
        help="batch to process (a rag_plan batch id); repeatable for dev runs "
        "— in the DAG each mapped task gets exactly one",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    for batch_id in args.batch_ids:
        try:
            report = run_embed_batch(batch_id)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print(f"batch {batch_id}: {report.counts}")
    return 0


if __name__ == "__main__":
    # Only needed when running in development. In production the env variables are provided by Vault.
    load_dotenv()
    raise SystemExit(main())
