import pytest

from rag_embed.__main__ import build_parser


def test_run_id_is_required():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_run_id_is_passed_through():
    # Airflow run ids carry ':' and '+', so no int or slug parsing
    args = build_parser().parse_args(["--run-id", "manual__2026-09-15T10:00:00+00:00"])
    assert args.run_id == "manual__2026-09-15T10:00:00+00:00"
