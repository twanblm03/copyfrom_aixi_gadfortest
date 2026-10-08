#!/usr/bin/env python3
"""Summarize complete multi-seed Cafe development-baseline runs."""

import argparse
import json
import math
import re
import statistics
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


MAP_METRICS = ("group_mAP_1.0", "group_mAP_0.5")
REQUIRED_METRICS = MAP_METRICS + ("outlier_mIoU",)
TIMESTAMP_RE = re.compile(r"\[(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})\]$")


@dataclass(frozen=True)
class RunRecord:
    seed: int
    path: Path
    sort_key: Tuple[int, datetime, int, str]
    selection_metric: str
    selection_epoch: int
    map_10_at_selection: float
    map_05_at_selection: float
    outlier_at_selection: float


def warn(message: str) -> None:
    print("warning: {}".format(message), file=sys.stderr)


def load_json_object(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("cannot read {}: {}".format(path.name, error)) from error
    if not isinstance(value, dict):
        raise ValueError("{} must contain a JSON object".format(path.name))
    return value


def integer(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError("{} must be an integer >= {}".format(name, minimum))
    return value


def finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("{} must be numeric".format(name))
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("{} must be finite".format(name))
    return result


def indexed_history(entries: Any, name: str) -> Dict[int, Dict[str, Any]]:
    if not isinstance(entries, list) or not entries:
        raise ValueError("history.{} must be a non-empty list".format(name))
    result: Dict[int, Dict[str, Any]] = {}
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError("history.{}[{}] must be an object".format(name, index))
        epoch = integer(entry.get("epoch"), "history.{}[{}].epoch".format(name, index), 1)
        if epoch in result:
            raise ValueError("history.{} contains duplicate epoch {}".format(name, epoch))
        result[epoch] = entry
    return result


def best_entry(best: Dict[str, Any], metric: str) -> Tuple[int, float]:
    entry = best.get(metric)
    if not isinstance(entry, dict):
        raise ValueError("best.{} is missing or invalid".format(metric))
    epoch = integer(entry.get("epoch"), "best.{}.epoch".format(metric), 1)
    value = finite_number(entry.get("value"), "best.{}.value".format(metric))
    return epoch, value


def run_sort_key(run_dir: Path, summary_path: Path) -> Tuple[int, datetime, int, str]:
    match = TIMESTAMP_RE.search(run_dir.name)
    timestamp = datetime.min
    has_timestamp = 0
    if match:
        try:
            timestamp = datetime.strptime(match.group(1), "%Y-%m-%d_%H-%M-%S")
            has_timestamp = 1
        except ValueError:
            pass
    try:
        modified_ns = summary_path.stat().st_mtime_ns
    except OSError:
        modified_ns = 0
    return has_timestamp, timestamp, modified_ns, str(run_dir)


def validate_complete_run(
    run_dir: Path,
    args_data: Dict[str, Any],
    summary_path: Path,
) -> RunRecord:
    summary = load_json_object(summary_path)
    summary_args = summary.get("args")
    if not isinstance(summary_args, dict):
        raise ValueError("summary.json args is missing or invalid")

    identity_keys = (
        "protocol",
        "experiment_tag",
        "split",
        "random_seed",
        "epochs",
        "test_freq",
        "selection_metric",
    )
    for key in identity_keys:
        if summary_args.get(key) != args_data.get(key):
            raise ValueError("args.json and summary.json disagree on '{}'".format(key))

    if summary_args.get("protocol") != "development":
        raise ValueError("summary protocol is not development")

    epochs = integer(summary_args.get("epochs"), "args.epochs", 1)
    test_freq = integer(summary_args.get("test_freq"), "args.test_freq", 1)
    seed = integer(summary_args.get("random_seed"), "args.random_seed")
    selection_metric = summary_args.get("selection_metric")
    if selection_metric not in ("group_mAP_1.0", "group_mAP_0.5", "loss"):
        raise ValueError("unsupported selection metric: {!r}".format(selection_metric))

    history = summary.get("history")
    if not isinstance(history, dict):
        raise ValueError("summary.json history is missing or invalid")
    train_by_epoch = indexed_history(history.get("train"), "train")
    val_by_epoch = indexed_history(history.get("val"), "val")

    expected_train_epochs = set(range(1, epochs + 1))
    actual_train_epochs = set(train_by_epoch)
    if actual_train_epochs != expected_train_epochs:
        missing = sorted(expected_train_epochs - actual_train_epochs)
        extra = sorted(actual_train_epochs - expected_train_epochs)
        raise ValueError(
            "incomplete train history (missing={}, extra={})".format(missing, extra)
        )

    expected_val_epochs = set(range(test_freq, epochs + 1, test_freq))
    actual_val_epochs = set(val_by_epoch)
    if not expected_val_epochs or actual_val_epochs != expected_val_epochs:
        missing = sorted(expected_val_epochs - actual_val_epochs)
        extra = sorted(actual_val_epochs - expected_val_epochs)
        raise ValueError(
            "incomplete validation history (missing={}, extra={})".format(missing, extra)
        )

    for epoch, entry in val_by_epoch.items():
        for metric in REQUIRED_METRICS:
            finite_number(entry.get(metric), "history.val[epoch={}].{}".format(epoch, metric))
        finite_number(entry.get("loss"), "history.val[epoch={}].loss".format(epoch))

    best = summary.get("best")
    if not isinstance(best, dict):
        raise ValueError("summary.json best is missing or invalid")

    # Validate the independently tracked best entries, but do not report them:
    # all paper-facing metrics must come from the primary-selected checkpoint.
    for metric in MAP_METRICS:
        metric_epoch, metric_value = best_entry(best, metric)
        if metric_epoch not in val_by_epoch:
            raise ValueError("best.{} epoch {} is absent from validation history".format(
                metric, metric_epoch))
        history_value = finite_number(
            val_by_epoch[metric_epoch].get(metric),
            "history.val[epoch={}].{}".format(metric_epoch, metric),
        )
        observed_max = max(
            finite_number(entry.get(metric), "history.val.{}".format(metric))
            for entry in val_by_epoch.values()
        )
        if not math.isclose(metric_value, history_value, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("best.{} disagrees with its validation epoch".format(metric))
        if not math.isclose(metric_value, observed_max, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("best.{} is not the maximum validation value".format(metric))

    selection_epoch, selection_value = best_entry(best, selection_metric)
    selected_entry = val_by_epoch.get(selection_epoch)
    if selected_entry is None:
        raise ValueError("primary selection epoch {} is absent from validation history".format(
            selection_epoch))
    selected_history_value = finite_number(
        selected_entry.get(selection_metric),
        "history.val[epoch={}].{}".format(selection_epoch, selection_metric),
    )
    if not math.isclose(selection_value, selected_history_value, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("primary best value disagrees with its validation epoch")
    selection_values = [
        finite_number(entry.get(selection_metric), "history.val.{}".format(selection_metric))
        for entry in val_by_epoch.values()
    ]
    expected_selection_value = (
        min(selection_values) if selection_metric == "loss" else max(selection_values)
    )
    if not math.isclose(
        selection_value, expected_selection_value, rel_tol=1e-12, abs_tol=1e-12
    ):
        raise ValueError("primary best value is not optimal in validation history")
    outlier_value = finite_number(
        selected_entry.get("outlier_mIoU"),
        "history.val[epoch={}].outlier_mIoU".format(selection_epoch),
    )
    map_10_value = finite_number(
        selected_entry.get("group_mAP_1.0"),
        "history.val[epoch={}].group_mAP_1.0".format(selection_epoch),
    )
    map_05_value = finite_number(
        selected_entry.get("group_mAP_0.5"),
        "history.val[epoch={}].group_mAP_0.5".format(selection_epoch),
    )

    return RunRecord(
        seed=seed,
        path=run_dir,
        sort_key=run_sort_key(run_dir, summary_path),
        selection_metric=selection_metric,
        selection_epoch=selection_epoch,
        map_10_at_selection=map_10_value,
        map_05_at_selection=map_05_value,
        outlier_at_selection=outlier_value,
    )


def parse_seeds(value: str) -> List[int]:
    fields = [field for field in re.split(r"[\s,]+", value.strip()) if field]
    if not fields:
        raise argparse.ArgumentTypeError("at least one seed is required")
    seeds: List[int] = []
    for field in fields:
        if not field.isdigit():
            raise argparse.ArgumentTypeError("invalid seed {!r}".format(field))
        seed = int(field)
        if seed in seeds:
            raise argparse.ArgumentTypeError("duplicate seed {}".format(seed))
        seeds.append(seed)
    return seeds


def discover_runs(
    result_root: Path,
    split: str,
    tag_prefix: str,
    expected_seeds: Sequence[int],
) -> Tuple[Dict[int, RunRecord], List[int]]:
    expected_set = set(expected_seeds)
    candidates: Dict[int, List[RunRecord]] = {seed: [] for seed in expected_seeds}

    for args_path in sorted(result_root.rglob("args.json")):
        run_dir = args_path.parent
        try:
            args_data = load_json_object(args_path)
        except ValueError as error:
            warn("ignoring {}: {}".format(run_dir, error))
            continue

        if args_data.get("protocol") != "development":
            continue
        if args_data.get("split") != split:
            continue
        tag = args_data.get("experiment_tag")
        if not isinstance(tag, str) or not tag.startswith(tag_prefix):
            continue

        try:
            seed = integer(args_data.get("random_seed"), "args.random_seed")
        except ValueError as error:
            warn("ignoring matching run {}: {}".format(run_dir, error))
            continue
        if seed not in expected_set:
            warn("ignoring matching run {} for unrequested seed {}".format(run_dir, seed))
            continue

        summary_path = run_dir / "summary.json"
        if not summary_path.is_file():
            warn("ignoring incomplete run {}: summary.json is missing".format(run_dir))
            continue
        try:
            record = validate_complete_run(run_dir, args_data, summary_path)
        except ValueError as error:
            warn("ignoring incomplete run {}: {}".format(run_dir, error))
            continue
        candidates[seed].append(record)

    selected: Dict[int, RunRecord] = {}
    missing: List[int] = []
    for seed in expected_seeds:
        seed_runs = candidates[seed]
        if not seed_runs:
            warn("no complete matching run found for seed {}".format(seed))
            missing.append(seed)
            continue
        seed_runs.sort(key=lambda record: record.sort_key)
        selected[seed] = seed_runs[-1]
        if len(seed_runs) > 1:
            ignored = ", ".join(str(record.path) for record in seed_runs[:-1])
            warn(
                "seed {} has {} complete matching runs; using latest {}; older run(s): {}".format(
                    seed, len(seed_runs), seed_runs[-1].path, ignored
                )
            )
    return selected, missing


def relative_display(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def metric_summary(values: Iterable[float]) -> Tuple[float, float]:
    materialized = list(values)
    return statistics.fmean(materialized), statistics.pstdev(materialized)


def shared_selection_metric(records: Sequence[RunRecord]) -> Optional[str]:
    metrics = sorted({record.selection_metric for record in records})
    if len(metrics) > 1:
        raise ValueError(
            "selected seeds use different primary selection metrics: {}".format(
                ", ".join(metrics)
            )
        )
    return metrics[0] if metrics else None


def print_report(
    records: Sequence[RunRecord],
    result_root: Path,
    split: str,
    tag_prefix: str,
) -> None:
    selection_metric = shared_selection_metric(records)
    print("Cafe development baseline summary")
    print("  result root: {}".format(result_root))
    print("  split: {}".format(split))
    print("  tag prefix: {}".format(tag_prefix))
    if not records:
        print("  complete runs: 0")
        return

    print("  primary selection metric: {}".format(selection_metric))

    header = (
        "seed",
        "select_metric",
        "select_epoch",
        "mAP@1.0@select",
        "mAP@0.5@select",
        "outlier@select",
        "run",
    )
    rows = [
        (
            str(record.seed),
            record.selection_metric,
            str(record.selection_epoch),
            "{:.4f}".format(record.map_10_at_selection),
            "{:.4f}".format(record.map_05_at_selection),
            "{:.4f}".format(record.outlier_at_selection),
            relative_display(record.path, result_root),
        )
        for record in records
    ]
    widths = [
        max(len(header[index]), *(len(row[index]) for row in rows))
        for index in range(len(header))
    ]
    row_format = "  ".join("{{:<{}}}".format(width) for width in widths)
    print()
    print(row_format.format(*header))
    print(row_format.format(*("-" * width for width in widths)))
    for row in rows:
        print(row_format.format(*row))

    map_10_mean, map_10_std = metric_summary(
        record.map_10_at_selection for record in records
    )
    map_05_mean, map_05_std = metric_summary(
        record.map_05_at_selection for record in records
    )
    outlier_mean, outlier_std = metric_summary(
        record.outlier_at_selection for record in records
    )
    print()
    print("aggregate over {} complete seed(s); std uses ddof=0".format(len(records)))
    print("  group_mAP_1.0 @ primary epoch: {:.4f} +/- {:.4f}".format(
        map_10_mean, map_10_std
    ))
    print("  group_mAP_0.5 @ primary epoch: {:.4f} +/- {:.4f}".format(
        map_05_mean, map_05_std
    ))
    print("  outlier_mIoU @ primary epoch: {:.4f} +/- {:.4f}".format(
        outlier_mean, outlier_std
    ))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize the latest complete Cafe development-baseline run per seed. "
            "Runs must contain consistent args.json and summary.json files; all "
            "reported metrics come from the primary-selected validation epoch."
        )
    )
    parser.add_argument(
        "result_root",
        nargs="?",
        default="result",
        type=Path,
        help="root recursively scanned for args.json (default: result)",
    )
    parser.add_argument("--split", required=True, choices=("place", "view"))
    parser.add_argument(
        "--tag-prefix",
        default=None,
        help="required experiment-tag prefix (default: baseline_val_<split>_seed)",
    )
    parser.add_argument(
        "--seeds",
        default=parse_seeds("1,2,3"),
        type=parse_seeds,
        metavar="LIST",
        help="expected comma/space-separated seeds (default: 1,2,3)",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    options = build_parser().parse_args(argv)
    result_root = options.result_root.expanduser().resolve()
    if not result_root.is_dir():
        print("error: result root is not a directory: {}".format(result_root), file=sys.stderr)
        return 2
    tag_prefix = options.tag_prefix or "baseline_val_{}_seed".format(options.split)
    if not tag_prefix:
        print("error: --tag-prefix cannot be empty", file=sys.stderr)
        return 2

    selected, missing = discover_runs(
        result_root=result_root,
        split=options.split,
        tag_prefix=tag_prefix,
        expected_seeds=options.seeds,
    )
    records = [selected[seed] for seed in options.seeds if seed in selected]
    try:
        shared_selection_metric(records)
    except ValueError as error:
        print("error: {}".format(error), file=sys.stderr)
        return 2
    print_report(records, result_root, options.split, tag_prefix)
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
