"""Read-only audit of frozen evaluator score ordering; DOES NOT rescore or fix AP."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.cafe_eval import read_text_file, make_groups


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--output', required=True)
    cli = parser.parse_args()
    out = Path(cli.output); out.mkdir(parents=True, exist_ok=False)
    (out/'command.txt').write_text(shlex.join([sys.executable]+sys.argv)+'\n')
    results = dict(metric_values_recomputed=False, evaluator_modified=False, runs={})
    for name, root in (('A', cli.baseline), ('B', cli.candidate)):
        prediction = Path(root)/'prediction_export_val.txt'
        with prediction.open() as stream:
            boxes, ids, labels, _, scores = read_text_file(stream, 'gt_base', 'pred')
        groups, activities, group_scores = make_groups(boxes, ids, labels, scores)
        rows = []
        for c in range(1, 7):
            values = [(clip, group, group_scores[clip][0][group]) for clip in groups for group in groups[clip][0]
                      if activities[clip][0][group] == {c} and len(groups[clip][0][group]) >= 2]
            if not all(len(row[2]) == 1 for row in values):
                raise ValueError('inconsistent confidence within group; separate investigation required')
            current = sorted(values, key=lambda row:row[2], reverse=True)
            numeric = sorted(values, key=lambda row:next(iter(row[2])), reverse=True)
            rows.append(dict(activity=c, predicted_groups=len(values), all_score_containers_singleton_sets=True,
                             current_sort_equals_export_iteration_order=current == values,
                             current_sort_equals_numeric_score_descending=current == numeric,
                             first_three_current=[dict(clip=r[0], slot=r[1], score=next(iter(r[2]))) for r in current[:3]]))
        results['runs'][name] = dict(path=str(prediction.resolve()), prediction_hash=sha(prediction), classes=rows)
    results['interpretation'] = ('Frozen evaluator sorts singleton sets rather than numeric scores. Different singleton sets '
                                 'are not numerically ordered; observed order follows export iteration. Existing A/B AP '
                                 'values remain unchanged; any evaluator correction requires separate, cohort-wide replay.')
    for source in (__file__, str(ROOT/'evaluation/cafe_eval.py')):
        path = Path(source); shutil.copyfile(path, out/path.name)
    results['evaluator_sha256'] = sha(ROOT/'evaluation/cafe_eval.py')
    (out/'ranking_audit.json').write_text(json.dumps(results, indent=2)+'\n')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
