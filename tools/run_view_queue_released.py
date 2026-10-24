#!/usr/bin/env python3
"""Release the place-SAM dependency after an explicit request to start B0 now.

All training, cache, GPU, and completion checks come from the frozen queue.
The previous idle queue must be stopped and archived before this is launched.
"""
import argparse
import importlib.util
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--release', required=True, type=Path)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    release = json.loads(args.release.read_text())
    if (release.get('action') != 'start_view_b0_now'
            or release.get('plan') != str(args.plan.resolve())
            or release.get('user_request') != '现在可以开始做view B0了。'):
        raise ValueError('explicit scheduling release does not match this queue')
    frozen = Path(plan['snapshot_root']) / 'tools/queue_view_dp4.py'
    spec = importlib.util.spec_from_file_location('frozen_view_queue', str(frozen))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class ReleasedQueue(module.Queue):
        def wait_dependency(self):
            self.status('place_dependency_released_by_user',
                        release_record=str(args.release.resolve()),
                        message='Start view B0 now; keep the existing place SAM running')

    queue = ReleasedQueue(args.plan)
    queue.verify_frozen()
    queue.audit_contract()
    for variant in module.VARIANTS:
        if (queue.base / variant / 'training.status').exists():
            raise RuntimeError('view training already has a run status: ' + variant)
        entry = plan['runs'][variant]
        tag = json.loads(Path(entry['resolved_args']).read_text())['experiment_tag']
        if any(('_[' + tag + ']_') in p.name for p in (queue.root / 'result').iterdir()):
            raise RuntimeError('view experiment already has output: ' + tag)
    if args.check:
        print('Frozen queue/configuration and unstarted view experiments: passed', flush=True)
    else:
        queue.run()


if __name__ == '__main__':
    main()
