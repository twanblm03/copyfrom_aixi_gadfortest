"""Check the allocated GPU2/3 pair before starting a crowding run."""
import os
import subprocess


ALLOCATION = (
    'GPU-f13b1d06-5ee1-64b8-bc92-4e5c28d72e7c',
    'GPU-bc83c5ed-4690-7c23-37a5-8f78022962fa',
)


def require_idle_allocation():
    visible = tuple(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(','))
    if visible != ALLOCATION:
        raise RuntimeError('explicit physical GPU2/3 UUID allocation required; GPU0/1 forbidden')
    output = subprocess.check_output([
        'nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name',
        '--format=csv,noheader,nounits',
    ]).decode()
    foreign = []
    for line in output.splitlines():
        fields = [part.strip() for part in line.split(',')]
        if fields[0] in visible and int(fields[1]) != os.getpid():
            foreign.append(line)
    if foreign:
        raise RuntimeError('allocated GPUs have other compute processes; no launch: ' + str(foreign))
    return output
