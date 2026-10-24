"""CPU checks for queue gates; no GPU workload is launched."""
import importlib.util
from pathlib import Path
import unittest

PATH = Path(__file__).resolve().parents[2] / 'tools/queue_view_dp4.py'
SPEC = importlib.util.spec_from_file_location('view_queue', PATH)
queue = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(queue)


class QueueGates(unittest.TestCase):
    def test_running_dependency_waits(self):
        self.assertEqual(queue.dependency_state('training_started date', True), 'waiting_place_sam')
        self.assertEqual(queue.dependency_state('training_complete date', True), 'waiting_dependency_exit')
        self.assertEqual(queue.dependency_state('training_complete date', False), 'ready')

    def test_failed_or_disappeared_dependency_blocks(self):
        for text, alive in [('training_failed code=1 date', False),
                            ('training_interrupted date', True),
                            ('training_started date', False), ('', False)]:
            with self.subTest(text=text), self.assertRaises(RuntimeError):
                queue.dependency_state(text, alive)

    def test_all_four_gpus_must_have_memory(self):
        ids = ['GPU-a', 'GPU-b', 'GPU-c', 'GPU-d']
        text = '\n'.join('%d, %s, %d' % (i, name, 26000 if i != 2 else 24999)
                         for i, name in enumerate(ids))
        self.assertFalse(queue.parse_gpus(text, ids, 25000)[0])
        self.assertTrue(queue.parse_gpus(text.replace('24999', '25000'), ids, 25000)[0])

    def test_device_mapping_or_count_change_is_rejected(self):
        ids = ['GPU-a', 'GPU-b', 'GPU-c', 'GPU-d']
        text = '\n'.join('%d, %s, 30000' % (i, name) for i, name in enumerate(ids))
        for changed in (text.replace('GPU-b', 'GPU-other'), '\n'.join(text.splitlines()[:3]),
                        text.replace('1, GPU-b', '2, GPU-b')):
            with self.subTest(text=changed), self.assertRaises(RuntimeError):
                queue.parse_gpus(changed, ids, 25000)


if __name__ == '__main__':
    unittest.main()
