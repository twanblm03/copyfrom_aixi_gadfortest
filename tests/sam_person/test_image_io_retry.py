import errno
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock
from PIL import Image

PATH = Path(__file__).resolve().parents[2] / 'util/image_io_retry.py'
SPEC = importlib.util.spec_from_file_location('image_retry', PATH)
retry = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(retry)


class ImageRetry(unittest.TestCase):
    def test_decoded_pixels_are_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'image.jpg'
            Image.new('RGB', (17, 11), (23, 87, 149)).save(path)
            with Image.open(path) as original:
                original.load()
                expected = original.tobytes()
            with retry.load_image_with_retry(path) as recovered:
                self.assertEqual(recovered.tobytes(), expected)
                self.assertEqual((recovered.mode, recovered.size), ('RGB', (17, 11)))

    def test_open_failure_retries_exact_same_image(self):
        image = Mock()
        opener = Mock(side_effect=[OSError(errno.EHOSTDOWN, 'Host is down'), image])
        delays = []
        self.assertIs(retry.load_image_with_retry('/same.jpg', sleep=delays.append, opener=opener), image)
        self.assertEqual([args.args[0] for args in opener.call_args_list], ['/same.jpg', '/same.jpg'])
        self.assertEqual(delays, [5])

    def test_lazy_decode_failure_closes_and_retries(self):
        first, second = Mock(), Mock()
        first.load.side_effect = OSError(errno.EIO, 'read failed')
        opener = Mock(side_effect=[first, second])
        self.assertIs(retry.load_image_with_retry('/same.jpg', sleep=lambda _: None, opener=opener), second)
        first.close.assert_called_once()

    def test_missing_or_corrupt_image_is_not_substituted(self):
        for error in [FileNotFoundError(errno.ENOENT, 'missing'), OSError('invalid image')]:
            opener = Mock(side_effect=error)
            with self.subTest(error=error), self.assertRaises(OSError):
                retry.load_image_with_retry('/same.jpg', sleep=lambda _: None, opener=opener)
            self.assertEqual(opener.call_count, 1)

    def test_retry_is_bounded(self):
        opener = Mock(side_effect=OSError(errno.EHOSTDOWN, 'Host is down'))
        delays = []
        with self.assertRaises(OSError):
            retry.load_image_with_retry('/same.jpg', attempts=3, sleep=delays.append, opener=opener)
        self.assertEqual(opener.call_count, 3)
        self.assertEqual(delays, [5, 10])


if __name__ == '__main__':
    unittest.main()
