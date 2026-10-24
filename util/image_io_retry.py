"""Decode the same image again on a transient filesystem/network error."""
import errno
import time
from PIL import Image

TRANSIENT_ERRNOS = {getattr(errno, key) for key in (
    'EHOSTDOWN', 'EHOSTUNREACH', 'ENETDOWN', 'ENETUNREACH',
    'ECONNRESET', 'ETIMEDOUT', 'ESTALE', 'EIO') if hasattr(errno, key)}


def load_image_with_retry(path, attempts=6, sleep=time.sleep, opener=None):
    if attempts < 1:
        raise ValueError('attempts must be positive')
    opener = opener or Image.open
    for attempt in range(attempts):
        image = None
        try:
            image = opener(path)
            # PIL decoding is lazy; load here so read errors are retried too.
            image.load()
            return image
        except OSError as error:
            if image is not None:
                image.close()
            if error.errno not in TRANSIENT_ERRNOS or attempt + 1 == attempts:
                raise
            delay = min(5 * 2 ** attempt, 30)
            print('[image I/O retry] errno=%s attempt=%d/%d wait=%ds path=%s' %
                  (error.errno, attempt + 1, attempts, delay, path), flush=True)
            sleep(delay)
