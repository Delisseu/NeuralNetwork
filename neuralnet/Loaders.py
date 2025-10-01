import cupy as cp
import numpy as np


class DataLoader:
    def __init__(self, x, y, batch_size=512, shuffle=True):
        assert len(x) == len(y)
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(x))

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

        self.current = 0
        return self

    def __next__(self):
        if self.current >= len(self.x):
            raise StopIteration
        start = self.current
        end = start + self.batch_size
        idx = self.indices[start:end]
        self.current = end

        next_x = cp.asarray(self.x[idx], dtype=cp.float32)

        if isinstance(self.y, (list, tuple)):
            next_y = [cp.asarray(v[idx], dtype=cp.float32) for v in self.y]
        else:
            next_y = cp.asarray(self.y[idx], dtype=cp.float32)

        return next_x, next_y


class AsyncCupyDataLoader:
    def __init__(self, x, y, batch_size=512, shuffle=True, stream=None):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.stream = stream
        if stream is None:
            self.stream = cp.cuda.Stream(non_blocking=True)
        self.indices = np.arange(len(x))
        self.current_batch = 0
        self.max_batches = (len(x) + batch_size - 1) // batch_size

        # буферы для хранения следующего батча
        self._next_x = None
        self._next_y = None

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

        self.current_batch = 0
        self._preload_next()  # загружаем первый батч
        return self

    def __next__(self):
        if self.current_batch >= self.max_batches:
            raise StopIteration

        self.stream.synchronize()
        cp.cuda.Stream.null.synchronize()

        x_batch = self._next_x
        y_batch = self._next_y

        self.current_batch += 1
        if self.current_batch < self.max_batches:
            self._preload_next()

        return x_batch, y_batch

    def _preload_next(self):
        start = self.current_batch * self.batch_size
        end = min((self.current_batch + 1) * self.batch_size, len(self.x))
        idx = self.indices[start:end]

        with self.stream:
            self._next_x = cp.asarray(self.x[idx], dtype=cp.float32)

            if isinstance(self.y, (list, tuple)):
                self._next_y = [cp.asarray(v[idx], dtype=cp.float32) for v in self.y]
            else:
                self._next_y = cp.asarray(self.y[idx], dtype=cp.float32)
