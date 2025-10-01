import cupy as cp

from neuralnet.CONSTANTS import ZERO, ONE, TWO, HALF


class Optimizer:
    def step(self, param, grad, trainable, lr):
        raise NotImplementedError("Необходимо переопределить метод step")


class InverseSqrtScheduler:
    def __init__(self, warmup_steps=500, n_init=0.00001):
        self.warmup_steps = cp.asarray(warmup_steps, dtype=cp.float32)
        self.n_init = cp.asarray(n_init, dtype=cp.float32)
        self.steps = {}

    def step_update(self, lr, key):
        if key not in self.steps:
            self.steps[key] = ZERO.copy()

        self.steps[key] += ONE
        step = self.steps[key]
        if step < self.warmup_steps:
            n = self.n_init + step*(lr-self.n_init)/self.warmup_steps
        else:
            alpha = lr * cp.sqrt(self.warmup_steps)
            n = alpha/cp.sqrt(step)

        return n


class InverseSqrtSchedulerMod1:
    def __init__(self, warmup_steps=500):
        self.warmup_steps = cp.asarray(warmup_steps, dtype=cp.float32)
        self.steps = {}

    def step_update(self, lr, key):
        if key not in self.steps:
            self.steps[key] = ZERO.copy()
        self.steps[key] += ONE
        step = self.steps[key]
        if step < self.warmup_steps:
            return lr
        return lr * (self.warmup_steps ** HALF) / (step ** HALF)


class InverseSqrtSchedulerMod2:
    def __init__(self, warmup_steps=500):
        self.warmup_steps = cp.asarray(warmup_steps, dtype=cp.float32)
        self.steps = {}

    def step_update(self, lr, key):
        if key not in self.steps:
            self.steps[key] = ZERO.copy()
        self.steps[key] += ONE
        step = self.steps[key]
        if step < self.warmup_steps:
            return lr * step / self.warmup_steps

        return lr * (self.warmup_steps ** HALF) / (step ** HALF)


class NoScheduler:
    def step_update(self, lr, key):
        return lr


class SGD(Optimizer):
    def __init__(self, normalize=None, eps=1e-8, scheduler=NoScheduler()):
        self.scheduler = scheduler
        self.eps = cp.array(eps, dtype=cp.float32)
        self.normalize = normalize

    def step(self, param, grad, trainable, lr):
        if trainable:
            key = id(param)
            lr = self.scheduler.step_update(lr, key)
            if self.normalize:
                param /= (cp.sqrt(cp.mean(param ** TWO)) + self.eps)

            param -= lr * grad


class Adam(Optimizer):
    def __init__(self, beta1=0.9, beta2=0.999, eps=1e-8, scheduler=NoScheduler(), weight_decay=0, clip_norm=None):
        self.scheduler = scheduler
        self.beta1 = cp.asarray(beta1, dtype=cp.float32)
        self.beta2 = cp.asarray(beta2, dtype=cp.float32)
        self.eps = cp.asarray(eps, dtype=cp.float32)

        self.m = {}
        self.v = {}
        self.t = {}

        self.weight_decay = cp.asarray(weight_decay, dtype=cp.float32)
        self.clip_norm = clip_norm

    def step(self, param, grad, trainable, lr):
        if not trainable:
            return
        key = id(param)
        lr = self.scheduler.step_update(lr, key)

        if key not in self.m:
            self.m[key] = cp.zeros_like(param, dtype=cp.float32)
            self.v[key] = cp.zeros_like(param, dtype=cp.float32)
            self.t[key] = ZERO.copy()

        self.t[key] += ONE
        t = self.t[key]

        # weight decay (L2)
        if self.weight_decay != ZERO:
            grad = grad + self.weight_decay * param

        # gradient clipping
        if self.clip_norm is not None:
            gnorm = cp.linalg.norm(grad)
            if gnorm > ZERO and gnorm > self.clip_norm:
                grad = grad * (self.clip_norm / (gnorm + self.eps))

        # moments update
        self.m[key] += (ONE - self.beta1) * (grad - self.m[key])
        self.v[key] += (ONE - self.beta2) * (grad ** TWO - self.v[key])

        # bias correction
        m_hat = self.m[key] / (ONE - self.beta1 ** t)
        v_hat = self.v[key] / (ONE - self.beta2 ** t)

        # Обновление параметров слоя
        param -= lr * m_hat / (cp.sqrt(v_hat) + self.eps)

    def state_dict(self):
        """Возвращает сериализуемое состояние оптимизатора.
        WARNING: ключи — id(param) (int). При восстановлении в новом процессе
        это работает только если параметры будут иметь те же id (обычно нет).
        Лучше сохранять state на уровне слоёв (см. README).
        """
        # преобразуем cp arrays в cpu numpy для сериализации
        m_cpu = {str(k): v.get() for k, v in self.m.items()}
        v_cpu = {str(k): v.get() for k, v in self.v.items()}
        t_cpu = {str(k): int(v) for k, v in self.t.items()}
        return {"m": m_cpu, "v": v_cpu, "t": t_cpu}

    def load_state_dict(self, state):
        """Попытка восстановить состояние. Требует маппинга id(param)->id(param) в текущем процессе.
        (простая версия: загружает только структуры, без привязки к конкретным объектам)
        """
        # если в state хранятся cpu-ndarray, просто сохраняем в cp
        for k_str, arr in state.get("m", {}).items():
            self.m[int(k_str)] = cp.asarray(arr, dtype=cp.float32)
        for k_str, arr in state.get("v", {}).items():
            self.v[int(k_str)] = cp.asarray(arr, dtype=cp.float32)
        for k_str, val in state.get("t", {}).items():
            self.t[int(k_str)] = int(val)
