import math


class SGDRLearningRate:

    def __init__(self, optimizer, learning_rate, t_0, mul=1.0):
        self.optimizer = optimizer
        self.initial_learning_rate = learning_rate
        self.learning_rate = learning_rate
        self.t_0 = t_0
        self.mul = mul

    def __call__(self, step):
        self.learning_rate, reset = self.value(step)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.learning_rate
        return reset

    def value(self, step):
        x = step / self.t_0
        i_restart = int(x)
        x = x - i_restart
        base = self.initial_learning_rate * (self.mul ** i_restart)

        return 0.5 * base * (math.cos(math.pi * x) + 1), x == 0
