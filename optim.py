import tensorflow as tf
import re
from typing import List, Union, Tuple


class SGD(tf.train.MomentumOptimizer):
    def __init__(self, lr: tf.Tensor, mom: float, wd: float):
        super().__init__(lr, momentum=mom, use_nesterov=True)
        self.wd = wd

    def compute_gradients(self, loss, var_list=None, **kwargs) -> List[Tuple]:
        grads_and_vars = super().compute_gradients(loss, var_list=var_list)

        l = len(grads_and_vars)
        for i in range(l):
            g, v = grads_and_vars[i]
            g += v * self.wd
            grads_and_vars[i] = (g, v)

        return grads_and_vars


class AdamWeightDecayOptimizer(tf.train.Optimizer):
    def _apply_sparse(self, grad, var):
        pass

    def _resource_apply_sparse(self, grad, handle, indices):
        pass

    def _resource_apply_dense(self, grad, handle):
        pass

    def _apply_dense(self, grad, var):
        pass

    def __init__(self, lr: Union[float, tf.Tensor] = 0.001, wd: float = 0.0, beta_1: float = 0.9, beta_2: float = 0.999,
                 epsilon: float = 1e-8, exclude_from_wd: List = None, name: str = "AdamWeightDecayOptimizer"):
        super().__init__(use_locking=False, name=name)

        self.lr = lr
        self.wd = wd
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.exclude_from_weight_decay = exclude_from_wd

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        assignments = []
        for (grad, param) in grads_and_vars:
            if grad is None or param is None:
                continue

            param_name = self._get_variable_name(param.name)

            m = tf.get_variable(name=param_name + "/adam_m", shape=param.shape.as_list(), dtype=tf.float32,
                                trainable=False, initializer=tf.zeros_initializer())
            v = tf.get_variable(name=param_name + "/adam_v", shape=param.shape.as_list(), dtype=tf.float32,
                                trainable=False, initializer=tf.zeros_initializer())

            next_m = m * self.beta_1 + grad * (1.0 - self.beta_1)
            next_v = v * self.beta_2 + tf.square(grad) * (1.0 - self.beta_2)

            update = next_m / (tf.sqrt(next_v) + self.epsilon)

            if self._do_use_weight_decay(param_name):
                update += self.wd * param

            next_param = param - self.lr * update

            assignments.extend([param.assign(next_param), m.assign(next_m), v.assign(next_v)])

        assignments.append(global_step.assign_add(1))
        return tf.group(*assignments, name=name)

    def _do_use_weight_decay(self, param_name):
        if not self.wd:
            return False
        if self.exclude_from_weight_decay:
            for r in self.exclude_from_weight_decay:
                if re.search(r, param_name) is not None:
                    return False
        return True

    @staticmethod
    def _get_variable_name(param_name):
        m = re.match("^(.*):\\d+$", param_name)
        if m is not None:
            param_name = m.group(1)
        return param_name
