from .imports import *


def get_variable_name(param_name: str) -> str:
    m = re.match("^(.*):\\d+$", param_name)
    if m is not None:
        param_name = m.group(1)
    return param_name


class SGD(tf.train.MomentumOptimizer):
    def __init__(self, lr: tf.Tensor, mom: float, wd: float):
        super().__init__(lr, momentum=mom, use_nesterov=True)
        self.wd = wd

    def compute_gradients(self, loss: tf.Tensor, var_list: List[tf.Tensor] = None, **kwargs) -> List[
        Tuple[tf.Tensor, tf.Tensor]]:
        grads_and_vars = super().compute_gradients(loss, var_list=var_list)

        l = len(grads_and_vars)
        for i in range(l):
            g, v = grads_and_vars[i]
            g += v * self.wd
            grads_and_vars[i] = (g, v)

        return grads_and_vars


class Adam(tf.train.AdamOptimizer):
    def __init__(self, lr: Union[float, tf.Tensor] = 0.001, wd: float = None, beta1: float = 0.9, beta2: float = 0.999,
                 epsilon: float = 1e-8, exclude_from_wd: List[str] = None, clip_norm: float = None):
        super().__init__(learning_rate=lr, beta1=beta1, beta2=beta2, epsilon=epsilon)
        self.wd = 1 - wd * lr if wd and wd > 0 else None
        self.exclude_from_wd = exclude_from_wd
        self.clip_norm = clip_norm

    def compute_gradients(self, loss: tf.Tensor, var_list: List[tf.Tensor] = None, **kwargs) -> Iterator[Tuple]:

        grads_and_vars = super().compute_gradients(loss, var_list)
        if not self.clip_norm and not self.wd:
            return grads_and_vars

        gs, vs = zip(*grads_and_vars)
        if self.clip_norm is not None:
            gs, _ = tf.clip_by_global_norm(gs, clip_norm=self.clip_norm)

        return zip(gs, vs)

    def _apply_dense(self, grad, var):
        if not self._do_use_wd(get_variable_name(var.name)):
            return super()._apply_dense(grad, var)

        m = self.get_slot(var, "m")
        v = self.get_slot(var, "v")
        beta1_power, beta2_power = self._get_beta_accumulators()

        lr = (self._lr_t * tf.sqrt(1 - beta2_power) / (1 - beta1_power))  # learning rate bias correction

        m_t = tf.assign(m, m * self._beta1_t + grad * (1 - self._beta1_t))
        v_t = tf.assign(v, v * self._beta2_t + tf.square(grad) * (1 - self._beta2_t))
        v_sqrt = tf.sqrt(v_t)

        var_update = tf.assign(var, var * self.wd - lr * m_t / (v_sqrt + self._epsilon_t))
        return tf.group(*[var_update, m_t, v_t])

    def _do_use_wd(self, param_name: str) -> bool:
        if self.wd is None:
            return False

        if self.exclude_from_wd:
            for r in self.exclude_from_wd:
                if re.search(r, param_name) is not None:
                    return False
        return True
