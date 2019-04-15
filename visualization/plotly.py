import IPython
import tensorflow as tf
import plotly.plotly
import plotly.graph_objs as go
import numpy as np
from IPython.display import display


def configure_plotly_browser_state():
    display(IPython.core.display.HTML('''
    <script src="/static/components/requirejs/require.js"></script>
    <script>
      requirejs.config({
        paths: {
          base: '/static/base',
          plotly: 'https://cdn.plot.ly/plotly-1.47.1.min.js?noext',
        },
      });
    </script>
    '''))


def setup():
    plotly.offline.init_notebook_mode(connected=True)
    IPython.get_ipython().events.register('pre_run_cell', configure_plotly_browser_state)


def simulate_lr_func(lr_func, total_steps):
    lr_values = np.zeros(total_steps)
    step = tf.placeholder(tf.int64)
    lr = lr_func(step=step)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(total_steps):
            lr_values[i] = sess.run(lr, feed_dict={step: i})

    return lr_values


def plot_lr_func(lr_func, total_steps):
    """
    Draw an interactive plot for learning rate vs. training step.

    :param lr_func: Learning rate schedule function.
    :param total_steps: Total number of training steps.
    :return: None.
    """
    trace = go.Scatter(y=simulate_lr_func(lr_func, total_steps))
    data = [trace]
    layout = go.Layout(autosize=False, width=350, height=350, yaxis=go.layout.YAxis(title='Learning rate'),
                       xaxis=go.layout.XAxis(title='Training step'), margin=go.layout.Margin(l=80, r=20, b=40, t=20))
    fig = go.Figure(data=data, layout=layout)
    plotly.offline.iplot(fig)
