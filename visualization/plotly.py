import tensorflow as tf
import pandas as pd
import plotly.plotly
import plotly.graph_objs as go
import numpy as np
import cufflinks as cf
import operator
import IPython

from IPython.display import display
from typing import List
from collections import Counter


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
    cf.set_config_file(offline=True)


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


def plot_df_counts(df: pd.DataFrame, col: str, max_items: int = 10):
    series = df[col].value_counts().sort_values(ascending=False)[:max_items]
    layout = go.Layout(height=350, width=350, yaxis=go.layout.YAxis(title='Count'),
                       margin=go.layout.Margin(l=80, r=20, b=40, t=20))
    series.iplot(kind='bar', yTitle='Count', layout=layout)


def plot_pie_df(pie_df: pd.DataFrame, width: int = 350):
    layout = go.Layout(height=350, width=width, margin=go.layout.Margin(l=50, r=0, b=0, t=0),
                       yaxis=go.layout.YAxis(title='y'), xaxis=go.layout.XAxis(title='x'))
    pie_df.iplot(kind='pie', labels='id', values='count', layout=layout, pull=.05, hole=0.2)


# todo: merge items beyond max_item into an 'others' class
def plot_counts_pie_df(df: pd.DataFrame, col: str, max_items: int = 10, width: int = 350):
    s = df[col].value_counts().sort_values(ascending=False)[:max_items]
    pie_df = pd.DataFrame({'id': s.index, 'count': s.values})
    plot_pie_df(pie_df, width=width)


# todo: merge items beyond max_item into an 'others' class
def plot_counts_pie(y: List[int], labels: List[str] = None, max_items: int = -1, width: int = 350):
    labels = labels or np.unique(y)

    if max_items < 0:
        max_items = len(labels)

    cnt = Counter(y)

    cnt_copy = cnt.copy()
    for k in cnt_copy:
        cnt[labels[k]] = cnt.pop(k)

    items = sorted(cnt.items(), key=operator.itemgetter(1), reverse=True)
    pie_df = pd.DataFrame(items[:max_items], columns=['id', 'count'])
    plot_pie_df(pie_df, width=width)


def plot_confusion_mat(xs, ys, zs, h: int = 350, w: int = 550):
    trace = {"x": xs, "y": ys, "z": zs,
             "autocolorscale": False,
             "colorscale": [
                 [0, "rgb(255,245,240)"],
                 [0.2, "rgb(254,224,210)"],
                 [0.4, "rgb(252,187,161)"],
                 [0.5, "rgb(252,146,114)"],
                 [0.6, "rgb(251,106,74)"],
                 [0.7, "rgb(239,59,44)"],
                 [0.8, "rgb(203,24,29)"],
                 [0.9, "rgb(165,15,21)"],
                 [1, "rgb(103,0,13)"]],
             }
    layout = {"autosize": False,
              "height": h, "width": w,
              "xaxis": {"title": "Predicted value"},
              "yaxis": {"title": "True Value"},
              "margin": go.layout.Margin(l=100, r=20, b=40, t=20),
              }

    data = [go.Heatmap(**trace)]
    fig = go.Figure(data=data, layout=layout)
    plotly.offline.iplot(fig)
