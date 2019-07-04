from ..imports import *

import plotly.plotly
import plotly.figure_factory as ff
import plotly.graph_objs as go
import cufflinks as cf
import operator
import IPython

from IPython.display import display
from collections import Counter
from sklearn.neighbors import KernelDensity
from statsmodels.nonparametric import bandwidths


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
    try:
        IPython.get_ipython().events.register('pre_run_cell', configure_plotly_browser_state)
    except:
        pass
    cf.set_config_file(offline=True)


def simulate_lr_func(lr_func: Callable, total_steps: int) -> np.array:
    lr_values = np.zeros(total_steps)
    step = tf.placeholder(tf.int64)
    lr = lr_func(step=step)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(total_steps):
            lr_values[i] = sess.run(lr, feed_dict={step: i})

    return lr_values


def layout_size_margin(layout: go.Layout, h: int, w: int, l: int, r: int, b: int, t: int):
    layout.width = w
    layout.height = h
    layout.margin = go.layout.Margin(l=l, r=r, b=b, t=t)


def layout_axes_title(layout: go.Layout, xtitle: str = None, ytitle: str = None):
    layout.xaxis = go.layout.XAxis(title=xtitle) if xtitle else None
    layout.yaxis = go.layout.YAxis(title=ytitle) if ytitle else None


def plot_scatter(ys, h: int = 350, w: int = 350, ytitle: str = None, xtitle: str = None):
    data = [(go.Scatter(y=ys))]
    fig = go.Figure(data=data)
    layout_size_margin(fig.layout, h, w, l=80, r=20, b=40, t=20)
    layout_axes_title(fig.layout, xtitle, ytitle)
    plotly.offline.iplot(fig)


def plot_lr_func(lr_func: Callable, total_steps: int):
    """
    Draw an interactive plot for learning rate vs. training step.

    :param lr_func: Learning rate schedule function.
    :param total_steps: Total number of training steps.
    :return: None.
    """
    ys = simulate_lr_func(lr_func, total_steps)
    plot_scatter(ys, ytitle='Learning rate', xtitle='Training step')


def plot_series_counts(s: pd.Series):
    series = s.value_counts().sort_values(ascending=False)
    layout = go.Layout()
    layout_size_margin(layout, h=300, w=350, l=50, r=20, b=40, t=0)
    layout_axes_title(layout, ytitle='Count')
    series.iplot(kind='bar', layout=layout)


def plot_series_histogram(s: pd.Series):
    layout = go.Layout()
    layout_size_margin(layout, h=300, w=350, l=50, r=20, b=40, t=0)
    layout_axes_title(layout, ytitle='Count')
    s.iplot(kind='histogram', layout=layout)


def plot_pie_df(df: pd.DataFrame, w: int = 350):
    layout = go.Layout()
    layout_size_margin(layout, h=350, w=w, l=50, r=0, b=0, t=0)
    layout_axes_title(layout, xtitle='x', ytitle='y')  # to make cufflinks happy
    df.iplot(kind='pie', labels='id', values='count', layout=layout, pull=.05, hole=0.2)


def plot_df_bar(df: pd.DataFrame, col: str, w: int = 350):
    series = df[col]
    series.index = series.index.astype(str)
    layout = go.Layout()
    layout_size_margin(layout, h=300, w=w, l=50, r=50, b=80, t=0)
    layout_axes_title(layout, ytitle=col)
    series.iplot(kind='bar', layout=layout)


# todo: merge items beyond max_item into an 'others' class
def plot_counts_pie_df(df: pd.DataFrame, col: str, max_items: int = 10, w: int = 350):
    s = df[col].value_counts().sort_values(ascending=False)[:max_items]
    pie_df = pd.DataFrame({'id': s.index, 'count': s.values})
    plot_pie_df(pie_df, w=w)


# todo: merge items beyond max_item into an 'others' class
def plot_counts_pie(y: List[int], labels: List[str] = None, max_items: int = -1, w: int = 350):
    if labels is None:
        labels = np.unique(y)

    if max_items < 0:
        max_items = len(labels)

    cnt = Counter(y)

    cnt_copy = cnt.copy()
    for k in cnt_copy:
        cnt[labels[k]] = cnt.pop(k)

    items = sorted(cnt.items(), key=operator.itemgetter(1), reverse=True)
    pie_df = pd.DataFrame(items[:max_items], columns=['id', 'count'])
    plot_pie_df(pie_df, w=w)


heatmap_colorscale = [[0, "rgb(255,245,240)"],
                      [0.2, "rgb(254,224,210)"],
                      [0.4, "rgb(252,187,161)"],
                      [0.5, "rgb(252,146,114)"],
                      [0.6, "rgb(251,106,74)"],
                      [0.7, "rgb(239,59,44)"],
                      [0.8, "rgb(203,24,29)"],
                      [0.9, "rgb(165,15,21)"],
                      [1, "rgb(103,0,13)"]]


def plot_heatmap(xs, ys, zs, h: int = 350, w: int = 550, xtitle: str = None, ytitle: str = None):
    data = [go.Heatmap(x=xs, y=ys, z=zs, colorscale=heatmap_colorscale)]
    fig = go.Figure(data=data)
    layout_size_margin(fig.layout, h, w, l=120, r=0, b=80, t=0)
    layout_axes_title(fig.layout, xtitle, ytitle)
    plotly.offline.iplot(fig)


def plot_df_corr(df: pd.DataFrame, h: int = 350, w: int = 450):
    df_corrs = df.corr()
    z = np.array(df_corrs)
    z_text = np.around(z, decimals=2)
    x = list(df_corrs.index)
    fig = ff.create_annotated_heatmap(z, annotation_text=z_text, x=x, y=x, colorscale=heatmap_colorscale,
                                      showscale=True)
    layout_size_margin(fig.layout, h, w, l=120, r=0, b=0, t=80)
    plotly.offline.iplot(fig)


def plot_confusion_mat(xs, ys, zs, h: int = 350, w: int = 550):
    plot_heatmap(xs, ys, zs, h, w, 'Predicted value', 'True Value')


def show(fig: go.Figure):
    plotly.offline.iplot(fig)


def trace_kde(s: pd.Series, trace_name: str = None) -> go.Scatter:
    x = s.dropna()
    bw = bandwidths.bw_scott(x)
    x_plot = np.linspace(x.min(), x.max(), 1000)[:, np.newaxis]
    x = np.array(x)[:, np.newaxis]
    kde = KernelDensity(bandwidth=bw).fit(x)
    log_dens = kde.score_samples(x_plot)
    return go.Scatter(x=x_plot[:, 0], y=np.exp(log_dens), fill='tozeroy', line=dict(color='#AAAAFF'), name=trace_name)


def plot_kde(s: pd.Series, h: int = 300, w: int = 350, trace_name: str = None):
    trace = trace_kde(s, trace_name)

    layout = go.Layout()
    layout_size_margin(layout, h, w, l=20, r=0, b=20, t=0)
    fig = go.Figure(data=[trace], layout=layout)
    show(fig)
