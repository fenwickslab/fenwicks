import tensorflow as tf
from matplotlib import animation, rc
import matplotlib.pylab as plt
from typing import List
from IPython.display import display


def setup():
    rc('animation', html='jshtml')


def show_images(X):
    def animate(i):
        ax.imshow(X[i])

    fig, ax = plt.subplots()
    plt.close()
    fig.tight_layout()
    fig.set_size_inches(3, 3)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.axis('off')
    anim = animation.FuncAnimation(fig, animate, frames=len(X), interval=1000)
    display(anim)


def show_image_files(files: List[str]):
    X = []
    for fn in files:
        x = plt.imread(fn)
        X.append(x)
    show_images(X)


def show_dataset(ds: tf.data.Dataset, num_batch: int = 1, n_img: int = 10):
    X = []
    data_op = ds.make_one_shot_iterator().get_next()

    with tf.Session() as sess:
        for _ in range(num_batch):
            x, _ = sess.run(data_op)
            if len(x) >= n_img:
                X.extend(x[:n_img])
                break
            X.extend(x / 2 + 0.5)  # fixme
            n_img -= len(x)

    show_images(X)


def show_transform(tfm, fn: str, n: int = 5):
    X = []

    img = tf.read_file(fn)
    img = tf.io.decode_image(img, channels=3, dtype=tf.float32)
    img.set_shape([None, None, 3])
    op = tfm(img)

    with tf.Session() as sess:
        for i in range(n):
            x = sess.run(op)
            X.append(x)

    show_images(X)
