import tensorflow as tf
import numpy as np

import random
import matplotlib.pylab as plt

from matplotlib import rc
from matplotlib.animation import FuncAnimation
from typing import List, Callable, Union
from IPython.display import Image

from .. import vision


def setup():
    rc('animation', html='jshtml')


def images_anim(images: Union[np.ndarray, List]) -> FuncAnimation:
    def animate(i):
        ax.imshow(images[i])

    fig, ax = plt.subplots()
    plt.close()
    fig.tight_layout()
    fig.set_size_inches(3, 3)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.axis('off')
    return FuncAnimation(fig, animate, frames=len(images), interval=1000)


def show_image_files(files: List[str], n_img: int = 20) -> FuncAnimation:
    n_img = min(n_img, len(files))
    X = []
    files = random.sample(files, n_img)
    for fn in files:
        x = plt.imread(fn)
        X.append(x)
    return images_anim(X)


def show_dataset(ds: tf.data.Dataset, n_batch: int = 1, n_img: int = 10,
                 converter: Callable = vision.transform.reverse_imagenet_normalize_tf) -> FuncAnimation:
    X = []
    data_op = ds.make_one_shot_iterator().get_next()

    with tf.Session() as sess:
        for _ in range(n_batch):
            x, _ = sess.run(data_op)
            if len(x) >= n_img:
                X.extend(converter(x[:n_img]))
                break
            X.extend(converter(x))
            n_img -= len(x)

    X = np.clip(np.array(X), 0.0, 1.0)
    return images_anim(X)


def anim_gif(anim: FuncAnimation, fps: int = 1, anim_fn: str = '/tmp/anim.gif') -> Image:
    anim.save(anim_fn, writer='imagemagick', fps=fps)
    anim_fn_png = f'{anim_fn}.png'
    tf.gfile.Copy(anim_fn, anim_fn_png, overwrite=True)
    return Image(anim_fn_png)


def show_transform(tfm, img_fn: str, n_frames: int = 5, gif: bool = True, fps: int = 5,
                   anim_fn: str = '/tmp/anim.gif') -> Union[Image, FuncAnimation]:
    images = []

    img = tf.read_file(img_fn)
    img = tf.io.decode_image(img, channels=3, dtype=tf.float32)
    img.set_shape([None, None, 3])
    op = tfm(img)

    with tf.Session() as sess:
        for i in range(n_frames):
            x = sess.run(op)
            images.append(x)

    anim = images_anim(images)
    return anim_gif(anim, fps, anim_fn) if gif else anim
