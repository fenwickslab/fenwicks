import tensorflow as tf
from matplotlib import animation, rc
import matplotlib.pylab as plt
from typing import List, Callable
from IPython.display import Image, display

from .. import vision


def setup():
    rc('animation', html='jshtml')


def images_anim(images):
    def animate(i):
        ax.imshow(images[i])

    fig, ax = plt.subplots()
    plt.close()
    fig.tight_layout()
    fig.set_size_inches(3, 3)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.axis('off')
    anim = animation.FuncAnimation(fig, animate, frames=len(images), interval=1000)
    return anim


def show_image_files(files: List[str]):
    X = []
    for fn in files:
        x = plt.imread(fn)
        X.append(x)
    return images_anim(X)


def show_dataset(ds: tf.data.Dataset, n_batch: int = 1, n_img: int = 10,
                 reverse_normalizer: Callable = vision.transform.reverse_imagenet_normalize_tf):
    X = []
    data_op = ds.make_one_shot_iterator().get_next()

    with tf.Session() as sess:
        for _ in range(n_batch):
            x, _ = sess.run(data_op)
            if len(x) >= n_img:
                X.extend(x[:n_img])
                break
            X.extend(reverse_normalizer(x))
            n_img -= len(x)

    anim = images_anim(X)
    display(anim)


def show_transform(tfm, img_fn: str, n_frames: int = 5, fps: int = 5, anim_fn: str = '/tmp/anim.gif'):
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
    anim.save(anim_fn, writer='imagemagick', fps=fps)
    anim_fn_png = f'{anim_fn}.png'
    tf.gfile.Copy(anim_fn, anim_fn_png, overwrite=True)
    display(Image(anim_fn_png))
