from matplotlib import animation, rc
import matplotlib.pylab as plt
from typing import List
from IPython.display import display


def setup():
    rc('animation', html='jshtml')


def show_images(files: List[str]):
    def animate(i):
        X = plt.imread(files[i])
        ax.imshow(X)

    fig, ax = plt.subplots()
    plt.close()
    fig.tight_layout()
    fig.set_size_inches(3, 3)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.axis('off')
    anim = animation.FuncAnimation(fig, animate, frames=4, interval=1000)
    display(anim)
