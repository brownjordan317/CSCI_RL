import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from PIL import Image
from IPython.display import HTML


def show_heatmap_animation_matplotlib(
    image_dir="heatmaps",
    prefix="V_iter_",
    ext=".png",
    fps=5,
    figsize=(6, 10)
):
    """
    Display saved heatmaps as an interactive matplotlib animation with:
    - play / pause
    - step forward / backward
    - speed control
    - slider scrub

    Works inline in Jupyter via HTML5.
    """

    # --------------------------------------------------
    # Load frames
    # --------------------------------------------------
    image_files = sorted(
        f for f in os.listdir(image_dir)
        if f.startswith(prefix) and f.endswith(ext)
    )

    if not image_files:
        raise RuntimeError("No heatmap images found.")

    frames = [
        np.asarray(Image.open(os.path.join(image_dir, f)))
        for f in image_files
    ]

    # --------------------------------------------------
    # Create figure
    # --------------------------------------------------
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")

    im = ax.imshow(frames[0])

    # --------------------------------------------------
    # Animation update
    # --------------------------------------------------
    def update(frame):
        im.set_data(frames[frame])
        return (im,)

    interval = int(1000 / fps)

    anim = FuncAnimation(
        fig,
        update,
        frames=len(frames),
        interval=interval,
        blit=True
    )

    plt.close(fig)

    return HTML(anim.to_jshtml())
