from typing import List, Optional
import numpy as np
import matplotlib.pyplot as plt


def gallery(
        images_y_then_x: List[List[np.ndarray]],
        x_axis_text: List[str],
        y_axis_text: List[str],
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        dpi: Optional[int] = None,
        fontsize: int = 12,
        figsize=(16, 6)):
    """
    Create a gallery of images

    Notes:
        extracted from `analysis_plots.gallery`

    Args:
        images_y_then_x: an array of y * x images, 0-255 image
        x_axis_text: the text for each x
        y_axis_text: the text for each y
        title: the title of the gallery
        save_path: where to save the figure
        dpi: dpi of the figure

    Returns:
        a figure
    """
    fig, axes_all = plt.subplots(nrows=len(images_y_then_x), ncols=len(images_y_then_x[0]), constrained_layout=True, dpi=dpi, figsize=figsize)
    if len(images_y_then_x) == 1:
        # special case: matplotlib doesn't add a row when nrow == 1
        axes_all = [axes_all]
    fig.subplots_adjust(hspace=0.01, wspace=0.01)
    for y, (axes_x, images_x) in enumerate(zip(axes_all, images_y_then_x)):
        for x, (ax, i) in enumerate(zip(axes_x, images_x)):
            assert i.max() <= 255
            assert i.min() >= 0
            ax.imshow(i, vmin=0, vmax=255)

            ax.set_xticks([])
            ax.set_yticks([])

            if x == 0:
                ax.set_ylabel(y_axis_text[y])
            if y + 1 == len(images_y_then_x):
                ax.set_xlabel(x_axis_text[x])

    if title is not None:
        fig.suptitle(title, fontsize=fontsize)

    if save_path is not None:
        fig.savefig(save_path, dpi=dpi)

    return fig
