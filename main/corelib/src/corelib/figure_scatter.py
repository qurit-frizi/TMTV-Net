import numpy as np
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, RANSACRegressor
import numpy as np


def plot_scatter(
        title: str, 
        x: np.ndarray, 
        y: np.ndarray, 
        xlabel: str, 
        ylabel: str, 
        path: str, 
        colors: List[Tuple[float, float, float]],
        fontsize: int = 12, 
        label_class: Optional[List[str]] = None, 
        xy_label: Optional[List[str]] = None, 
        display_nb_outliers: int = 3, 
        min_outlier_error: float = 0, 
        robust_linear_fit: bool = False) -> Dict:
    """
    Plot a scatter graph, linearly fit (x, y) and calculate
    metrics (linear fit, R2, outliers)
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize + 4)
    
    nb_classes = len(set(label_class))
    if label_class is None or nb_classes == 1:
        ax.scatter(x=x, y=y)
    else:
        # plot for each class independently
        classes = set(label_class)
        for c_id, c in enumerate(classes):
            indices = np.asarray(label_class) == c
            xs = x[indices]
            ys = y[indices]
            cs = [colors[c_id]] * len(xs)
            ls = c
            ax.scatter(x=xs, y=ys, c=cs, label=ls)
    
    if robust_linear_fit:
        ransac_regression = RANSACRegressor()
        ransac_regression.fit(x.reshape(-1, 1), y.reshape(-1, 1))
        linear_regression = ransac_regression.estimator_
    else:
        linear_regression = LinearRegression(normalize=True)
        linear_regression.fit(x.reshape(-1, 1), y.reshape(-1, 1))

    metrics = {}
    a = float(linear_regression.coef_)
    b = float(linear_regression.intercept_)
    xs = np.linspace(x.min(), x.max(),100)
    ys = a * xs + b
    r2 = linear_regression.score(x.reshape(-1, 1), y.reshape(-1, 1))
    r2_no_outlier = -1
    metrics['r2'] = r2
    metrics['coef'] = a
    metrics['intercept'] = b

    if display_nb_outliers is not None and display_nb_outliers > 0 and xy_label is not None and len(x) > 5:
        # with the samples with worst
        residual = abs(linear_regression.predict(x.reshape([-1, 1])).squeeze() - y).squeeze()
        assert len(xy_label) == len(x)
        outlier_indices = np.argsort(residual)[::-1]
        for i in outlier_indices[:display_nb_outliers]:
            e = residual[i]
            if e > min_outlier_error:
                ax.text(x=x[i], y=y[i], s=xy_label[i])

        # re-estimate without outliers
        x_good = x[outlier_indices[display_nb_outliers + 1:]]
        y_good = y[outlier_indices[display_nb_outliers + 1:]]
        linear_regression.fit(x_good.reshape(-1, 1), y_good.reshape(-1, 1))
        r2_no_outlier = linear_regression.score(x_good.reshape(-1, 1), y_good.reshape(-1, 1))
        metrics['r2_no_outlier'] = r2_no_outlier
        metrics['outliers'] = np.asarray(xy_label)[outlier_indices[:display_nb_outliers]]

    ax.plot(xs, ys, '-r', label=f'{a:.3f} * x + {b:.3f}, R2={r2: .3f}, R2_no_outlier={r2_no_outlier: .3f}')
    ax.legend(loc='upper left')
    ax.grid()

    fig.savefig(path)
    plt.close(fig)
    return metrics
