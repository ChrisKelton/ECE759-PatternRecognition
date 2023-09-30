import numpy as np
from typing import Callable, Tuple
from pathlib import Path
import matplotlib.pyplot as plt


def contour_plot_2d_func(
    func: Callable,
    output_path: Path,
    x_range: Tuple[float, float] = (-50, 50),
    y_range: Tuple[float, float] = (-50, 50),
    steps: int = 100,
):
    v_func = np.vectorize(func)
    x, y = np.meshgrid(np.linspace(x_range[0], x_range[1], steps), np.linspace(y_range[0], y_range[1], steps))
    fig, ax = plt.subplots(1)
    ax.contour(x, y, v_func(x, y))
    plt.grid()
    plt.show()
    fig.savefig(output_path)
    plt.close()
    return fig, ax


def cost_func(x, y):
    return 0.5 * (((x - 1) ** 2) + (4 * ((y - 0) ** 2)))


def constraint_func(x, y):
    return ((x - 3) ** 2) + (y ** 2) - 1


def main():
    # x_range = (-1, 4)
    # y_range = (-4, 4)
    # output_path = Path(r"C:\Users\cblim\Documents\NCSU\Courses\ECE759\HW\HW04\problem2-cost-func-contour-plot.png")
    # contour_plot_2d_func(cost_func, output_path, x_range=x_range, y_range=y_range)
    #
    # # output_path = output_path.parent / "problem2-cost-func-and-constraint-func-contour-plot.png"
    # # contour_plot_2d_func(constraint_func, output_path)

    # output_path = output_path.parent / "problem2-constraint-func-contour-plot.png"
    # contour_plot_2d_func(constraint_func, output_path, x_range=(-5, 7), y_range=(-5, 7))

    x = 11/3
    y = np.sqrt(5) / 3
    print(f"J({x}, {y}) = {cost_func(x, y)}")


if __name__ == '__main__':
    main()
