import os
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from matplotlib import cm
from benchmarks.Examplers import Example, get_example_by_name
from plots.plot_fcns import plot_circle_sets, vector_field, plot_square_sets, Rectangle


def barrier_3d(X, Y, B, is_zero=True):
    # Plot Barrier functions
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    suf = ax.plot_surface(X, Y, B, rstride=5, cstride=5, alpha=0.4, cmap=cm.jet)
    if is_zero:
        zero = ax.contour(X, Y, B, 0, zdir='z', offset=0, colors='k')
    return ax


def init_plot_ground(plot_limit, x, B):
    X = np.linspace(-plot_limit, plot_limit, 100)
    Y = np.linspace(-plot_limit, plot_limit, 100)
    x0, x1 = np.meshgrid(X, Y)
    lambda_b = sp.lambdify(x, str(B), modules=['numpy'])
    plot_b = lambda_b(x0, x1)  # 100*100 numpy-array
    return plot_b, x0, x1, X, Y


def plot_vector_field(ax, plot_limit, f, plot_b, X, Y):
    xv = np.linspace(-plot_limit, plot_limit, 10)
    yv = np.linspace(-plot_limit, plot_limit, 10)
    Xv, Yv = np.meshgrid(xv, yv)
    t = np.linspace(0, 5, 100)
    vector_field(f, Xv, Yv, t)
    ax.contour(X, Y, plot_b, 0, linewidths=2, colors='k')


def set_3d_labels_and_title(ax, x_label, y_label, z_label, title):
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)
    plt.title(title)


def set_2d_labels_and_title(x_label, y_label, title):
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)


def plot_benchmark2d(ex: Example, B):
    # path = 'result/' + ex.name + '/'
    # if not os.path.exists(path):
    #     os.mkdir(path)
    x = sp.symbols(['x{}'.format(i + 1) for i in range(ex.n)])
    plot_limit = max(ex.local.up)
    plot_b, x0, x1, X, Y = init_plot_ground(plot_limit, x, B)
    ax = barrier_3d(x0, x1, plot_b)

    ax = plot_square_sets(ax, ex.target.low,
                          [ex.target.up[0] - ex.target.low[0], ex.target.up[1] - ex.target.low[1]], 'r',
                          'Unsafe Set')
    set_3d_labels_and_title(ax, '$x_1$', '$x_2$', 'B', 'Barrier Certificate')
    # plt.savefig(path + ex.name + '_3d.pdf')
    ################################
    # PLOT 2D
    ################################
    plt.figure()
    ax = plt.gca()
    # target Region
    p = Rectangle(ex.target.low, *[ex.target.up[0] - ex.target.low[0], ex.target.up[1] - ex.target.low[1]],
                  linestyle='--', color='r',
                  linewidth=1.5, fill=False, label='Target Set')
    ax.add_patch(p)
    # plots vector field
    plot_vector_field(ax, plot_limit, ex.f, plot_b, X, Y)
    set_2d_labels_and_title('$x_1$', '$x_2$', 'Barrier Border')
    # plt.savefig(path + ex.name + '_barrier2d.pdf')
    plt.show()


def plot_exponential_bench(x, B):
    # elementary domains:
    # X: -2 <= x, y <= 2
    # X_i: (x+0.5)**2 + (y-0.5)**2 <= 0.16
    # X_u: (x-0.7)**2 + (y+0.7)**2 <= 0.09

    plot_limit = 1.0
    plot_b, x0, x1, X, Y = init_plot_ground(plot_limit, x, B)

    ax = barrier_3d(x0, x1, plot_b)
    ax = plot_circle_sets(ax, [-0.5, +0.5], 0.16, 'g', 'Initial Set')
    ax = plot_circle_sets(ax, [0.7, -0.7], 0.09, 'r', 'Unsafe Set')
    set_3d_labels_and_title(ax, '$x$', '$y$', 'B', 'Barrier Certificate')
    ################################
    # PLOT 2D
    ################################
    plt.figure()
    ax = plt.gca()

    def f(v):
        x, y = v
        dydt = [np.exp(-x) + y - 1, -np.sin(x) ** 2]
        return dydt

    # Initial Region
    ax = plot_circle_sets(ax, [-0.5, +0.5], 0.16, 'g', 'Initial Set')
    ax = plot_circle_sets(ax, [0.7, -0.7], 0.09, 'r', 'Unsafe Set')
    # plots vector field
    plot_vector_field(ax, plot_limit, f, plot_b, X, Y)
    set_2d_labels_and_title('$x$', '$y$', 'Barrier Border')
    plt.show()


def plot_pjmod_bench(x, B):
    # prajna domains:
    # X: -3 <= x <= 2.5, -2 <= y <= 1
    # X_i: circle+L
    # X_u: circle+L
    plot_limit = 2.0
    plot_b, x0, x1, X, Y = init_plot_ground(plot_limit, x, B)

    ax = barrier_3d(x0, x1, plot_b)
    ax = plot_square_sets(ax, [-1.8, -0.1], [0.6, 0.2], 'g', 'Initial Set')
    ax = plot_square_sets(ax, [-1.4, -0.5], [0.2, 0.6], 'g', '')
    ax = plot_circle_sets(ax, (1.5, 0), 0.25, 'g', '')

    ax = plot_square_sets(ax, [0.4, 0.1], [0.2, 0.4], 'r', 'Unsafe Set')
    ax = plot_square_sets(ax, [0.4, 0.1], [0.4, 0.2], 'r', '')
    ax = plot_circle_sets(ax, (-1, -1), 0.16, 'r', '')

    set_3d_labels_and_title(ax, '$x_1$', '$x_2$', 'B', 'Barrier Certificate')

    ################################
    # PLOT 2D
    ################################
    plt.figure()
    ax = plt.gca()

    def f(v):
        x, y = v
        dydt = [y, - x - y + 1.0 / 3.0 * x ** 3]
        return dydt

    # init Region
    p = Rectangle((-1.8, -0.1), 0.6, 0.2, linestyle='--', color='g', \
                  linewidth=1.5, fill=False, label='Initial Set')
    ax.add_patch(p)
    p = Rectangle((-1.4, -0.5), 0.2, 0.6, linestyle='--', color='g', \
                  linewidth=1.5, fill=False, label='')
    ax.add_patch(p)
    ax = plot_circle_sets(ax, (1.5, 0), 0.25, 'g', '')

    # unsafe
    p = Rectangle((0.4, 0.1), 0.2, 0.4, linestyle='--', color='r', \
                  linewidth=1.5, fill=False, label='Unsafe Set')
    ax.add_patch(p)
    p = Rectangle((0.4, 0.1), 0.4, 0.2, linestyle='--', color='r', \
                  linewidth=1.5, fill=False, label='')
    ax.add_patch(p)
    ax = plot_circle_sets(ax, (-1, -1), 0.16, 'r', '')

    # plots vector field
    plot_vector_field(ax, plot_limit, f, plot_b, X, Y)
    set_2d_labels_and_title('$x_1$', '$x_2$', 'Barrier Border')
    plt.show()


if __name__ == '__main__':
    '''
    test_code
    '''
    s = '263.754161501217*x1**2 + 49.7589614277513*x1*x2 + 0.825419347747078*x1 + 250.98120892253*x2**2 - 0.333889552596392*x2 - 1.63350417664261'
    s = s.replace(' ', '')
    V = sp.symbols(s)
    plot_benchmark2d(get_example_by_name('Car'), V)
