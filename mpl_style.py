import matplotlib
import matplotlib.ticker
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('mvstyle')

# class MyLocator(matplotlib.ticker.AutoMinorLocator):
#     def __init__(self, n=23):
#         super().__init__(n=n)


# class MyLogLocator(matplotlib.ticker.LogLocator):
#     def __init__(self, n=10):
#         super().__init__(base = 10, numticks=n)



# ###################################################

# matplotlib.ticker.AutoMinorLocator = MyLocator
# matplotlib.ticker.LogLocator = MyLogLocator

def loglog(ax, *args, **kwargs):

    ax.loglog(*args, **kwargs)

    locmajx = matplotlib.ticker.LogLocator(base = 10.0, numticks=20)
    ax.xaxis.set_major_locator(locmajx)

    locminx = matplotlib.ticker.LogLocator(base=10.0, subs=np.linspace(0.1, 0.9, 9), numticks=100)
    ax.xaxis.set_minor_locator(locminx)
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

    locmajy = matplotlib.ticker.LogLocator(base=10.0, numticks=20)
    ax.yaxis.set_major_locator(locmajy)
    
    locminy = matplotlib.ticker.LogLocator(base=10.0, subs=np.linspace(0.1, 0.9, 9), numticks=100)
    ax.yaxis.set_minor_locator(locminy)
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

    return ax


def choose_subplot_dimensions(k):
    if k < 2:
        return k, 1
    elif k < 11:
        return int(np.ceil(k/2)), 2
    else:
        # I've chosen to have a maximum of 3 columns
        return int(np.ceil(k/3)), 3


def generate_subplots(k, row_wise=True):
    nrow, ncol = choose_subplot_dimensions(k)
    # Choose your share X and share Y parameters as you wish:
    width, height = plt.rcParams.get('figure.figsize')
    fig, axes = plt.subplots(nrow, ncol, sharex=True, sharey=True, figsize = (ncol*width, nrow*height))
    fig.subplots_adjust(hspace=0, wspace=0)

    # Check if it's an array. If there's only one plot, it's just an Axes obj
    if not isinstance(axes, np.ndarray):
        return fig, [axes]
    else:
        # Choose the traversal you'd like: 'F' is col-wise, 'C' is row-wise
        axes = axes.flatten(order=('C' if row_wise else 'F'))

        # Delete any unused axes from the fig, so that they don't show
        # blank x- and y-axis lines
        for idx, ax in enumerate(axes[k:]):
            fig.delaxes(ax) 

            # Turn ticks on for the last ax in each column, wherever it lands
            idx_to_turn_on_ticks = idx + k - ncol if row_wise else idx + k - 1
            # for tk in axes[idx_to_turn_on_ticks].get_xticklabels():

            #     tk.set_visible(True)
            last_ax = axes[idx_to_turn_on_ticks]
            last_ax.xaxis.set_tick_params(which='both', labelbottom=True, labelleft = 'off')
            last_ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer = True, prune = 'lower'))

        axes = axes[:k]
        return fig, axes

# def MultiCol(ax, data, *args, **kwargs):

#     for m, row in enumerate(ax):
#         for n, col in enumerate(row):
            

# if __name__ == "__main__":

#     fig, ax = plt.subplots()

#     x = np.logspace(1, 10, 100)
#     y = x/(1 + x**2)

#     p = loglog(ax, x, y)
#     plt.show()

def label_line(ax, data, label, x, y, halign='right', valign='center', xshift=0, yshift=0, txt_col='black', size=6):
    """Add a label to a line, at the proper angle.

    Arguments
    ---------
    line : matplotlib.lines.Line2D object,
    label : str
    x : float
        x-position to place center of text (in data coordinated
    y : float
        y-position to place center of text (in data coordinates)
    color : str
    size : float
    """
    xdata, ydata = data[0], data[1]
    x1 = data[0, np.where(data[0] == x)[0][0]]
    x2 = data[0, np.where(data[0] == x)[0][0] + 1]
    y1 = data[1, np.where(data[1] == y)[0][0]]
    y2 = data[1, np.where(data[1] == y)[0][0] + 1]

    text = ax.annotate(label, xy=(x, y), xytext=(xshift, yshift),
                       textcoords='offset points',
                       size=size,
                       horizontalalignment=halign,
                       verticalalignment=valign,
                       color=txt_col)

    sp1 = ax.transData.transform_point((x1, y1))
    sp2 = ax.transData.transform_point((x2, y2))

    rise = (sp2[1] - sp1[1])
    run = (sp2[0] - sp1[0])

    slope_degrees = np.degrees(np.arctan2(rise, run))
    text.set_rotation(slope_degrees)
    return text
