import math
import matplotlib
import matplotlib.ticker
import matplotlib.pyplot as plt
from matplotlib import text as mtext
import numpy as np
import re

# plt.style.use('mvstyle')

# class MyLocator(matplotlib.ticker.AutoMinorLocator):
#     def __init__(self, n=23):
#         super().__init__(n=n)


# class MyLogLocator(matplotlib.ticker.LogLocator):
#     def __init__(self, n=10):
#         super().__init__(base = 10, numticks=n)



# ###################################################

# matplotlib.ticker.AutoMinorLocator = MyLocator
# matplotlib.ticker.LogLocator = MyLogLocator


class MyLogFormatter(matplotlib.ticker.LogFormatterMathtext):
    def __call__(self, x, pos=None):
        # call the original LogFormatter
        rv = matplotlib.ticker.LogFormatterMathtext.__call__(self, x, pos)

        # check if we really use TeX
        if matplotlib.rcParams["text.usetex"]:
            # if we have the string ^{- there is a negative exponent
            # where the minus sign is replaced by the short hyphen
            rv = re.sub(r'\^\{-', r'^{\\text{-}', rv)

        return rv

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

    ax.yaxis.set_major_formatter(MyLogFormatter())
    ax.xaxis.set_major_formatter(MyLogFormatter())

    ymin, ymax = ax.get_ylim()
    ax.set_ylim(10**np.round(np.log10(ymin)), 10**np.round(np.log10(ymax)))

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



class CurvedText(mtext.Text):
    """
    A text object that follows an arbitrary curve.
    """
    def __init__(self, x, y, text, axes, **kwargs):
        super(CurvedText, self).__init__(x[0],y[0],' ', **kwargs)

        axes.add_artist(self)

        ##saving the curve:
        self.__x = x
        self.__y = y
        self.__zorder = self.get_zorder()

        ##creating the text objects
        self.__Characters = []
        for c in text:
            if c == ' ':
                ##make this an invisible 'a':
                t = mtext.Text(0,0,'a')
                t.set_alpha(0.0)
            else:
                t = mtext.Text(0,0,c, **kwargs)

            #resetting unnecessary arguments
            t.set_ha('center')
            t.set_rotation(0)
            t.set_zorder(self.__zorder +1)

            self.__Characters.append((c,t))
            axes.add_artist(t)


    ##overloading some member functions, to assure correct functionality
    ##on update
    def set_zorder(self, zorder):
        super(CurvedText, self).set_zorder(zorder)
        self.__zorder = self.get_zorder()
        for c,t in self.__Characters:
            t.set_zorder(self.__zorder+1)

    def draw(self, renderer, *args, **kwargs):
        """
        Overload of the Text.draw() function. Do not do
        do any drawing, but update the positions and rotation
        angles of self.__Characters.
        """
        self.update_positions(renderer)

    def update_positions(self,renderer):
        """
        Update positions and rotations of the individual text elements.
        """

        #preparations

        ##determining the aspect ratio:
        ##from https://stackoverflow.com/a/42014041/2454357

        ##data limits
        xlim = self.axes.get_xlim()
        ylim = self.axes.get_ylim()
        ## Axis size on figure
        figW, figH = self.axes.get_figure().get_size_inches()
        ## Ratio of display units
        _, _, w, h = self.axes.get_position().bounds
        ##final aspect ratio
        aspect = ((figW * w)/(figH * h))*(ylim[1]-ylim[0])/(xlim[1]-xlim[0])

        #points of the curve in figure coordinates:
        x_fig,y_fig = (
            np.array(l) for l in zip(*self.axes.transData.transform([
            (i,j) for i,j in zip(self.__x,self.__y)
            ]))
        )

        #point distances in figure coordinates
        x_fig_dist = (x_fig[1:]-x_fig[:-1])
        y_fig_dist = (y_fig[1:]-y_fig[:-1])
        r_fig_dist = np.sqrt(x_fig_dist**2+y_fig_dist**2)

        #arc length in figure coordinates
        l_fig = np.insert(np.cumsum(r_fig_dist),0,0)

        #angles in figure coordinates
        rads = np.arctan2((y_fig[1:] - y_fig[:-1]),(x_fig[1:] - x_fig[:-1]))
        degs = np.rad2deg(rads)


        rel_pos = 10
        for c,t in self.__Characters:
            #finding the width of c:
            t.set_rotation(0)
            t.set_va('center')
            bbox1  = t.get_window_extent(renderer=renderer)
            w = bbox1.width
            h = bbox1.height

            #ignore all letters that don't fit:
            if rel_pos+w/2 > l_fig[-1]:
                t.set_alpha(0.0)
                rel_pos += w
                continue

            elif c != ' ':
                t.set_alpha(1.0)

            #finding the two data points between which the horizontal
            #center point of the character will be situated
            #left and right indices:
            il = np.where(rel_pos+w/2 >= l_fig)[0][-1]
            ir = np.where(rel_pos+w/2 <= l_fig)[0][0]

            #if we exactly hit a data point:
            if ir == il:
                ir += 1

            #how much of the letter width was needed to find il:
            used = l_fig[il]-rel_pos
            rel_pos = l_fig[il]

            #relative distance between il and ir where the center
            #of the character will be
            fraction = (w/2-used)/r_fig_dist[il]

            ##setting the character position in data coordinates:
            ##interpolate between the two points:
            x = self.__x[il]+fraction*(self.__x[ir]-self.__x[il])
            y = self.__y[il]+fraction*(self.__y[ir]-self.__y[il])

            #getting the offset when setting correct vertical alignment
            #in data coordinates
            t.set_va(self.get_va())
            bbox2  = t.get_window_extent(renderer=renderer)

            bbox1d = self.axes.transData.inverted().transform(bbox1)
            bbox2d = self.axes.transData.inverted().transform(bbox2)
            dr = np.array(bbox2d[0]-bbox1d[0])

            #the rotation/stretch matrix
            rad = rads[il]
            rot_mat = np.array([
                [math.cos(rad), math.sin(rad)*aspect],
                [-math.sin(rad)/aspect, math.cos(rad)]
            ])

            ##computing the offset vector of the rotated character
            drp = np.dot(dr,rot_mat)

            #setting final position and rotation:
            t.set_position(np.array([x,y])+drp)
            t.set_rotation(degs[il])

            t.set_va('center')
            t.set_ha('center')

            #updating rel_pos to right edge of character
            rel_pos += w-used


def label_line(ax, data, label, x_pos_data_coord, halign='center', valign='bottom', xshift=0, yshift=0, rotn_adj=0, txt_col='black', size=10):

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
    data = np.array(data, dtype='float64')

    loc_point = np.abs(data[0] - x_pos_data_coord).argmin()

    x1 = np.log10(data[0][loc_point])
    x2 = np.log10(data[0][loc_point + 2])
    y1 = np.log10(data[1][loc_point])
    y2 = np.log10(data[1][loc_point + 2])

    text = ax.annotate(label, xy=(10**x1, 10**y1), xytext=(xshift, yshift),
                       textcoords='offset points',
                       size=size,
                       horizontalalignment=halign,
                       verticalalignment=valign,
                       color=txt_col,
                       transform=ax.transAxes)

    # sp1 = ax.transData.transform_point((x1, y1))
    # sp2 = ax.transData.transform_point((x2, y2))

    # rise = (sp2[1] - sp1[1])
    # run = (sp2[0] - sp1[0])

    rise = y2 - y1
    run  = x2 - x1

    slope_degrees = np.degrees(np.arctan2(rise, run))
    text.set_rotation(slope_degrees + rotn_adj)
    return text
