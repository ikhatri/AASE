# Style script adapted from Kyle Vedder
# https://github.com/kylevedder

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

def set_font(size):
    matplotlib.rcParams.update({# Use mathtext, not LaTeX
                            'text.usetex': False,
                            # Use the Computer modern font
                            'font.family': 'Times New Roman',
                            # 'font.serif': 'cmr10',
                            'font.size' : size,
                            'mathtext.fontset': 'cm',
                            # Use ASCII minus
                            'axes.unicode_minus': False,
                            })
linewidth = 0.5
minor_tick_color = (0.9, 0.9, 0.9)
kFontSize = 8

def setupfig(current_fig=None, halfsize=False, thirdsize=False, quartersize=False):
    if current_fig is None:
        plt.clf()
        fig = plt.gcf()
    else:
        fig = current_fig
    set_font(kFontSize)
    kScaleDown = 2.1
    kVerticalScale = 0.9
    halfsize_scale = 1
    if halfsize:
        halfsize_scale = 0.45
    if thirdsize:
        halfsize_scale = 0.30
    if quartersize:
        halfsize_scale = 0.19
    # fig.set_size_inches(7 /kScaleDown * 1.58 * halfsize_scale , 7.8 / kScaleDown / 1.61 * kVerticalScale, forward=True)
    fig.set_size_inches(7 , 1.25, forward=True)
    plt.gca().set_axisbelow(True)
    # change the color of the top and right spines to opaque gray
    plt.gca().spines['right'].set_color((.8,.8,.8))
    plt.gca().spines['top'].set_color((.8,.8,.8))

def grid(plt=plt):
    plt.grid(linewidth=linewidth/2)
    plt.grid(which='minor', color=minor_tick_color, linestyle='--', alpha=0.7, clip_on=True, linewidth=linewidth/4)

def color(count, total_elements):
    start = 0.2
    stop = 0.8
    cm_subsection = np.linspace(start, stop, total_elements)
    return [ matplotlib.cm.magma(x) for x in cm_subsection ][count]

def alpha(color, val=0.5):
    assert(type(color) == tuple)
    assert(len(color) == 4)
    r, g, b, a=color
    return (r, g, b, val)

def legend(loc, plt=plt):
    if type(loc)  is str:
        if loc == 'ul':
            loc = 2
        elif loc == 'br':
            loc = 0
        else:
            assert(False)
    leg = plt.legend(loc=loc, prop={'size': 4})
    # set the linewidth of each legend object
    for legobj in leg.legendHandles:
        legobj.set_linewidth(linewidth * 3)

def save_fig(filename):
    print("Saving figout/{}.*".format(filename))
    #plt.savefig("figout/{}.pgf".format(filename), bbox_inches='tight')
    #plt.savefig("figout/{}.png".format(filename), bbox_inches='tight', dpi=200)
    plt.savefig("figout/{}.pdf".format(filename), bbox_inches='tight', pad_inches=0)