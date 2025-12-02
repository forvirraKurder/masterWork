import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import numpy as np
import matplotlib.cm as cm

def plot_eigs(eigs):
    fig, ax = plt.subplots(1)
    sc = ax.scatter(eigs.real, eigs.imag)
    ax.axvline(0, color='k', linewidth=0.5)
    ax.axhline(0, color='k', linewidth=0.5)
    ax.grid(True)

    annot = ax.annotate("", xy=(0, 0), xytext=(20, 20), textcoords="offset points",
                        bbox=dict(boxstyle="round", fc="w"),
                        arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)





def plot_mode_shape(mode_shape, ax=None, normalize=False, xy0=np.empty(0), linewidth=2, auto_lim=False, colors=cm.get_cmap('Set1')):

    if not ax:
        ax = plt.subplot(111, projection='polar')
    if auto_lim:
        ax.set_rlim(0, max(abs(mode_shape)))

    if xy0.shape == (0,):
        xy0 = np.zeros_like(mode_shape)
    ax.axes.get_xaxis().set_major_formatter(NullFormatter())
    ax.axes.get_yaxis().set_major_formatter(NullFormatter())
    ax.grid(color=[0.85, 0.85, 0.85])
    # f_txt = ax.set_xlabel('f={0:.2f}'.format(f), color=cluster_color_list(), weight='bold', family='Times New Roman', )

    if normalize:
        mode_shape_max = mode_shape[np.argmax(np.abs(mode_shape))]
        if abs(mode_shape_max) > 0:
            mode_shape = mode_shape * np.exp(-1j * np.angle(mode_shape_max)) / np.abs(mode_shape_max)

    pl = []
    for i, (vec, xy0_) in enumerate(zip(mode_shape, xy0)):
        pl.append(ax.annotate("",
                              xy=(np.angle(vec), np.abs(vec)),
                              xytext=(np.angle(xy0_), np.abs(xy0_)),
                              arrowprops=dict(arrowstyle="->",
                                              #linewidth=linewidth,
                                              #linestyle=style_,
                                              color=colors(i),
                                              )))  # , headwidth=1, headlength = 1))

    return pl
