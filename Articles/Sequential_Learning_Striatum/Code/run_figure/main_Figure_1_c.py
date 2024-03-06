import sys
import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

sys.path.insert(1, os.path.join(sys.path[0], '../../../../'))

from Articles.Sequential_Learning_Striatum.Figures.FigureSequential import set_blank_axis
import Articles.Sequential_Learning_Striatum.Figures.cfg_pdf as cfg

A_list = np.array([[1., -1.], [-1., 1.]])


def f_STDP(x, Apostpre_, Aprepost_):
    y = np.zeros(len(x))
    for i in np.arange(len(x)):
        if x[i] > 0:
            y[i] = Aprepost_ * np.exp(-x[i])
        else:
            y[i] = Apostpre_ * np.exp(x[i])
    return y


linestyle_list_figure = ['-', '--']

fig_y, fig_x = 6., 6.
top, bottom, left, right = 0., 0.5, 2., 1.
dict_margins = dict(top=1.-top/fig_x, bottom=bottom/fig_x, left=left/fig_y, right=1.-right/fig_y)
hspace = 0.
height_average = (fig_x - top - bottom - hspace) / 2.
dict_margins['height_ratios'] = [5. * height_average, 3. * height_average]
dict_margins['hspace'] = hspace / height_average
fig, ax = plt.subplots(2, 1, figsize=(fig_y * cfg.cm, fig_x * cfg.cm),
                       gridspec_kw=dict(**dict_margins))

set_blank_axis(ax[1])

ax[0].set_zorder(10)

ax[0].plot([-5., 5.], [0., 0.], linestyle='-', color=cfg.colors['black'], alpha=0.2)
ax[0].plot([0., 0.], [-1.2, 1.2], linestyle='-', color=cfg.colors['black'], alpha=0.2)
x_1 = np.linspace(0., 5., 1000)[1:]
x_2 = np.linspace(-5., 0., 1000)[:-1]
for k, (Apostpre, Aprepost) in enumerate(A_list):
    ax[0].plot(x_1, f_STDP(x_1, Apostpre, Aprepost), color=cfg.colors['dark purple'],
               linestyle=linestyle_list_figure[k])
    ax[0].plot(x_2, f_STDP(x_2, Apostpre, Aprepost), color=cfg.colors['orange'],
               linestyle=linestyle_list_figure[k])
ax[0].set_xlim(-5., 5.)
ax[0].patch.set_facecolor(cfg.colors['white'])
ax[0].set_yticks([-1., 0., 1.])
ax[0].set_ylim(-1.2, 2.)
ax[0].set_xticks([])
ax[0].set_xlabel(r'{}\Delta t=t_{{\rm post}}-t_{{\rm pre}}{}'.format(cfg.FigureSequential.latex,
                                                                     cfg.FigureSequential.latex))
ax[0].set_ylabel(r'{}\Delta W{}'.format(cfg.FigureSequential.latex, cfg.FigureSequential.latex), labelpad=-2)
ax[0].spines['right'].set_visible(False)
ax[0].spines['top'].set_visible(False)

legend_elements_A = [Line2D([0], [0], color=cfg.colors['orange'], lw=0.5, linestyle='-',
                     label=r'{}A_{{\rm post-pre}}{}'.format(cfg.FigureSequential.latex, cfg.FigureSequential.latex)),
                     Line2D([0], [0], color=cfg.colors['dark purple'], lw=0.5, linestyle='-',
                     label=r'{}A_{{\rm pre-post}}{}'.format(cfg.FigureSequential.latex, cfg.FigureSequential.latex))
                     ]

legend_elements_B = [Line2D([0], [0], color=cfg.colors['black'], lw=0.5, linestyle=linestyle_list_figure[k],
                     label='{}({:.0f}, {:.0f}){}'.format(cfg.FigureSequential.latex, Apostpre,
                                                         Aprepost, cfg.FigureSequential.latex)) for k, (
                                                         Apostpre, Aprepost) in enumerate(A_list)]

ax[0].legend(handles=legend_elements_A, loc=9, fontsize=6, ncol=1,
             frameon=False, handlelength=1.)

current_legend = ax[1].legend(handles=legend_elements_B, loc=8, fontsize=6, ncol=2,
                              title=r'with {}(A_{{\rm post-pre}}, A_{{\rm pre-post}}){}'.format(
                                    cfg.FigureSequential.latex, cfg.FigureSequential.latex),
                              frameon=False, handlelength=2., title_fontsize=6)
plt.setp(current_legend.get_title(), multialignment='center')

ax[1].text(0.5, 0.6,
           r'{}\Delta W=A_{{\rm post-pre}}'.format(
            cfg.FigureSequential.latex)
           + r'\exp(\Delta t/\tau_{{\rm s}}),\,\Delta t {{<}} 0{}'.format(
            cfg.FigureSequential.latex) + '\n'
           + r'{}A_{{\rm pre-post}}'.format(
            cfg.FigureSequential.latex)
           + r'\exp(-\Delta t/\tau_{{\rm s}}),\, \Delta t {{>}} 0{}'.format(
            cfg.FigureSequential.latex), fontsize=6, ha='center', va='center')
fig.savefig('Figures/article/Figure_1_c.png', dpi=1000)
