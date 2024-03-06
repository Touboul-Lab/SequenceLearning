import matplotlib
from Articles.Sequential_Learning_Striatum.Figures.cfg_color import colors, cm

matplotlib.rcParams['font.size'] = 8
matplotlib.rcParams['text.usetex'] = False
matplotlib.rcParams['svg.fonttype'] = 'none'
matplotlib.rcParams['lines.linewidth'] = 0.5
matplotlib.rcParams['axes.linewidth'] = 1.

# set tick width
matplotlib.rcParams['xtick.major.size'] = 2
matplotlib.rcParams['xtick.major.width'] = 0.5
matplotlib.rcParams['xtick.minor.size'] = 1
matplotlib.rcParams['xtick.minor.width'] = 0.5
matplotlib.rcParams['ytick.major.size'] = 2
matplotlib.rcParams['ytick.major.width'] = 0.5
matplotlib.rcParams['ytick.minor.size'] = 1
matplotlib.rcParams['ytick.minor.width'] = 0.5


import Articles.Sequential_Learning_Striatum.Figures.FigureSequential as FigureSequential
import Articles.Sequential_Learning_Striatum.Figures.FigurePlot as FigurePlot

FigureSequential.latex = '\$'
FigurePlot.latex = '\$'

extension = 'svg'
