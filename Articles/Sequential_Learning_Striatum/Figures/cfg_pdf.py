import matplotlib
from Articles.Sequential_Learning_Striatum.Figures.cfg_color import colors, cm

matplotlib.rcParams.update({
    "font.family": "serif",  # use serif/main font for text elements
    "text.latex.preamble": "\n".join([
         r"\usepackage{amsfonts,amssymb,mathtools,amsmath,xcolor}",
         r"\def\eps{\varepsilon}",
         r"\newcommand{\E}{\ensuremath{\mathbb{E}}}",
         r"\renewcommand{\P}{\ensuremath{\mathbb{P}}}",
    ]),
})


matplotlib.rcParams['font.size'] = 8
matplotlib.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['text.usetex'] = False
matplotlib.rcParams['svg.fonttype'] = 'none'
matplotlib.rcParams['lines.linewidth'] = 1.
matplotlib.rcParams['axes.linewidth'] = 0.5

# set tick width

matplotlib.rcParams['xtick.major.size'] = 2
matplotlib.rcParams['xtick.major.width'] = 0.5
matplotlib.rcParams['xtick.minor.size'] = 1
matplotlib.rcParams['xtick.minor.width'] = 0.5
matplotlib.rcParams['ytick.major.size'] = 2
matplotlib.rcParams['ytick.major.width'] = 0.5
matplotlib.rcParams['ytick.minor.size'] = 1
matplotlib.rcParams['ytick.minor.width'] = 0.5

matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Arial'
matplotlib.rcParams['mathtext.it'] = 'Arial:italic'
matplotlib.rcParams['mathtext.bf'] = 'Arial:bold'

import Articles.Sequential_Learning_Striatum.Figures.FigureSequential as FigureSequential
import Articles.Sequential_Learning_Striatum.Figures.FigurePlot as FigurePlot

FigureSequential.latex = '$'
FigurePlot.latex = '$'

extension = 'pdf'
