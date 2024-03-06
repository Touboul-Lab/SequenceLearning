import matplotlib as mpl
mpl.use('pgf')
import matplotlib
from Articles.Sequential_Learning_Striatum.Figures.cfg_color import colors, cm

matplotlib.rcParams.update({
    "font.family": "serif",
    "pgf.texsystem": "pdflatex",
    "pgf.preamble": "\n".join([
         r"\usepackage{amsfonts,amssymb,mathtools,amsmath,xcolor}",
         r"\def\eps{\varepsilon}",
         r"\newcommand{\E}{\ensuremath{\mathbb{E}}}",
         r"\renewcommand{\P}{\ensuremath{\mathbb{P}}}",
            r"\DeclareUnicodeCharacter{2212}{-}",
            r"\DeclareUnicodeCharacter{00D7}{$\times$}",
    ]),
})

matplotlib.rcParams['font.size'] = 8
matplotlib.rcParams['axes.titlesize'] = 8
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

import Articles.Sequential_Learning_Striatum.Figures.FigureSequential as FigureSequential
import Articles.Sequential_Learning_Striatum.Figures.FigurePlot as FigurePlot

FigureSequential.latex = '$'
FigurePlot.latex = '$'

extension = 'pgf'
