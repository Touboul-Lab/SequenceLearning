import numpy as np

def RGB(R, G, B):
    return R/255., G/255., B/255.


colors = {
        'red': RGB(249, 102, 94),
        'pale red': RGB(252, 216, 226),
        'blue': RGB(68, 119, 178),
        'green': RGB(3, 192, 60),
        'yellow': RGB(241, 219, 5),
        'black': RGB(0, 0, 0),
        'brown': RGB(206, 156, 111),
        'white': RGB(255, 255, 255),
        'grey': RGB(145, 143, 144),
        'pale grey': RGB(222, 221, 222),
        'orange': RGB(255, 152, 6),
        'dark purple': RGB(92, 29, 100),
        'light purple': RGB(202, 111, 214),
        'pale purple': RGB(177, 156, 217),
        'dark green': RGB(86, 189, 94),
        'light green': RGB(78, 211, 78),
        'dark blue': RGB(89, 129, 172),
        'light blue': RGB(173, 190, 213),
        'cyan': RGB(68, 160, 193),
        'green0': RGB(58, 206, 58),
        'green1': RGB(99, 216, 99),
        'green2': RGB(139, 226, 139),
        'green3': RGB(180, 236, 180),
        'purple0': RGB(132, 58, 206),
        'purple1': RGB(157, 99, 216),
        'purple2': RGB(193, 139, 226),
        'purple3': RGB(208, 180, 236),
        'brown0': RGB(131, 106, 83),
        'brown1': RGB(160, 131, 105),
        'brown2': RGB(180, 157, 137),
        'brown3': RGB(200, 183, 168)
        }

cm = 1/2.54
