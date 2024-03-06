def RGB(R,G,B):
    return (R/255.,G/255.,B/255.)


colors = {
        'red' : RGB(217,25,32),
        'blue' : RGB(32,123,193),
        'green' : RGB(0,159,76),
        'yellow' : RGB(251,200,52),
        'black' : RGB(0,0,0),
        'brown' : RGB(149,89,28),
        'white' : RGB(255,255,255),
        'grey': RGB(145,143,144),
        'orange' : RGB(225,123,16),
        }

c_list = ['red', 'blue','brown','yellow','orange','green','black']
style = ['o','+','s','.','x']
line_style = ['+-','o-','x-']