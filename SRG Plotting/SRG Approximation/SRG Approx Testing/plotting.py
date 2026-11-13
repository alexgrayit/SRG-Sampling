import matplotlib.pyplot as plt

import platform
try:
    # Configure Matplotlib to use LaTeX font, makes graphs pretty :)
    if platform.system() == 'Linux':
        font_size = 14
        # If on a Unix system, use LaTeX
        plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Computer Modern Roman",
        'font.size': font_size,
        'text.latex.preamble' : r'\usepackage{amssymb} \usepackage{amsmath}'
        })
    else:
        font_size = 18
        # If on a Windows system, use the system font
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "Computer Modern Roman",
            'font.size': font_size,
            'text.latex.preamble' : r'\usepackage{amssymb} \usepackage{amsmath}'
        })
    
    
except:
    print("LaTeX not initialised")

# Set default grid formatting and minorticks for all plots
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.5
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.color'] = 'gray'
plt.rcParams['xtick.minor.visible'] = True
plt.rcParams['ytick.minor.visible'] = True
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
# plt.grid(which='minor', linestyle=':', linewidth='0.5', color='gray', alpha=0.5)