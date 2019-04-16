"""
Configuration for plotting and saving plots, logs and tables.

Plotting style (from seaborn's `sns.axes_style()`) and context 
(`sns.plotting_context`) configuration.
"""

axes_styles = {'axes.facecolor': 'white',
                'axes.edgecolor': '.15',
                'axes.grid': False,
                'axes.axisbelow': True,
                'axes.labelcolor': '.15',
                'figure.facecolor': 'white',
                'grid.color': '.8',
                'grid.linestyle': '-',
                'text.color': '.15',
                'xtick.color': '.15',
                'ytick.color': '.15',
                'xtick.direction': 'out',
                'ytick.direction': 'out',
                'lines.solid_capstyle': 'round',
                'patch.edgecolor': 'w',
                'image.cmap': 'rocket',
                'font.family': ['sans-serif'],
                'font.sans-serif': ['Liberation Sans'], 
                'patch.force_edgecolor': True,
                'xtick.bottom': True,
                'xtick.top': False,
                'ytick.left': False,
                'ytick.right': False,
                'axes.spines.left': True,
                'axes.spines.bottom': True,
                'axes.spines.right': True,
                'axes.spines.top': True}

font_size = 8  # default for paper: 9.600000000000001
smaller_font_size = 7 # default for paper: 8.8

plotting_context = {'font.size': font_size,
                    'axes.labelsize': font_size,
                    'axes.titlesize': font_size,
                    'xtick.labelsize': smaller_font_size,
                    'ytick.labelsize': smaller_font_size,
                    'legend.fontsize': smaller_font_size,
                    'axes.linewidth': 1.0,
                    'grid.linewidth': 0.8,
                    'lines.linewidth': 1.2000000000000002,
                    'lines.markersize': 4.800000000000001,
                    'patch.linewidth': 0.8,
                    'xtick.major.width': 1.0,
                    'ytick.major.width': 1.0,
                    'xtick.minor.width': 0.8,
                    'ytick.minor.width': 0.8,
                    'xtick.major.size': 4.800000000000001,
                    'ytick.major.size': 4.800000000000001,
                    'xtick.minor.size': 3.2,
                    'ytick.minor.size': 3.2}

saving_config = {'savepath': '../figs/',
                 'table_savepath': '../tables/',
                 'dpi': 350,
                 'facecolor': 'w',
                 'edgecolor': 'w',
                 'orientation': 'portrait',
                 'papertype': 'letter', 
                 'format': 'jpg',  #'eps',
                 'transparent': True, 
                 'bbox_inches': 'tight', 
                 'pad_inches': 0.1,
                 'frameon': None,
                 'metadata': None,
                 'page_width': 8.5,  # in inches
                 'page_height': 11,
                 'text_width': 6.5,
                 'text_height': 9}

palette = 'Paired'
# palette = 'cubehelix'
