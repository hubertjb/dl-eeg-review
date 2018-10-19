"""
Utilities for making literature review figures.

TODO:
- Make a CLI for creating all figures at once.
- Make sure that the seaborn parameters will apply to pure matplotlib figures.
"""

import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import plt_config as cfg


sns.set_context(cfg.plotting_context)
sns.set_style(cfg.axes_styles)


def load_data_items():
    """Load data items table.

    TODO:
    - Normalize column names?
    - Double check all the required columns are there?
    """
    fname = 'data_items.csv'
    df = pd.read_csv(fname, header=1)

    # A little cleaning up
    df = df.iloc[:195, :]
    df = df.dropna(axis=0, how='all')
    df = df.dropna(axis=1, how='all', thresh=int(df.shape[0] * 0.1))

    return df


def check_data_items(df):
    """Check data items to make sure it contains the right stuff. 

    - Years
    - Number of layers
    - Domains
    - Checked by 2 people
    - Invasive

    TODO:
    - Should this be some kind of unit test?
    """
    pass
    # assert(df['Year'].dropna(axis=0) > 2010)


def plot_years(df, save_cfg=cfg.saving_config):
    fig, ax = plt.subplots()
    sns.distplot(df['Year'].dropna(axis=0), ax=ax)

    if save_cfg is not None:
        fname = os.path.join(save_cfg['savepath'], 'years')
        fig.savefig(fname + '.' + save_cfg['format'], **save_cfg)

    return ax


if __name__ == '__main__':

    df = load_data_items()
    check_data_items(df)
    plot_years(df)
