"""
Utilities for making literature review figures.

TODO:
- Make sure the data will be loaded correctly wherever the code is run from.
- Write function that makes sure the DataFrame is fine.
"""

import re
import warnings
import os
from collections import OrderedDict

import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu, kruskal, pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns


dirname = os.path.dirname(__file__)
repo_root = os.path.join(dirname, '../')


def lstrip(list_of_strs, lower=True):
    """Remove left space and make lowercase."""
    return [a.lstrip().lower() if lower else a.lstrip() for a in list_of_strs] 

    
def replace_nans_in_column(df, column_name, replace_by=' '):
    nan_ind = df[column_name].apply(lambda x:
        np.isnan(x) if isinstance(x, float) else False)
    df.loc[nan_ind, column_name] = replace_by
    return df


def tex_escape(text):
    """Add escape character in front of LaTeX special characters in string.

        :param text: a plain text message
        :return: the message escaped to appear correctly in LaTeX
        
        From https://stackoverflow.com/a/25875504
    """
    conv = {
        '&': r'\&',
        '%': r'\%',
        '$': r'\$',
        '#': r'\#',
        '_': r'\_',
        '{': r'\{',
        '}': r'\}',
        '~': r'\textasciitilde{}',
        '^': r'\^{}',
        '\\': r'\textbackslash{}',
        '<': r'\textless{}',
        '>': r'\textgreater{}',
    }
    regex = re.compile('|'.join(re.escape(str(key)) 
        for key in sorted(conv.keys(), key = lambda item: - len(item))))
    return regex.sub(lambda match: conv[match.group()], text)


def split_column_with_multiple_entries(df, col, ref_col='Citation', sep=';\n', 
                                       lower=True, mismatch='drop'):
    """Split the content of a column that contains more than one value per cell.
    
    Split the content of cells that contain more than one value. Some cells 
    contain two or more values for a single data item, e.g., 
        
        Number of subjects: '15, 203, 23'
        
    A DataFrame where each row contains a single value per cell is returned.
    
    Args:
        df (pd.DataFrame)
        col (str or list of str): name of the column(s) to split.
        
    Keyword Args:
        ref_col (str or list of str): identifier column(s) to use to identify 
            the row of origin of a splitted value.
        sep (str): separator between multiple values
        lower (bool): if True, make all values lowercase
        mismatch (str): [NOT IMPLEMENTED YET]
            only applies if `col` is a list and the cells of the
            different columns do not contain the same number of elements.
            If `drop`, remove rows for which there is a missing value.
            If `fill`, fill missing values with NaNs.
        
    Returns:
        (pd.DataFrame)
    """
    if not isinstance(ref_col, list):
        ref_col = [ref_col]
            
    if isinstance(col, list):
        # Find rows for which there is a mismatch
        cell_counts = list()
        for c in col:
            cell_counts.append(df[c].str.split(sep).apply(len))
        cell_counts_df = pd.concat(cell_counts, axis=1)
        inds_to_remove = cell_counts_df.loc[
            cell_counts_df.apply(lambda x: min(x) != max(x), 1)].index
        warnings.warn('{} rows had incompatible numbers of elements in the '
              'columns of interest and were dropped:'.format(len(inds_to_remove)))
        if 'Citation' in ref_col:
            for i in inds_to_remove:
                warnings.warn('\t{}'.format(df.iloc[i].loc['Citation']))
        df = df.drop(inds_to_remove)
        
        # Aggregate split columns
        temp_df = list()
        for i, c in enumerate(col):
            inds = [c] + ref_col if i != 0 else c
            temp_df.append(
                split_column_with_multiple_entries(
                    df, c, ref_col=ref_col, sep=sep, lower=lower)[inds])
            
        return pd.concat(temp_df, axis=1)
    
    else:
        df['temp'] = df[col].str.split(sep).apply(lstrip, lower=lower)

        value_per_row = list()
        for i, items in df[[*ref_col, 'temp']].iterrows():
            for m in items['temp']:
                value_per_row.append([i, *items[ref_col].tolist(), m])

        df = df.drop(['temp'], axis=1)

        return pd.DataFrame(value_per_row, columns=['paper nb', *ref_col, col])


def extract_main_domains(df):
    """Create column with the main domains.

    The main domains were picked by looking at the data and going with what made
    sense (there is no clear rule for defining them).
    """
    main_domains = ['Epilepsy', 'Sleep', 'BCI', 'Affective', 'Cognitive', 
                    'Improvement of processing tools', 'Generation of data']
    domains_df = df[['Domain 1', 'Domain 2', 'Domain 3', 'Domain 4']]
    df['Main domain'] = [row[row.isin(main_domains)].values[0] 
        if any(row.isin(main_domains)) else 'Others' 
        for ind, row in domains_df.iterrows()]

    return df


def extract_ref_numbers_from_bbl(df, filename=None):
    """Extract reference numbers from .bbl file and add them to df.
    
    Args:
        df (pd.DataFrame): dataframe containing the data items
            spreadsheet.
    
    Keyword Args:
        filename (str): path to the .bbl file (created when compiling
            the main tex file).
            
    Returns:
        (pd.DataFrame): dataframe with new column 'ref_nb'.
    """
    filename = '../data/output.bbl'
    with open(filename, 'r', encoding = 'ISO-8859-1') as f:
        text = ''.join(f.readlines())

    ref_nbs = re.findall(r'\\bibitem\{(.*)\}', text)
    ref_dict = {ref: i + 1 for i, ref in enumerate(ref_nbs)}

    df['ref_nb'] = df['Citation'].apply(lambda x: '[{}]'.format(ref_dict[x]))

    return df


def load_data_items(start_year=2010):
    """Load data items table.

    TODO:
    - Normalize column names?
    - Double check all the required columns are there?
    """
    fname = repo_root + '/data/data_items.csv'
    df = pd.read_csv(fname, header=1)

    # A little cleaning up
    df = df.dropna(axis=0, how='all')
    df = df.dropna(axis=1, how='all', thresh=int(df.shape[0] * 0.1))
    df = df[df['Year'] >= start_year]

    # Remove retracted paper and Supplement
    df = df[df['Citation'] != 'Pramod2015']
    df = df[df['Type of paper'] != 'Supplement']

    df = extract_main_domains(df)
    # df = extract_ref_numbers_from_bbl(df)

    return df


def load_reported_results_data():
    """Load table of reported results (second tab on spreadsheet).
    """
    fname = repo_root + '/data/reporting_results.csv'
    df = pd.read_csv(fname, header=0)
    df = df.drop(columns=['Unnamed: 0', 'Title', 'Comment'])
    df['Result'] = pd.to_numeric(df['Result'], errors='coerce')
    df['Architecture'] = df['Architecture'].fillna('-')
    df = df.dropna()

    def extract_model_type(x):
        if 'arch' in x:
            out = 'Proposed'
        elif 'trad' in x:
            out = 'Baseline (traditional)'
        elif 'dl' in x:
            out = 'Baseline (deep learning)'
        else:
            raise ValueError('Model type {} not supported.'.format(x))
        
        return out

    df['model_type'] = df['Model'].apply(extract_model_type)

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


def wrap_text(string, max_char=25):
    """Wrap string at `max_char` per line.

    Args:
        string (str): string to be wrapped.

    Keyword Args:
        max_char (int): maximum number of characters per line.

    Returns:
        (str): wrapped string.
    """
    string_parts = string.split()
    if len(string) > max_char and len(string_parts) > 1:
        out_string = string_parts[0]
        line_len = len(out_string)
        for i in string_parts[1:]:
            if line_len + 1 + len(i) > max_char:
                out_string += '\n'
                line_len = len(i)
            else:
                out_string += ' '
                line_len += len(i) + 1
            out_string += i
    else:
        out_string = string
        
    return out_string


def plot_multiple_proportions(data, height=0.3, print_count=True, 
                              respect_order=None, figsize=None, xlabel=None, 
                              ylabel=None, title=None):
    """Horizontal stacked bar plot for multiple proportions.

    Horizontal stacked bar plot used to display many simple proportions with
    potentially different categories.

    Args:
        data (dict): dictionary containing the different items, categories 
            and counts per item. E.g.,

            data = {'item1': {'cat1': 100, 'cat2': 56},
                    'item2': {'cat3': 60, 'cat4': 46, 'cat5': 50},
                    'item3': {'cat6': 50, 'cat7': 53, 'cat8': 53}}

    Keyword Args:
        height (float): height of bars
        print_count (bool or int): if True, print the count (number of elements)
            of each category in the middle of the bars. If False, don't print
            the counts. If provided as an int, it defines the smaller number 
            that will be printed on a bar (that way small numbers that wouldn't
            fit in a bar because it's too small won't be printed). 
        respect_order (list or None): if provided, the categories of each item 
            should respect the given order. E.g., `['Yes', 'No', 'N/M']` means
            that whenever the categories 'Yes', 'No' or 'N/M' are found for an
            item, they should appear in that order in the bar.
        figisize (tuple or None): size of the figure.
        xlabel (str or None): x-axis label.
        ylabel (str of None): y-axis label.

    Returns:
        (fig)
        (ax)
    """
    df = pd.DataFrame(data=list(data.keys()), columns=['items'])
    df['counts'] = np.zeros(len(data))
    df['items'] = df['items'].apply(wrap_text, max_char=20)

    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(x='counts', y='items', data=df, ax=ax)
    ax.set_ylabel('' if ylabel is None else ylabel)

    ylabels = ax.get_yticklabels()
    ax.set_yticklabels(ylabels, ha='right')

    ax.set_xlabel('Percentage (%)' if xlabel is None else xlabel)
    ax.set_xlim([0, 100])
    if title is not None:
        ax.set_title(title)

    for ind, (item, values) in enumerate(data.items()):
        bottom = 0
        n_values = sum(list(values.values()))
        ax.set_prop_cycle(None)  # reset color cycle
        bars = list()

        if respect_order is not None:
            ordered_values = OrderedDict()
            for ordered_cat in respect_order:
                if ordered_cat in values:
                    ordered_values[ordered_cat] = values.pop(ordered_cat)
            ordered_values.update(values)
            values = ordered_values

        for cat, val in values.items():
            width = val / n_values * 100
            bar = ax.barh(
                ind, width=width, height=height, left=bottom, label=cat)
            bars.append(bar)

            if (print_count is True) or (isinstance(print_count, int) and 
                                         val >= print_count):
                w = bar[0].get_width()
                ax.text(bottom + w / 2, ind, str(val), ha='center', va='center')

            bottom += width

        legend = plt.legend(handles=bars, bbox_to_anchor=(105, ind),
                            bbox_transform=ax.transData, loc='center left',
                            frameon=False)
        ax.add_artist(legend)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)

    plt.tight_layout()

    return fig, ax


def run_mannwhitneyu(df, condition_col, conditions, value_col='acc_diff',
                     min_n_obs=10, plot=False):
    """Run Mann-Whitney rank-sum test.

    Args:
        df (pd.DataFrame): dataframe where each row is a paper.
        condition_col (str): name of column to use as condition.
        conditions (list): list of two strings containing the values of the
            condition to compare.

    Keyword Args:
        value_col (str): name of column to use as the numerical value to run the
            test on.
        min_n_obs (int): minimum number of observations in each sample in order
            to run the test.

    Returns:
        (float): U statistic
        (float): p-value
    """
    assert len(conditions) == 2, '`conditions` must be of length 2, got {}'.format(
        len(conditions))
    data1 = df[df[condition_col] == conditions[0]][value_col]
    data2 = df[df[condition_col] == conditions[1]][value_col]

    if len(data1) >= min_n_obs and len(data2) >= min_n_obs:
        stat, p = mannwhitneyu(data1, data2)
    else:
        stat, p = np.nan, np.nan
        print('Not enough observations in each sample ({} and {}).'.format(
            len(data1), len(data2)))

    if plot:
        fig, ax = plt.subplots()
        sns.violinplot(
            data=df[df[condition_col].isin(conditions)], x=condition_col, 
            y=value_col, ax=ax)
        ax.set_title('Mann-Whitney for {} vs. {}\n(pvalue={:0.4f})'.format(
            condition_col, value_col, p))
    else:
        fig = None

    return {'test': 'mannwhitneyu', 'pvalue': p, 'stat': stat, 'fig': fig}


def run_kruskal(df, condition_col, value_col='acc_diff', min_n_obs=6, 
                plot=False):
    """Run Kruskal-Wallis analysis of variance test.

    Args:
        df (pd.DataFrame): dataframe where each row is a paper.
        condition_col (str): name of column to use as condition.

    Keyword Args:
        value_col (str): name of column to use as the numerical value to run the
            test on.
        min_n_obs (int): minimum number of observations in each sample in order
            to run the test.

    Returns:
        (float): U statistic
        (float): p-value
    """
    data = [i for name, i in df.groupby(condition_col)[value_col]
            if len(i) >= min_n_obs]

    if len(data) > 2:
        stat, p = kruskal(*data)
    else:
        stat, p = np.nan, np.nan
        print('Not enough samples with more than {} observations.'.format(min_n_obs))

    if plot:
        enough_samples = df[condition_col].value_counts() >= min_n_obs
        enough_samples = enough_samples.index[enough_samples].tolist()
        fig, ax = plt.subplots()
        sns.violinplot(
            data=df[df[condition_col].isin(enough_samples)], x=condition_col, 
            y=value_col, ax=ax)
        ax.set_title('Kruskal-Wallis for {} vs. {}\n(pvalue={:0.4f})'.format(
            condition_col, value_col, p))
    else:
        fig = None

    return {'test': 'kruskal', 'pvalue': p, 'stat': stat, 'fig': fig}


def run_spearmanr(df, condition_col, value_col='acc_diff', log=False, 
                  plot=False):
    """Run Spearman's rank correlation analysis.

    Args:
        df (pd.DataFrame): dataframe where each row is a paper.
        condition_col (str): name of column to use as condition.

    Keyword Args:
        value_col (str): name of column to use as the numerical value to run the
            test on.
        log (bool): if True, use log of `condition_col` before computing the
            correlation.

    Returns:
        (float): U statistic
        (float): p-value
    """
    data1 = np.log10(df[condition_col]) if log else df[condition_col]
    data2 = df[value_col]
    corr, p = spearmanr(data1, data2)

    if plot:
        log_condition_col = 'log_' + condition_col
        df[log_condition_col] = np.log10(df[condition_col])
        fig, ax = plt.subplots()
        sns.regplot(data=df, x=log_condition_col, y=value_col, robust=True, ax=ax)
        ax.set_title('Spearman Rho for {} vs. {}\n(pvalue={:0.4f}, ρ={:0.4f})'.format(
            log_condition_col, value_col, p, corr))
    else:
        fig = None

    return {'test': 'spearmanr', 'pvalue': p, 'stat': corr, 'fig': fig}


def keep_single_valued_rows(df, condition_col, mult_str='\n', id_col='Citation'):
    """Keep rows for which a single value exist.

    This function filters a dataframe to keep only rows where the column 
    `condition_col` does not contain the string `mult_str` (which would indicate
    multiple values).

    Args:
        df (pd.DataFrame): dataframe.
    
    Keyword Args:
        mult_str (str): string that indicates multiple values in a row.
        id_col (str): name of column to use to identify different rows.
True
    Returns:
        (pd.DataFrame): filtered dataframe
    """
    rows_with_multiple = df[df[condition_col].str.contains(mult_str)][id_col]
    return df[~df[id_col].isin(rows_with_multiple)]
