"""
Utilities for making literature review figures.

TODO:
- Make sure the data will be loaded correctly wherever the code is run from.
- Write function that makes sure the DataFrame is fine.
"""

import re
import warnings

import pandas as pd
import numpy as np


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
    fname = '../data/data_items.csv'
    df = pd.read_csv(fname, header=1)

    # A little cleaning up
    df = df.dropna(axis=0, how='all')
    df = df.dropna(axis=1, how='all', thresh=int(df.shape[0] * 0.1))
    df = df[df['Year'] >= start_year]

    df = extract_main_domains(df)
    df = extract_ref_numbers_from_bbl(df)

    return df


def load_reported_results_data():
    """Load table of reported results (second tab on spreadsheet).
    """
    fname = '../data/reporting_results.csv'
    df = pd.read_csv(fname, header=0)
    df = df.drop(columns=['Unnamed: 0', 'Title', 'Comment'])
    df['Result'] = pd.to_numeric(df['Result'], errors='coerce')
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
