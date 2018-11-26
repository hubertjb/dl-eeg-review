"""
Utilities for making literature review figures.

TODO:
- Make a CLI for creating all figures at once.
- Make sure that the seaborn parameters will apply to pure matplotlib figures.
"""

import os
import re

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS
from graphviz import Digraph

import plt_config as cfg


sns.set_context(cfg.plotting_context)
sns.set_style(cfg.axes_styles)


def load_data_items(start_year=2012):
    """Load data items table.

    TODO:
    - Normalize column names?
    - Double check all the required columns are there?
    """
    fname = 'data/data_items.csv'
    df = pd.read_csv(fname, header=1)

    # A little cleaning up
    df = df.iloc[:195, :]
    df = df.dropna(axis=0, how='all')
    df = df.dropna(axis=1, how='all', thresh=int(df.shape[0] * 0.1))
    df = df[df['Year'] >= start_year]

    return df


def load_reported_results_data():
    """Load table of reported results (second tab on spreadsheet).
    """
    fname = 'data/reporting_results.csv'
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
    # assert(df['Year'].dropna(axis=0) > 2010)


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


def get_saturation(level, min_s, max_s, n_levels):
    return (min_s - max_s) / (n_levels - 1) * level + max_s


def get_font_size(n_papers, min_font, max_font, max_n_papers):
    return (max_font - min_font) / (max_n_papers - 1) * n_papers + min_font


def make_box(dot, text, max_char, n_instances, max_n_instances, level, n_levels, 
             min_sat, max_sat, min_font_size, max_font_size, parent_name, 
             counter=None, n_categories=None, hue=None, node_name=None):
    """Make graphviz box for tree graph.

    Args:
        dot (): graphviz Digraph object
        text (str): text to put in the box
        max_char (int): maximum number of characters on a line
        n_instances (int): number of instances (to be written under `text`)
        max_n_instances (int): maximum number of instances a box can have
        level (int): value from 0 to `n_levels`-1
        n_levels (int): number of levels in graph
        min_sat (float): minimum saturation value between [0, 1]
        max_sat (float): maximum saturation value between [0, 1]
        min_font_size (float): minimum font size
        max_font_size (float): maximum font size
        parent_name (str): name of parent node
    
    Keyword Args:
        counter (None or int): counter from 0 to `n_categories`-1. If None, use 
            the provided `hue`.
        n_categories (None or int): number of categories on that level. If None, 
            use the provided `hue`.
        hue (None or str): hue of the box. If None, compute it using `counter` 
            and `n_categories`.
        node_name (None or str): internal name of the node. If None, use `text` 
            as the internal node name.
        
    Returns:
        (str): node name
        (float): hue of the box
    """
    node_text = wrap_text(text, max_char=max_char)
    if node_name is None:
        node_name = text
    
    if hue is None:
        assert counter is not None
        assert n_categories is not None
        hue = (counter + 1) / n_categories
    fillcolor = '{} {} 1'.format(
        hue, get_saturation(level, min_sat, max_sat, n_levels))
    fontsize = str(get_font_size(
        n_instances, min_font_size, max_font_size, max_n_instances))

    dot.node(node_name, '{}\n({})'.format(node_text, n_instances), 
             fillcolor=fillcolor, fontsize=fontsize)
    dot.edge(parent_name, node_name)
    
    return node_name, hue


def plot_domain_tree(df, first_box='DL + EEG studies', min_font_size=6, 
                     max_font_size=12, max_char=16, min_n_items=2, 
                     save_cfg=cfg.saving_config):
    """Plot tree graph showing the breakdown of study domains.

    Args:
        df (pd.DataFrame): data items table

    Keyword Args:
        first_box (str): text of the first box
        min_font_size (int): minimum font size
        max_font_size (int): maximum font size
        max_char (int): maximum number of characters per line
        min_n_items (int): if a node has less than this number of elements, 
            put it inside a node called "Others".
        save_cfg (dict or None):
    
    Returns:
        (graphviz.Digraph): graphviz object

    TODO:
    - Restructure...

    NOTES:
    - To unflatten automatically, apply the following on the .dot file:
        >> unflatten -l 3 -c 10 domains | dot -Teps -o outfile.eps
    """
    df = df[['Domain 1', 'Domain 2', 'Domain 3', 'Domain 4']].copy()
    df = df[~df['Domain 1'].isnull()]

    n_samples, n_levels = df.shape
    format = save_cfg['format'] if isinstance(save_cfg, dict) else 'svg' 
    
    dot = Digraph(format=format)
    dot.attr('graph', rankdir='TB')  # LR (left to right), TB (top to bottom)
    dot.attr('node', fontname='Helvetica', fontsize=str(max_font_size), 
             shape='box', style='filled, rounded',  margin='0.2,0.01', 
             penwidth='0.5')
    dot.node('A', '{}\n({})'.format(first_box, len(df)), 
             fillcolor='azure')
    
    min_sat, max_sat = 0.05, 0.4
    
    sub_df = df['Domain 1'].value_counts()
    n_categories = len(sub_df)

    for i, (d1, count1) in enumerate(sub_df.iteritems()):
        node1, hue = make_box(dot, d1, max_char, count1, n_samples, 0, n_levels, 
                              min_sat, max_sat, min_font_size, max_font_size, 'A',
                              counter=i, n_categories=n_categories)
        
        for d2, count2 in df[df['Domain 1'] == d1]['Domain 2'].value_counts().iteritems():
            node2, _ = make_box(
                dot, d2, max_char, count2, n_samples, 1, n_levels, min_sat, 
                max_sat, min_font_size, max_font_size, node1, hue=hue)
            
            n_others3 = 0
            for d3, count3 in df[df['Domain 2'] == d2]['Domain 3'].value_counts().iteritems():
                if isinstance(d3, str) and d3 != 'TBD':
                    if count3 < min_n_items:
                        n_others3 += 1
                    else:
                        node3, _ = make_box(
                            dot, d3, max_char, count3, n_samples, 2, n_levels,
                            min_sat, max_sat, min_font_size, max_font_size, 
                            node2, hue=hue)

                        n_others4 = 0
                        for d4, count4 in df[df['Domain 3'] == d3]['Domain 4'].value_counts().iteritems():
                            if isinstance(d4, str) and d4 != 'TBD':
                                if count4 < min_n_items:
                                    n_others4 += 1
                                else:
                                    make_box(
                                        dot, d4, max_char, count4, n_samples, 3, 
                                        n_levels, min_sat, max_sat, min_font_size, 
                                        max_font_size, node3, hue=hue)

                        if n_others4 > 0:
                            make_box(
                                dot, 'Others', max_char, n_others4, n_samples, 
                                3, n_levels, min_sat, max_sat, min_font_size, 
                                max_font_size, node3, hue=hue, 
                                node_name=node3+'others')

            if n_others3 > 0:
                make_box(
                    dot, 'Others', max_char, n_others3, n_samples, 2, n_levels,
                    min_sat, max_sat, min_font_size, max_font_size, node2, hue=hue, 
                    node_name=node2+'others') 

    if save_cfg is not None:
        fname = os.path.join(save_cfg['savepath'], 'domains')  # + '.' + save_cfg['format']
        dot.render(filename=fname, cleanup=False)
                
    return dot


def plot_years(df, save_cfg=cfg.saving_config):
    fig, ax = plt.subplots()
    sns.distplot(df['Year'].dropna(axis=0), ax=ax)

    if save_cfg is not None:
        fname = os.path.join(save_cfg['savepath'], 'years')
        fig.savefig(fname + '.' + save_cfg['format'], **save_cfg)

    return ax


def plot_model_comparison(df, save_cfg=cfg.saving_config):
    """Plot bar graph showing the types of baseline models used.
    """
    fig, ax = plt.subplots()
    sns.countplot(df['Baseline model type'].dropna(axis=0), ax=ax)
    ax.set_ylabel('Number of papers')
    ax.set_xlabel('')

    if save_cfg is not None:
        fname = os.path.join(save_cfg['savepath'], 'model_comparison')
        fig.savefig(fname + '.' + save_cfg['format'], **save_cfg)

    return ax


def plot_performance_metrics(df, cutoff=3, eeg_clf=None, 
                             save_cfg=cfg.saving_config):
    """Plot bar graph showing the types of performance metrics used.

    Args:
        df (DataFrame)

    Keyword Args:
        cutoff (int): Metrics with less than this number of papers will be cut
            off from the bar graph.
        eeg_clf (bool): If True, only use studies that focus on EEG 
            classification. If False, only use studies that did not focus on 
            EEG classification. If None, use all studies.
        save_cfg (dict)

    Assumptions, simplifications:
    - Rates have been simplified (e.g., "false positive rate" -> "false positives")
    - RMSE and MSE have been merged under MSE
    - Training/testing times have been simplified to "time"
    - Macro f1-score === f1=score
    """
    def lstrip(list_of_strs):
        """Remove left space and make lowercase."""
        return [a.lstrip().lower() for a in list_of_strs] 
    
    if eeg_clf is True:
        metrics = df[df['Domain 1'] == 'Classification of EEG signals']['Performance metrics (clean)']
    elif eeg_clf is False:
        metrics = df[df['Domain 1'] != 'Classification of EEG signals']['Performance metrics (clean)']
    elif eeg_clf is None:
        metrics = df['Performance metrics (clean)']

    metrics = metrics.str.split(',').apply(lstrip)

    metric_per_article = list()
    for i, metric_list in metrics.iteritems():
        for m in metric_list:
            metric_per_article.append([i, m])

    metrics_df = pd.DataFrame(metric_per_article, columns=['paper nb', 'metric'])

    # Replace equivalent terms by standardized term
    equivalences = {'selectivity': 'specificity',
                    'true negative rate': 'specificity',
                    'sensitivitiy': 'sensitivity',
                    'sensitivy': 'sensitivity',
                    'recall': 'sensitivity',
                    'hit rate': 'sensitivity', 
                    'true positive rate': 'sensitivity',
                    'sensibility': 'sensitivity',
                    'positive predictive value': 'precision',
                    'f-measure': 'f1-score',
                    'f-score': 'f1-score',
                    'f1-measure': 'f1-score',
                    'macro f1-score': 'f1-score',
                    'macro-averaging f1-score': 'f1-score',
                    'kappa': 'cohen\'s kappa',
                    'mae': 'mean absolute error',
                    'false negative rate': 'false negatives',
                    'fpr': 'false positives',
                    'false positive rate': 'false positives',
                    'false prediction rate': 'false positives',
                    'roc curves': 'roc',
                    'rmse': 'mean squared error',
                    'mse': 'mean squared error',
                    'training time': 'time',
                    'testing time': 'time',
                    'test error': 'error'}
    metrics_df = metrics_df.replace(equivalences)

    # Removing low count categories
    metrics_counts = metrics_df['metric'].value_counts()
    metrics_df = metrics_df[metrics_df['metric'].isin(
        metrics_counts[(metrics_counts >= cutoff)].index)]

    fig, ax = plt.subplots()
    ax = sns.countplot(y='metric', data=metrics_df, 
                       order=metrics_df['metric'].value_counts().index)
    ax.set_xlabel('Number of papers')
    ax.set_ylabel('')
    plt.tight_layout()

    if save_cfg is not None:
        savename = 'performance_metrics'
        if eeg_clf is True:
            savename += '_eeg_clf'
        elif eeg_clf is False:
            savename += '_not_eeg_clf'
        fname = os.path.join(save_cfg['savepath'], savename)
        fig.savefig(fname + '.' + save_cfg['format'], **save_cfg)

    return ax


def plot_reported_results(df, data_items_df=None, save_cfg=cfg.saving_config):
    """Plot figures to described the reported results in the studies.

    Args:
        df (DataFrame): contains reported results (second tab in spreadsheet)

    Keyword Args:
        data_items_df (DataFrame): contains data items (first tab in spreadsheet)
        save_cfg (dict)

    Returns:
        (list): list of axes to created figures
    """
    acc_df = df[df['Metric'] == 'accuracy']  # Extract accuracy rows only

    # Create new column that contains both citation and task information
    acc_df['citation_task'] = acc_df[['Citation', 'Task']].apply(
        lambda x: ' ['.join(x) + ']', axis=1)

    # Create a new column with the year
    acc_df['year'] = acc_df['Citation'].apply(
        lambda x: int(x[x.find('2'):x.find('2') + 4]))

    # Order by average proposed model accuracy
    acc_ind = acc_df[acc_df['model_type']=='Proposed'].groupby(
        'Citation').mean().sort_values(by='Result').index
    acc_df['Citation'] = acc_df['Citation'].astype('category')
    acc_df['Citation'].cat.set_categories(acc_ind, inplace=True)
    acc_df = acc_df.sort_values(['Citation'])

    # Only keep 2 best per task and model type
    acc2_df = acc_df.sort_values(
        ['Citation', 'Task', 'model_type', 'Result'], ascending=True).groupby(
            ['Citation', 'Task', 'model_type']).tail(2)

    axes = list()
    axes.append(_plot_results_per_citation_task(acc2_df, save_cfg))

    # Only keep the maximum accuracy per citation & task
    best_df = acc_df.groupby(
        ['Citation', 'Task', 'model_type'])['Result'].max().reset_index()

    # Only keep citations/tasks that have a traditional baseline
    best_df = best_df.groupby(['Citation', 'Task']).filter(
        lambda x: 'Baseline (traditional)' in x.values).reset_index()

    # Compute difference between proposed and traditional baseline
    diff_df = best_df.groupby(['Citation', 'Task']).apply(
                lambda x: x[x['model_type'] == 'Proposed']['Result'].iloc[0] - \
                          x[x['model_type'] == 'Baseline (traditional)'][
                              'Result'].iloc[0]).reset_index()
    diff_df = diff_df.rename(columns={0: 'acc_diff'})

    axes.append(_plot_results_accuracy_diff_scatter(diff_df, save_cfg))
    axes.append(_plot_results_accuracy_diff_distr(diff_df, save_cfg))

    # Pivot dataframe to plot proposed vs. baseline accuracy as a scatterplot
    best_df['citation_task'] = best_df[['Citation', 'Task']].apply(
        lambda x: ' ['.join(x) + ']', axis=1)
    acc_comparison_df = best_df.pivot(
        index='citation_task', columns='model_type', values='Result')

    axes.append(_plot_results_accuracy_comparison(acc_comparison_df, save_cfg))

    if data_items_df is not None:
        domains_df = data_items_df.filter(regex='(?=Domain*|Citation)')

        # Concatenate domains into one string
        def concat_domains(x):
            domain = ''
            for i in x[1:]:
                if isinstance(i, str):
                    domain += i + '/'
            return domain[:-1]

        domains_df['domain'] = data_items_df.filter(regex='(?=Domain*)').apply(
                                                    concat_domains, axis=1)
        diff_domain_df = diff_df.merge(domains_df, on='Citation', how='left')
        diff_domain_df = diff_domain_df.sort_values(by='domain')

        axes.append(_plot_results_accuracy_per_domain(
            diff_domain_df, diff_df, save_cfg))

    return axes    


def _plot_results_per_citation_task(results_df, save_cfg):
    """Plot scatter plot of accuracy for each condition and task.
    """
    figsize = plt.rcParams.get('figure.figsize')
    fig, ax = plt.subplots(figsize=(figsize[0], figsize[1] * 4))
    # Need to make the graph taller otherwise the y axis labels are on top of
    # each other.
    sns.catplot(y='citation_task', x='Result', hue='model_type', data=results_df, 
                ax=ax)
    ax.set_xlabel('accuracy')
    ax.set_ylabel('')
    plt.tight_layout()

    if save_cfg is not None:
        savename = 'reported_results'
        fname = os.path.join(save_cfg['savepath'], savename)
        fig.savefig(fname + '.' + save_cfg['format'], **save_cfg)

    return ax


def _plot_results_accuracy_diff_scatter(results_df, save_cfg):
    """Plot difference in accuracy for each condition/task as a scatter plot.
    """
    figsize = plt.rcParams.get('figure.figsize')
    fig, ax = plt.subplots(figsize=(figsize[0], figsize[1] * 2))
    sns.catplot(y='Task', x='acc_diff', data=results_df, ax=ax)
    ax.set_xlabel('Accuracy difference')
    ax.set_ylabel('')
    ax.axvline(0, c='k', alpha=0.2)

    if save_cfg is not None:
        savename = 'reported_accuracy_diff_scatter'
        fname = os.path.join(save_cfg['savepath'], savename)
        fig.savefig(fname + '.' + save_cfg['format'], **save_cfg)

    return ax


def _plot_results_accuracy_diff_distr(results_df, save_cfg):
    """Plot the distribution of difference in accuracy.
    """
    fig, ax = plt.subplots()
    sns.distplot(results_df['acc_diff'], kde=False, rug=True, ax=ax)
    ax.set_xlabel('Accuracy difference')
    ax.set_ylabel('Number of studies')

    if save_cfg is not None:
        savename = 'reported_accuracy_diff_distr'
        fname = os.path.join(save_cfg['savepath'], savename)
        fig.savefig(fname + '.' + save_cfg['format'], **save_cfg)

    return ax


def _plot_results_accuracy_comparison(results_df, save_cfg):
    """Plot the comparison between the best model and best baseline.
    """
    fig, ax = plt.subplots()
    sns.scatterplot(data=results_df, x='Baseline (traditional)', y='Proposed', 
                    ax=ax)
    ax.plot([0, 1.1], [0, 1.1], c='k', alpha=0.2)
    plt.axis('square')
    ax.set_xlim([0, 1.1])
    ax.set_ylim([0, 1.1])

    if save_cfg is not None:
        savename = 'reported_accuracy_comparison'
        fname = os.path.join(save_cfg['savepath'], savename)
        fig.savefig(fname + '.' + save_cfg['format'], **save_cfg)

    return ax


def _plot_results_accuracy_per_domain(results_df, diff_df, save_cfg):
    """
    """
    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(6, 8), 
                             gridspec_kw = {'height_ratios':[5, 1]})

    sns.catplot(y='domain', x='acc_diff', size=3, jitter=True, 
                data=results_df, ax=axes[0])
    axes[0].set_xlabel('')
    axes[0].set_ylabel('')
    axes[0].axvline(0, c='k', alpha=0.2)

    sns.boxplot(x='acc_diff', data=diff_df, ax=axes[1])
    sns.swarmplot(x='acc_diff', data=diff_df, color="0", size=3, ax=axes[1])
    axes[1].axvline(0, c='k', alpha=0.2)
    axes[1].set_xlabel('Accuracy difference')

    if save_cfg is not None:
        savename = 'reported_accuracy_per_domain'
        fname = os.path.join(save_cfg['savepath'], savename)
        fig.savefig(fname + '.' + save_cfg['format'], **save_cfg)

    return axes


def generate_wordcloud(df, save_cfg=cfg.saving_config):
    brain_mask = np.array(Image.open("./img/brain_stencil.png"))

    def transform_format(val):
        if val == 0:
            return 255
        else:
            return val

    text = (df['Title']).to_string()

    stopwords = set(STOPWORDS)
    stopwords.add("using")
    stopwords.add("based")

    wc = WordCloud(background_color="white", max_words=2000, max_font_size=50, mask=brain_mask,
                   stopwords=stopwords, contour_width=1, contour_color='steelblue')

    # generate word cloud
    wc.generate(text)

    # store to file
    if save_cfg is not None:
        fname = os.path.join(save_cfg['savepath'], 'DL-EEG_WordCloud')
        wc.to_file(fname + '.' + save_cfg['format']) #, **save_cfg)

    # plt.figure()
    # plt.imshow(wc, interpolation="bilinear")
    # plt.axis("off")
    # plt.show()


if __name__ == '__main__':

    df = load_data_items()
    check_data_items(df)
    plot_years(df)
    plot_domain_tree(df)
    plot_model_comparison(df)
    plot_performance_metrics(df)
    plot_performance_metrics(df, cutoff=1, eeg_clf=True)
    plot_performance_metrics(df, cutoff=1, eeg_clf=False)
    generate_wordcloud(df)

    results_df = load_reported_results_data()
    plot_reported_results(results_df, data_items_df=df)
