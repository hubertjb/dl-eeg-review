"""Functions to plot and compute results for the literature review.
"""

import os
import logging
import logging.config
from collections import OrderedDict

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.patches as patches
import matplotlib as mpl
import seaborn as sns
import numpy as np
from PIL import Image
from wordcloud import WordCloud, STOPWORDS
from graphviz import Digraph
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci

import plt_config as cfg
import utils as ut


# Set style, context and palette
sns.set_style(rc=cfg.axes_styles)
sns.set_context(rc=cfg.plotting_context)
sns.set_palette(cfg.palette)

for key, val in cfg.axes_styles.items():
    mpl.rcParams[key] = val
for key, val in cfg.plotting_context.items():
    mpl.rcParams[key] = val


# Initialize logger for saving results and stats. Use `logger.info('message')`
# to log results.
logging.config.dictConfig({
    'version': 1,
    'disable_existing_loggers': True,
})
logger = logging.getLogger()
log_savename = os.path.join(cfg.saving_config['savepath'], 'results') + '.log'
handler = logging.FileHandler(log_savename, mode='w')
formatter = logging.Formatter(
        '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


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
    node_text = ut.wrap_text(text, max_char=max_char)
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


def plot_prisma_diagram(save_cfg=cfg.saving_config):
    """Plot diagram showing the number of selected articles.
    """
    # save_format = save_cfg['format'] if isinstance(save_cfg, dict) else 'svg'
    save_format = 'pdf'
    dot = Digraph(format=save_format)
    dot.attr('graph', rankdir='TB', overlap='false')
    dot.attr('node', fontname='Liberation Sans', fontsize=str(9), shape='box', 
             style='filled', margin='0.15,0.07', penwidth='0.5')

    fillcolor = 'gray98'

    dot.node('A', 'PubMed (n=39)\nGoogle Scholar (n=409)\narXiv (n=105)', 
             fillcolor='gray95')
    dot.node('B', 'Articles identified\nthrough database\nsearching\n(n=553)', 
             fillcolor=fillcolor)
    # dot.node('B2', 'Excluded\n(n=446)', fillcolor=fillcolor)
    dot.node('C', 'Articles after content\nscreening and\nduplicate removal\n(n=107) ', 
             fillcolor=fillcolor)
    dot.node('D', 'Articles included in\nthe analysis\n(n=156)', 
             fillcolor=fillcolor)
    dot.node('E', 'Additional articles\nidentified through\nbibliography search\n(n=49)', 
             fillcolor=fillcolor)

    dot.edge('B', 'C')
    # dot.edge('B', 'B2')
    dot.edge('C', 'D')
    dot.edge('E', 'D')

    # dot = Digraph('G', filename='hello.gv')
    # dot.edge('Hello', 'World')

    if save_cfg is not None:
        fname = os.path.join(save_cfg['savepath'], 'prisma_diagram')
        dot.render(filename=fname, view=False, cleanup=False)
                
    return dot


def plot_domain_tree(df, first_box='DL + EEG studies', min_font_size=10, 
                     max_font_size=14, max_char=16, min_n_items=2, 
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

    NOTES:
    - To unflatten automatically, apply the following on the .dot file:
        >> unflatten -l 3 -c 10 domains | dot -Teps -o domains_unflattened.eps
    - To produce a circular version instead (uses space more efficiently), use
        >> neato -Tps domains -o domains_neato.eps
    """
    df = df[['Domain 1', 'Domain 2', 'Domain 3', 'Domain 4']].copy()
    df = df[~df['Domain 1'].isnull()]

    n_samples, n_levels = df.shape
    format = save_cfg['format'] if isinstance(save_cfg, dict) else 'svg'
    size = '{},{}!'.format(save_cfg['page_width'], save_cfg['page_height'])
    
    dot = Digraph(format=format)
    dot.attr('graph', rankdir='TB', overlap='false', ratio='fill', size=size)  # LR (left to right), TB (top to bottom)
    dot.attr('node', fontname='Liberation Sans', fontsize=str(max_font_size), 
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
        fname = os.path.join(save_cfg['savepath'], 'dom_domains_tree')
        dot.render(filename=fname, cleanup=False)
                
    return dot


def plot_model_comparison(df, save_cfg=cfg.saving_config):
    """Plot bar graph showing the types of baseline models used.
    """
    fig, ax = plt.subplots(figsize=(save_cfg['text_width'] / 4 * 2, 
                                    save_cfg['text_height'] / 5))
    sns.countplot(y=df['Baseline model type'].dropna(axis=0), ax=ax)
    ax.set_xlabel('Number of papers')
    ax.set_ylabel('')
    plt.tight_layout()

    model_prcts = df['Baseline model type'].value_counts() / df.shape[0] * 100
    logger.info('% of studies that used at least one traditional baseline: {}'.format(
        model_prcts['Traditional pipeline'] + model_prcts['DL & Trad.']))
    logger.info('% of studies that used at least one deep learning baseline: {}'.format(
        model_prcts['DL'] + model_prcts['DL & Trad.']))
    logger.info('% of studies that did not report baseline comparisons: {}'.format(
        model_prcts['None']))

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
    if eeg_clf is True:
        metrics = df[df['Domain 1'] == 'Classification of EEG signals'][
            'Performance metrics (clean)']
    elif eeg_clf is False:
        metrics = df[df['Domain 1'] != 'Classification of EEG signals'][
            'Performance metrics (clean)']
    elif eeg_clf is None:
        metrics = df['Performance metrics (clean)']

    metrics = metrics.str.split(',').apply(ut.lstrip)

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

    fig, ax = plt.subplots(figsize=(save_cfg['text_width'] / 2, 
                                    save_cfg['text_height'] / 5))
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
        domains_df = data_items_df.filter(
            regex='(?=Domain*|Citation|Main domain|Journal / Origin)')

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
        diff_domain_df['arxiv'] = diff_domain_df['Journal / Origin'] == 'Arxiv'

        axes.append(_plot_results_accuracy_per_domain(
            diff_domain_df, diff_df, save_cfg))

    return axes    


def _plot_results_per_citation_task(results_df, save_cfg):
    """Plot scatter plot of accuracy for each condition and task.
    """
    fig, ax = plt.subplots(figsize=(save_cfg['text_width'], 
                                    save_cfg['text_height'] * 1.3))
    # figsize = plt.rcParams.get('figure.figsize')
    # fig, ax = plt.subplots(figsize=(figsize[0], figsize[1] * 4))
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
    fig, ax = plt.subplots(figsize=(save_cfg['text_width'], 
                                    save_cfg['text_height'] * 1.3))
    # figsize = plt.rcParams.get('figure.figsize')
    # fig, ax = plt.subplots(figsize=(figsize[0], figsize[1] * 2))
    sns.catplot(y='Task', x='acc_diff', data=results_df, ax=ax)
    ax.set_xlabel('Accuracy difference')
    ax.set_ylabel('')
    ax.axvline(0, c='k', alpha=0.2)
    plt.tight_layout()

    if save_cfg is not None:
        savename = 'reported_accuracy_diff_scatter'
        fname = os.path.join(save_cfg['savepath'], savename)
        fig.savefig(fname + '.' + save_cfg['format'], **save_cfg)

    return ax


def _plot_results_accuracy_diff_distr(results_df, save_cfg):
    """Plot the distribution of difference in accuracy.
    """
    fig, ax = plt.subplots(figsize=(save_cfg['text_width'], 
                                    save_cfg['text_height'] * 0.5))
    sns.distplot(results_df['acc_diff'], kde=False, rug=True, ax=ax)
    ax.set_xlabel('Accuracy difference')
    ax.set_ylabel('Number of studies')
    plt.tight_layout()

    if save_cfg is not None:
        savename = 'reported_accuracy_diff_distr'
        fname = os.path.join(save_cfg['savepath'], savename)
        fig.savefig(fname + '.' + save_cfg['format'], **save_cfg)

    return ax


def _plot_results_accuracy_comparison(results_df, save_cfg):
    """Plot the comparison between the best model and best baseline.
    """
    fig, ax = plt.subplots(figsize=(save_cfg['text_width'], 
                                    save_cfg['text_height'] * 0.5))
    sns.scatterplot(data=results_df, x='Baseline (traditional)', y='Proposed', 
                    ax=ax)
    ax.plot([0, 1.1], [0, 1.1], c='k', alpha=0.2)
    plt.axis('square')
    ax.set_xlim([0, 1.1])
    ax.set_ylim([0, 1.1])
    plt.tight_layout()

    if save_cfg is not None:
        savename = 'reported_accuracy_comparison'
        fname = os.path.join(save_cfg['savepath'], savename)
        fig.savefig(fname + '.' + save_cfg['format'], **save_cfg)

    return ax


def _plot_results_accuracy_per_domain(results_df, diff_df, save_cfg):
    """Make scatterplot + boxplot to show accuracy difference by domain.
    """
    fig, axes = plt.subplots(
        nrows=2, ncols=1, sharex=True, 
        figsize=(save_cfg['text_width'], save_cfg['text_height'] / 3), 
        gridspec_kw = {'height_ratios':[5, 1]})

    results_df['Main domain'] = results_df['Main domain'].apply(
        ut.wrap_text, max_char=20)

    sns.catplot(y='Main domain', x='acc_diff', s=3, jitter=True, 
                data=results_df, ax=axes[0])
    axes[0].set_xlabel('')
    axes[0].set_ylabel('')
    axes[0].axvline(0, c='k', alpha=0.2)

    sns.boxplot(x='acc_diff', data=diff_df, ax=axes[1])
    sns.swarmplot(x='acc_diff', data=diff_df, color="0", size=2, ax=axes[1])
    axes[1].axvline(0, c='k', alpha=0.2)
    axes[1].set_xlabel('Accuracy difference')

    fig.subplots_adjust(wspace=0, hspace=0.02)
    plt.tight_layout()

    logger.info('Number of studies included in the accuracy improvement analysis: {}'.format(
        results_df.shape[0]))
    median = diff_df['acc_diff'].median()
    iqr = diff_df['acc_diff'].quantile(.75) - diff_df['acc_diff'].quantile(.25)
    logger.info('Median gain in accuracy: {:.6f}'.format(median))
    logger.info('Interquartile range of the gain in accuracy: {:.6f}'.format(iqr))
    best_improvement = diff_df.nlargest(1, 'acc_diff')
    logger.info('Best improvement in accuracy: {}, in {}'.format(
        best_improvement['acc_diff'].values[0], 
        best_improvement['Citation'].values[0]))

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


def plot_model_inspection_and_table(df, cutoff=1, save_cfg=cfg.saving_config):
    """Make bar graph and table listing method inspection techniques.

    Args:
        df (DataFrame)

    Keyword Args:
        cutoff (int): Metrics with less than this number of papers will be cut
            off from the bar graph.
        save_cfg (dict)
    """
    df['inspection_list'] = df[
        'Model inspection (clean)'].str.split(',').apply(ut.lstrip)

    inspection_per_article = list()
    for i, items in df[['Citation', 'inspection_list']].iterrows():
        for m in items['inspection_list']:
            inspection_per_article.append([i, items['Citation'], m])
            
    inspection_df = pd.DataFrame(
        inspection_per_article, 
        columns=['paper nb', 'Citation', 'inspection method'])

    # Remove "no" entries, because they make it really hard to see the 
    # actual distribution
    n_nos = inspection_df['inspection method'].value_counts()['no']
    n_papers = inspection_df.shape[0]
    logger.info('Number of papers without model inspection method: {}'.format(n_nos))
    inspection_df = inspection_df[inspection_df['inspection method'] != 'no']

    # # Replace "no" by "None"
    # inspection_df['inspection method'][
    #     inspection_df['inspection method'] == 'no'] = 'None'

    # Removing low count categories
    inspection_counts = inspection_df['inspection method'].value_counts()
    inspection_df = inspection_df[inspection_df['inspection method'].isin(
        inspection_counts[(inspection_counts >= cutoff)].index)]
    
    inspection_df['inspection method'] = inspection_df['inspection method'].apply(
        lambda x: x.capitalize())
    print(inspection_df['inspection method'])

    # Making table
    inspection_table = inspection_df.groupby(['inspection method'])[
        'Citation'].apply(list)
    order = inspection_df['inspection method'].value_counts().index
    inspection_table = inspection_table.reindex(order)
    inspection_table = inspection_table.apply(lambda x: r'\cite{' + ', '.join(x) + '}')

    with open(os.path.join(save_cfg['savepath'], 'inspection_methods.tex'), 'w') as f:
        with pd.option_context("max_colwidth", 1000):
            f.write(inspection_table.to_latex(escape=False))


    fig, ax = plt.subplots(figsize=(save_cfg['text_width'] / 4 * 3, 
                                    save_cfg['text_height'] / 2))
    ax = sns.countplot(y='inspection method', data=inspection_df, 
                    order=inspection_df['inspection method'].value_counts().index)
    ax.set_xlabel('Number of papers')
    ax.set_ylabel('')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()

    logger.info('% of studies that used model inspection techniques: {}'.format(
        100 - 100 * (n_nos / n_papers)))

    if save_cfg is not None:
        savename = 'model_inspection'
        fname = os.path.join(save_cfg['savepath'], savename)
        fig.savefig(fname + '.' + save_cfg['format'], **save_cfg)

    return ax


def plot_type_of_paper(df, save_cfg=cfg.saving_config):
    """Plot bar graph showing the type of each paper (journal, conference, etc.).
    """
    # Move supplements to journal paper category for the plot (a value of one is
    # not visible on a bar graph).
    df_plot = df.copy()
    df_plot.loc[df['Type of paper'] == 'Supplement', :] = 'Journal'

    fig, ax = plt.subplots(figsize=(save_cfg['text_width'] / 4, 
                                    save_cfg['text_height'] / 5))
    sns.countplot(x=df_plot['Type of paper'], ax=ax)
    ax.set_xlabel('')
    ax.set_ylabel('Number of papers')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    plt.tight_layout()

    counts = df['Type of paper'].value_counts()
    logger.info('Number of journal papers: {}'.format(counts['Journal']))
    logger.info('Number of conference papers: {}'.format(counts['Conference']))
    logger.info('Number of preprints: {}'.format(counts['Preprint']))
    logger.info('Number of papers that were initially published as preprints: '
                '{}'.format(df[df['Type of paper'] != 'Preprint'][
                    'Preprint first'].value_counts()['Yes']))

    if save_cfg is not None:
        fname = os.path.join(save_cfg['savepath'], 'type_of_paper')
        fig.savefig(fname + '.' + save_cfg['format'], **save_cfg)

    return ax


def plot_country(df, save_cfg=cfg.saving_config):
    """Plot bar graph showing the country of the first author's affiliation.
    """
    fig, ax = plt.subplots(figsize=(save_cfg['text_width'] / 4 * 3, 
                                    save_cfg['text_height'] / 5))
    sns.countplot(x=df['Country'], ax=ax,
                order=df['Country'].value_counts().index)
    ax.set_ylabel('Number of papers')
    ax.set_xlabel('')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    plt.tight_layout()

    top3 = df['Country'].value_counts().index[:3]
    logger.info('Top 3 countries of first author affiliation: {}'.format(top3.values))

    if save_cfg is not None:
        fname = os.path.join(save_cfg['savepath'], 'country')
        fig.savefig(fname + '.' + save_cfg['format'], **save_cfg)

    return ax


def plot_countrymap(df, save_cfg=cfg.saving_config):
    """Plot bar graph showing the country of the first author's affiliation.
    """
    #fig, ax = plt.subplots(figsize=(save_cfg['text_width'] / 4 * 3, 
    #                                save_cfg['text_height'] / 5))

    shapefile = os.path.expanduser('../img/countries/ne_10m_admin_0_countries.shp')

    gdf = gpd.read_file(shapefile)[['ADMIN', 'geometry']].to_crs('+proj=robin')
    #gdf.head()

    dfx = df
    dfx = dfx.Country.value_counts().reset_index().rename(
        columns={'index': 'Country', 'Country': 'Count'})
    #dfx.head()

    #print("Renaming Exceptions!")
    #print(dfx.loc[~dfx['Country'].isin(gdf['ADMIN'])])

    # Exception #1 - USA: United States of America
    dfx.loc[dfx['Country'] == 'USA', 'Country'] = 'United States of America'

    # Exception #2 - UK: United Kingdom
    dfx.loc[dfx['Country'] == 'UK', 'Country'] = 'United Kingdom'

    # Exception #3 - Bosnia: Bosnia and Herzegovina
    dfx.loc[dfx['Country'] == 'Bosnia', 'Country'] = 'Bosnia and Herzegovina'

    if len(dfx.loc[~dfx['Country'].isin(gdf['ADMIN'])]) > 0:
        print("## ERROR ## - Unhandled Countries!")

    # Adding 0 to all other countries!
    for c in gdf['ADMIN']:
        if any(dfx['Country'].str.contains(c)):
            gdf.loc[gdf['ADMIN'] == c, 'Count'] = dfx[
                dfx['Country'].str.contains(c)]['Count'].values[0]
        else:
            gdf.loc[gdf['ADMIN'] == c, 'Count'] = 0

    figsize = (16, 10)
    ax = gdf.plot(column='Count', figsize=figsize, cmap='Blues', 
                  scheme='Fisher_Jenks', k=10, legend=True, axes=None)
    
    ax.set_axis_off()
    #ax.set_xlim([-1.5e7, 1.7e7])
    #ax.get_legend().set_bbox_to_anchor((.12, .4))
    fig = ax.get_figure()
    
    if save_cfg is not None:
        fname = os.path.join(save_cfg['savepath'], 'countrymap')
        fig.savefig(fname + '.' + save_cfg['format'], **save_cfg)

    return ax


def compute_prct_statistical_test(df):
    """Compute the number of studies that used statistical tests.
    """
    prct = 100 - 100 * df['Statistical analysis of performance'].value_counts()['No'] / df.shape[0]
    logger.info('% of studies that used statistical test: {}'.format(prct))


def make_domain_table(df):
    """Make domain table that contains every reference.
    """
    # Replace NaNs by ' ' in 'Domain 3' and 'Domain 4' columns
    df = ut.replace_nans_in_column(df, 'Domain 3', replace_by=' ')
    df = ut.replace_nans_in_column(df, 'Domain 4', replace_by=' ')

    cols = ['Domain 1', 'Domain 2', 'Domain 3', 'Domain 4', 'Architecture (clean)']
    df[cols] = df[cols].applymap(ut.tex_escape)

    # Make tuple of first 2 domain levels
    domains_df = df.groupby(cols)['Citation'].apply(list).apply(
        lambda x: '\cite{' + ', '.join(x) + '}').unstack()
    domains_df = domains_df.applymap(
        lambda x: ' ' if isinstance(x, float) and np.isnan(x) else x)

    with open('domains_architecture_table.tex', 'w') as f:
        with pd.option_context("max_colwidth", 1000):
            f.write(domains_df.to_latex(
                escape=False, 
                column_format='p{1.5cm}' * 4 + 'p{0.6cm}' * domains_df.shape[1]))


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
    df['items'] = df['items'].apply(ut.wrap_text, max_char=20)

    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(x='counts', y='items', data=df, ax=ax)
    ax.set_ylabel('' if ylabel is None else ylabel)

    ylabels = ax.get_yticklabels()
    ax.set_yticklabels(ylabels, ha='right')

    ax.set_xlabel('Percentage (%)' if xlabel is None else xlabel)
    ax.set_xlim([0, 100])
    if title is not None:
        ax.set_title(title)

    sns.set_palette(sns.color_palette('Paired'))

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


def plot_preprocessing_proportions(df, save_cfg=cfg.saving_config):
    """Plot proportions for preprocessing-related data items.
    """
    data = dict()
    data['(a) Preprocessing of EEG data'] = df[
         'Preprocessing (clean)'].value_counts().to_dict()
    data['(b) Artifact handling'] = df[
         'Artefact handling (clean)'].value_counts().to_dict()
    data['(c) Extracted features'] = df[
         'Features (clean)'].value_counts().to_dict()

    fig, ax = plot_multiple_proportions(
        data, print_count=5, respect_order=['Yes', 'No', 'Other', 'N/M'],
        figsize=(save_cfg['text_width'] / 4 * 4, save_cfg['text_height'] / 7 * 2))
    
    if save_cfg is not None:
        fname = os.path.join(save_cfg['savepath'], 'preprocessing')
        fig.savefig(fname + '.' + save_cfg['format'], **save_cfg)

    return ax


def plot_hyperparams_proportions(df, save_cfg=cfg.saving_config):
    """Plot proportions for hyperparameter-related data items.
    """
    data = dict()
    data['(a) Training procedure'] = df[
         'Training procedure (clean)'].value_counts().to_dict()
    data['(b) Regularization'] = df[
         'Regularization (clean)'].value_counts().to_dict()
    data['(c) Optimizer'] = df[
         'Optimizer (clean)'].value_counts().to_dict()

    fig, ax = plot_multiple_proportions(
        data, print_count=5, respect_order=['Yes', 'No', 'Other', 'N/M'],
        figsize=(save_cfg['text_width'] / 4 * 4, save_cfg['text_height'] / 7 * 2))
    
    if save_cfg is not None:
        fname = os.path.join(save_cfg['savepath'], 'hyperparams')
        fig.savefig(fname + '.' + save_cfg['format'], **save_cfg)

    return ax


def plot_reproducibility_proportions(df, save_cfg=cfg.saving_config):
    """Plot proportions for reproducibility-related data items.
    """
    df['Code hosted on'] = df['Code hosted on'].replace(np.nan, 'N/M', regex=True)
    df['Limited data'] = df['Limited data'].replace(np.nan, 'N/M', regex=True)
    df['Code available'] = df['Code available'].replace(np.nan, 'N/M', regex=True)

    data = dict()
    data['(a) Dataset availability'] = df[
         'Dataset accessibility'].value_counts().to_dict()
    data['(b) Code availability'] = df[
         'Code hosted on'].value_counts().to_dict()
    data['(c) Type of baseline'] = df[
         'Baseline model type'].value_counts().to_dict()

    df['reproducibility'] = 'Hard'
    df.loc[(df['Code available'] == 'Yes') & 
           (df['Dataset accessibility'] == 'Public'), 'reproducibility'] = 'Easy' 
    df.loc[(df['Code available'] == 'Yes') & 
           (df['Dataset accessibility'] == 'Both'), 'reproducibility'] = 'Medium' 
    df.loc[(df['Code available'] == 'No') & 
           (df['Dataset accessibility'] == 'Private'), 'reproducibility'] = 'Impossible' 

    data['(d) Reproducibility'] = df[
         'reproducibility'].value_counts().to_dict()

    logger.info('Stats on reproducibility - Dataset Accessibility: {}'.format(data['(a) Dataset availability']))
    logger.info('Stats on reproducibility - Code Accessibility: {}'.format(df['Code available'].value_counts().to_dict()))
    logger.info('Stats on reproducibility - Code Hosted On: {}'.format(data['(b) Code availability']))
    logger.info('Stats on reproducibility - Baseline: {}'.format(data['(c) Type of baseline']))
    logger.info('Stats on reproducibility - Reproducibility Level: {}'.format(data['(d) Reproducibility']))
    logger.info('Stats on reproducibility - Limited data: {}'.format(df['Limited data'].value_counts().to_dict()))
    logger.info('Stats on reproducibility - Shared their Code: {}'.format(df[df['Code available'] == 'Yes']['Citation'].to_dict()))

    fig, ax = plot_multiple_proportions(
        data, print_count=5, respect_order=['Easy', 'Medium', 'Hard', 'Impossible'],
        figsize=(save_cfg['text_width'] / 4 * 4, save_cfg['text_height'] / 7 * 3))
    
    if save_cfg is not None:
        fname = os.path.join(save_cfg['savepath'], 'reproducibility')
        fig.savefig(fname + '.' + save_cfg['format'], **save_cfg)

    return ax


def plot_domains_per_year(df, save_cfg=cfg.saving_config):
    """Plot stacked bar graph of domains per year.
    """
    fig, ax = plt.subplots(
        figsize=(save_cfg['text_width'] / 4 * 2, save_cfg['text_height'] / 4))

    df['Year'] = df['Year'].astype('int32')
    main_domains = ['Epilepsy', 'Sleep', 'BCI', 'Affective', 'Cognitive', 
                    'Improvement of processing tools', 'Generation of data']
    domains_df = df[['Domain 1', 'Domain 2', 'Domain 3', 'Domain 4']]
    df['Main domain'] = [row[row.isin(main_domains)].values[0] 
        if any(row.isin(main_domains)) else 'Others' 
        for ind, row in domains_df.iterrows()]
    df.groupby(['Year', 'Main domain']).size().unstack('Main domain').plot(
        kind='bar', stacked=True, title='', ax=ax)
    ax.set_ylabel('Number of papers')
    ax.set_xlabel('')

    legend = plt.legend()
    for l in legend.get_texts():
        l.set_text(ut.wrap_text(l.get_text(), max_char=14))

    if save_cfg is not None:
        fname = os.path.join(save_cfg['savepath'], 'domains_per_year')
        fig.savefig(fname + '.' + save_cfg['format'], **save_cfg)

    return ax


def plot_hardware(df, save_cfg=cfg.saving_config):
    """Plot bar graph showing the hardware used in the study.
    """
    col = 'EEG Hardware'
    hardware_df = ut.split_column_with_multiple_entries(
        df, col, ref_col='Citation', sep=',', lower=False)

    # Remove N/Ms because they make it hard to see anything
    hardware_df = hardware_df[hardware_df[col] != 'N/M']
    
    # Add low cost column
    hardware_df['Low-cost'] = False
    low_cost_devices = ['EPOC (Emotiv)', 'OpenBCI (OpenBCI)', 'Muse (InteraXon)', 
                        'Mindwave Mobile (Neurosky)', 'Mindset (NeuroSky)']
    hardware_df.loc[hardware_df[col].isin(low_cost_devices), 
                    'Low-cost'] = True

    fig, ax = plt.subplots(figsize=(save_cfg['text_width'] / 4 * 2, 
                                    save_cfg['text_height'] / 5 * 2))
    sns.countplot(hue=hardware_df['Low-cost'], y=hardware_df[col], ax=ax,
                  order=hardware_df[col].value_counts().index, 
                  dodge=False)
    # sns.catplot(row=hardware_df['low_cost'], y=hardware_df['hardware'])
    ax.set_xlabel('Number of papers')
    ax.set_ylabel('')
    plt.tight_layout()

    if save_cfg is not None:
        fname = os.path.join(save_cfg['savepath'], 'hardware')
        fig.savefig(fname + '.' + save_cfg['format'], **save_cfg)

    return ax


def plot_architectures(df, save_cfg=cfg.saving_config):
    """Plot bar graph showing the architectures used in the study.
    """
    fig, ax = plt.subplots(figsize=(save_cfg['text_width'] / 3, 
                                    save_cfg['text_width'] / 3))
    colors = sns.color_palette('Paired')
    counts = df['Architecture (clean)'].value_counts()
    _, _, pct = ax.pie(counts.values, labels=counts.index, autopct='%1.1f%%',
           wedgeprops=dict(width=0.3, edgecolor='w'), colors=colors,
           pctdistance=0.55)
    for i in pct:
        i.set_fontsize(5)

    ax.axis('equal')
    plt.tight_layout()

    if save_cfg is not None:
        fname = os.path.join(save_cfg['savepath'], 'architectures')
        fig.savefig(fname + '.' + save_cfg['format'], **save_cfg)

    return ax
    

def plot_architectures_per_year(df, save_cfg=cfg.saving_config):
    """Plot stacked bar graph of architectures per year.
    """
    fig, ax = plt.subplots(
        figsize=(save_cfg['text_width'] / 3 * 2, save_cfg['text_width'] / 3))
    colors = sns.color_palette('Paired')

    df['Year'] = df['Year'].astype('int32')
    col_name = 'Architecture (clean)'
    df['Arch'] = df[col_name]
    order = df[col_name].value_counts().index
    counts = df.groupby(['Year', 'Arch']).size().unstack('Arch')
    counts = counts[order]

    counts.plot(kind='bar', stacked=True, title='', ax=ax, color=colors)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.set_ylabel('Number of papers')
    ax.set_xlabel('')

    plt.tight_layout()

    if save_cfg is not None:
        fname = os.path.join(save_cfg['savepath'], 'architectures_per_year')
        fig.savefig(fname + '.' + save_cfg['format'], **save_cfg)

    return ax


def plot_architectures_vs_input(df, save_cfg=cfg.saving_config):
    """Plot stacked bar graph of architectures vs input type.
    """
    fig, ax = plt.subplots(
        figsize=(save_cfg['text_width'] / 4 * 2, save_cfg['text_width'] / 3))

    df['Input'] = df['Features (clean)']
    col_name = 'Architecture (clean)'
    df['Arch'] = df[col_name]
    order = df[col_name].value_counts().index
    counts = df.groupby(['Input', 'Arch']).size().unstack('Input')
    counts = counts.loc[order, :]

    # To reduce the height of the figure, wrap long xticklabels
    counts = counts.rename({'CNN+RNN': 'CNN+\nRNN'}, axis='index')

    counts.plot(kind='bar', stacked=True, title='', ax=ax)
    # ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.set_ylabel('Number of papers')
    ax.set_xlabel('')

    plt.tight_layout()

    if save_cfg is not None:
        fname = os.path.join(save_cfg['savepath'], 'architectures_vs_input')
        fig.savefig(fname + '.' + save_cfg['format'], **save_cfg)

        save_cfg2 = save_cfg.copy()
        save_cfg2['format'] = 'png'
        fig.savefig(fname + '.png', **save_cfg2)

    return ax


def plot_optimizers_per_year(df, save_cfg=cfg.saving_config):
    """Plot stacked bar graph of optimizers per year.
    """
    fig, ax = plt.subplots(
        figsize=(save_cfg['text_width'] / 4 * 2, save_cfg['text_width'] / 5 * 2))

    df['Input'] = df['Features (clean)']
    col_name = 'Optimizer (clean)'
    df['Opt'] = df[col_name]
    order = df[col_name].value_counts().index
    counts = df.groupby(['Year', 'Opt']).size().unstack('Opt')
    counts = counts[order]

    counts.plot(kind='bar', stacked=True, title='', ax=ax)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.set_ylabel('Number of papers')
    ax.set_xlabel('')

    plt.tight_layout()

    if save_cfg is not None:
        fname = os.path.join(save_cfg['savepath'], 'optimizers_per_year')
        fig.savefig(fname + '.' + save_cfg['format'], **save_cfg)

    return ax


def plot_intra_inter_per_year(df, save_cfg=cfg.saving_config):
    """Plot stacked bar graph of intra-/intersubject studies per year.
    """
    fig, ax = plt.subplots(
        figsize=(save_cfg['text_width'] / 4 * 2, save_cfg['text_height'] / 4))

    df['Year'] = df['Year'].astype(int)
    col_name = 'Intra/Inter subject'
    order = df[col_name].value_counts().index
    counts = df.groupby(['Year', col_name]).size().unstack(col_name)
    counts = counts[order]

    logger.info('Stats on inter/intra subjects: {}'.format(
        df[col_name].value_counts() / df.shape[0] * 100))

    counts.plot(kind='bar', stacked=True, title='', ax=ax)
    # ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.set_ylabel('Number of papers')
    ax.set_xlabel('')

    plt.tight_layout()

    if save_cfg is not None:
        fname = os.path.join(save_cfg['savepath'], 'intra_inter_per_year')
        fig.savefig(fname + '.' + save_cfg['format'], **save_cfg)

    return ax


def plot_number_layers(df, save_cfg=cfg.saving_config):
    """Plot histogram of number of layers.
    """
    fig, ax = plt.subplots(
        figsize=(save_cfg['text_width'] / 4 * 2, save_cfg['text_width'] / 3))

    n_layers_df = df['Layers (clean)'].value_counts().reindex(
        [str(i) for i in range(1, 32)] + ['N/M'])
    n_layers_df = n_layers_df.dropna().astype(int)

    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(sns.color_palette(None).as_hex())

    n_layers_df.plot(kind='bar', width=0.8, rot=0, colormap=cmap, ax=ax)
    ax.set_xlabel('Number of layers')
    ax.set_ylabel('Number of papers')
    plt.tight_layout()

    if save_cfg is not None:
        fname = os.path.join(save_cfg['savepath'], 'number_layers')
        fig.savefig(fname + '.' + save_cfg['format'], **save_cfg)

        save_cfg2 = save_cfg.copy()
        save_cfg2['format'] = 'png'
        save_cfg2['dpi'] = 300
        fig.savefig(fname + '.png', **save_cfg2)

    return ax   


def plot_number_subjects_by_domain(df, save_cfg=cfg.saving_config):
    """Plot number of subjects in studies by domain.
    """
    # Split values into separate rows and remove invalid values
    col = 'Data - subjects'
    nb_subj_df = ut.split_column_with_multiple_entries(
        df, col, ref_col='Main domain')
    nb_subj_df = nb_subj_df.loc[~nb_subj_df[col].isin(['n/m', 'tbd'])]
    nb_subj_df[col] = nb_subj_df[col].astype(int)
    nb_subj_df = nb_subj_df.loc[nb_subj_df[col] > 0, :]

    nb_subj_df['Main domain'] = nb_subj_df['Main domain'].apply(
        ut.wrap_text, max_char=13)

    fig, ax = plt.subplots(
        figsize=(save_cfg['text_width'] / 3 * 2, save_cfg['text_height'] / 3))
    ax.set(xscale='log', yscale='linear')
    sns.swarmplot(
        y='Main domain', x=col, data=nb_subj_df, 
        ax=ax, size=3, order=nb_subj_df.groupby(['Main domain'])[
            col].median().sort_values().index)
    ax.set_xlabel('Number of subjects')
    ax.set_ylabel('')
    
    logger.info('Stats on number of subjects per model: {}'.format(nb_subj_df[col].describe()))

    plt.tight_layout()

    if save_cfg is not None:
        fname = os.path.join(save_cfg['savepath'], 'nb_subject_per_domain')
        fig.savefig(fname + '.' + save_cfg['format'], **save_cfg)

    return ax 


def plot_number_channels(df, save_cfg=cfg.saving_config):
    """Plot histogram of number of channels.
    """
    nb_channels_df = ut.split_column_with_multiple_entries(
        df, 'Nb Channels', ref_col='Citation', sep=';\n', lower=False)
    nb_channels_df['Nb Channels'] = nb_channels_df['Nb Channels'].astype(int)
    nb_channels_df = nb_channels_df.loc[nb_channels_df['Nb Channels'] > 0, :]

    fig, ax = plt.subplots(
        figsize=(save_cfg['text_width'] / 2, save_cfg['text_height'] / 4))
    sns.distplot(nb_channels_df['Nb Channels'], kde=False, norm_hist=False, ax=ax)
    ax.set_xlabel('Number of EEG channels')
    ax.set_ylabel('Number of papers')

    logger.info('Stats on number of channels per model: {}'.format(nb_channels_df['Nb Channels'].describe()))

    plt.tight_layout()

    if save_cfg is not None:
        fname = os.path.join(save_cfg['savepath'], 'nb_channels')
        fig.savefig(fname + '.' + save_cfg['format'], **save_cfg)

    return ax


def compute_stats_sampling_rate(df):
    """Compute the statistics for hardware sampling rate.
    """
    fs_df = ut.split_column_with_multiple_entries(
    df, 'Sampling rate', ref_col='Citation', sep=';\n', lower=False)
    fs_df['Sampling rate'] = fs_df['Sampling rate'].astype(float)
    fs_df = fs_df.loc[fs_df['Sampling rate'] > 0, :]

    logger.info('Stats on sampling rate per model: {}'.format(fs_df['Sampling rate'].describe()))


def plot_cross_validation(df, save_cfg=cfg.saving_config):
    """Plot bar graph of cross validation approaches.
    """
    col = 'Cross validation (clean)'
    df[col] = df[col].fillna('N/M')
    cv_df = ut.split_column_with_multiple_entries(
        df, col, ref_col='Citation', sep=';\n', lower=False)
    
    fig, ax = plt.subplots(
        figsize=(save_cfg['text_width'] / 2, save_cfg['text_height'] / 5))
    sns.countplot(y=cv_df[col], order=cv_df[col].value_counts().index, ax=ax)
    ax.set_xlabel('Number of papers')
    ax.set_ylabel('')
    
    plt.tight_layout()

    if save_cfg is not None:
        fname = os.path.join(save_cfg['savepath'], 'cross_validation')
        fig.savefig(fname + '.' + save_cfg['format'], **save_cfg)

    return ax


def make_dataset_table(df, min_n_articles=2, save_cfg=cfg.saving_config):
    """Make table that reports most used datasets.

    Args:
        df

    Keyword Args:
        min_n_articles (int): minimum number of times a dataset must have been
            used to be listed in the table. If under that number, will appear as
            'Other' in the table.
        save_cfg (dict)
    """
    def merge_dataset_names(s):
        if 'bci comp' in s.lower():
            s = 'BCI Competition'
        elif 'tuh' in s.lower():
            s = 'TUH'
        elif 'mahnob' in s.lower():
            s = 'MAHNOB'
        return s

    col = 'Dataset name'
    datasets_df = ut.split_column_with_multiple_entries(
        df, col, ref_col=['Main domain', 'Citation'], sep=';\n', lower=False)

    # Remove not mentioned and internal recordings, as readers won't be able to 
    # use these datasets anyway
    datasets_df = datasets_df.loc[~datasets_df[col].isin(
        ['N/M', 'Internal Recordings', 'TBD'])]

    datasets_df['Dataset'] = datasets_df[col].apply(merge_dataset_names).apply(
        ut.tex_escape)

    # Replace datasets that were used rarely by 'Other'
    counts = datasets_df['Dataset'].value_counts()
    datasets_df.loc[datasets_df['Dataset'].isin(
        counts[counts < min_n_articles].index), 'Dataset'] = 'Other'

    # Group by dataset and order by number of articles
    dataset_table = datasets_df.groupby(
        ['Main domain', 'Dataset'], as_index=True)['Citation'].apply(list)
    dataset_table = pd.concat([dataset_table.apply(len), dataset_table], axis=1)
    dataset_table.columns = [r'\# articles', 'References']

    dataset_table = dataset_table.sort_values(
        by=['Main domain', r'\# articles'], ascending=[True, False])
    dataset_table['References'] = dataset_table['References'].apply(
        lambda x: r'\cite{' + ', '.join(x) + '}')

    tex_table = dataset_table.to_latex(escape=False, multicolumn=False)
    
    with open(os.path.join(save_cfg['savepath'], 'dataset_table.tex'), 'w') as f:
        with pd.option_context("max_colwidth", 1000):
            f.write(dataset_table.to_latex(escape=False, multicolumn=False))


def plot_data_quantity(df, save_cfg=cfg.saving_config):
    """Plot the quantity of data used by domain.
    """
    data_df = ut.split_column_with_multiple_entries(
        df, ['Data - samples', 'Data - time'], ref_col=['Citation', 'Main domain'], 
        sep=';\n', lower=False)

    # Remove N/M and TBD
    col = 'Data - samples'
    data_df.loc[data_df[col].isin(['N/M', 'TBD', '[TBD]']), col] = np.nan
    data_df[col] = data_df[col].astype(float)

    col2 = 'Data - time'
    data_df.loc[data_df[col2].isin(['N/M', 'TBD', '[TBD]']), col2] = np.nan
    data_df[col2] = data_df[col2].astype(float)

    # Wrap main domain text
    data_df['Main domain'] = data_df['Main domain'].apply(
        ut.wrap_text, max_char=13)

    # Extract ratio
    data_df['data_ratio'] = data_df['Data - samples'] / data_df['Data - time']
    data_df = data_df.sort_values(['Main domain', 'data_ratio'])

    # Plot
    fig, axes = plt.subplots(
        ncols=3, 
        figsize=(save_cfg['text_width'], save_cfg['text_height'] / 3))

    axes[0].set(xscale='log', yscale='linear')
    sns.swarmplot(y='Main domain', x=col2, data=data_df, ax=axes[0], size=3)
    axes[0].set_xlabel('Recording time (min)')
    axes[0].set_ylabel('')
    max_val = int(np.ceil(np.log10(data_df[col2].max())))
    axes[0].set_xticks(np.power(10, range(0, max_val + 1)))

    axes[1].set(xscale='log', yscale='linear')
    sns.swarmplot(y='Main domain', x=col, data=data_df, ax=axes[1], size=3)
    axes[1].set_xlabel('Number of examples')
    axes[1].set_yticklabels('')
    axes[1].set_ylabel('')
    min_val = int(np.floor(np.log10(data_df[col].min())))
    max_val = int(np.ceil(np.log10(data_df[col].max())))
    axes[1].set_xticks(np.power(10, range(min_val, max_val + 1)))

    axes[2].set(xscale='log', yscale='linear')
    sns.swarmplot(y='Main domain', x='data_ratio', data=data_df, ax=axes[2], 
                  size=3)
    axes[2].set_xlabel('Ratio (examples/min)')
    axes[2].set_ylabel('')
    axes[2].set_yticklabels('')
    min_val = int(np.floor(np.log10(data_df['data_ratio'].min())))
    max_val = int(np.ceil(np.log10(data_df['data_ratio'].max())))
    axes[2].set_xticks(np.power(10, np.arange(min_val, max_val + 1, dtype=float)))

    plt.tight_layout()

    if save_cfg is not None:
        fname = os.path.join(save_cfg['savepath'], 'data_quantity')
        fig.savefig(fname + '.' + save_cfg['format'], **save_cfg)

    return axes


def get_real_eeg_data(start=0, stop=4, chans=4):
    """Get real EEG data for plotting.

    Keyword Args:
        start (float): start of the EEG segment, in seconds.
        stop (float): end of the EEG segment, in seconds.
        chans (int or list): number of channels to extract, or list of channel
            indices to be interpreted by MNE's get_data() function.
    """
    raw_fnames = eegbci.load_data(1, 2)
    raws = [read_raw_edf(f, preload=True) for f in raw_fnames]
    raw = concatenate_raws(raws)

    fs = raw.info['sfreq']
    start = int(fs * start)
    stop = int(fs * stop)

    if not isinstance(chans, list):
        chans = np.arange(chans)
    data, t = raw.get_data(picks=chans, start=start, stop=stop, return_times=True)
    data = data.T

    return data, t, fs


def create_fake_eeg(fs=256, signal_len=4, n_channels=4):
    """Create fake EEG data.
    """
    n_points = fs * signal_len
    t = np.arange(n_points) / fs
    data = np.random.rand(n_points, n_channels)

    return data, t


def draw_brace(ax, xspan, text, beta_factor=300, y_offset=None):
    """Draws an annotated brace on the axes.
    
    Adapted from https://stackoverflow.com/a/53383764"""
    xmin, xmax = xspan
    xspan = xmax - xmin
    ax_xmin, ax_xmax = ax.get_xlim()
    xax_span = ax_xmax - ax_xmin
    ymin, ymax = ax.get_ylim()
    yspan = ymax - ymin
    resolution = int(xspan/xax_span*100)*2+1 # guaranteed uneven
    beta = beta_factor / xax_span # the higher this is, the smaller the radius

    x = np.linspace(xmin, xmax, resolution)
    x_half = x[:int(np.ceil(resolution/2))]
    y_half_brace = (1/(1.+np.exp(-beta*(x_half-x_half[0])))
                    + 1/(1.+np.exp(-beta*(x_half-x_half[-1]))))
    y = np.concatenate((y_half_brace, y_half_brace[-2::-1]))
    if y_offset is not None:
        ymin = y_offset
    y = ymin + (.035*y - .01) * yspan  # adjust vertical position

    # ax.autoscale(False)
    ax.plot(x, y, color='black', lw=1)

    ax.text((xmax+xmin)/2., ymin+.05*yspan, text, ha='center', va='bottom')



def plot_eeg_intro(save_cfg=cfg.saving_config):
    """Plot a figure that shows basic EEG notions such as epochs and samples.
    """

    # Visualization parameters
    win_len = 1  # in s
    step = 0.5  # in s
    first_epoch = 1

    data, t, fs = get_real_eeg_data(start=30, stop=34, chans=[0, 10, 20, 30])
    t = t - t[0]

    # Offset data for visualization
    data -= data.mean(axis=0)
    max_std = np.max(data.std(axis=0))
    offsets = np.arange(data.shape[1])[::-1] * 4 * max_std
    data += offsets

    rect_y_border = 0.6 * max_std
    min_y = data.min() - rect_y_border
    max_y = data.max() + rect_y_border

    # Make figure
    fig, ax = plt.subplots(
        figsize=(save_cfg['text_width'] / 4 * 3, save_cfg['text_height'] / 3))
    ax.plot(t, data)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(r'Amplitude (e.g., $\mu$V)')
    ax.set_yticks(offsets)
    ax.set_yticklabels(['channel {}'.format(i + 1) for i in range(data.shape[1])])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Display epochs as dashed line rectangles
    rect1 = patches.Rectangle((first_epoch, min_y + rect_y_border / 4), 
                            win_len, max_y - min_y, 
                            linewidth=1, linestyle='--', edgecolor='k',
                            facecolor='none')
    rect2 = patches.Rectangle((first_epoch + step, min_y - rect_y_border / 4), 
                            win_len, max_y - min_y, 
                            linewidth=1, linestyle='--', edgecolor='k',
                            facecolor='none')

    ax.add_patch(rect1)
    ax.add_patch(rect2)

    # Annotate epochs
    ax.annotate(
        r'$\bf{Window}$ or $\bf{epoch}$ or $\bf{trial}$' +
        '\n({:.0f} points in a \n1-s window at {:.0f} Hz)'.format(fs, fs), #fontsize=14, 
        xy=(first_epoch, min_y), 
        arrowprops=dict(facecolor='black', shrink=0.05, width=2, headwidth=6),
        xytext=(0, min_y - 3.5 * max_std),
        xycoords='data', ha='center', va='top')

    # Annotate sample
    special_ind = np.where((t >= 2.4) & (t < 2.5))[0][0]
    special_point = data[special_ind, 0]
    ax.plot(t[special_ind], special_point, '.', c='k')
    ax.annotate(
        r'$\bf{Point}$ or $\bf{sample}$', #fontsize=14, 
        xy=(t[special_ind], special_point), 
        arrowprops=dict(facecolor='black', shrink=0.05, width=2, headwidth=6),
        xytext=(3, max_y),
        xycoords='data', ha='left', va='bottom')

    # Annotate overlap
    draw_brace(ax, (first_epoch + step, first_epoch + step * 2), 
            r'0.5-s $\bf{overlap}$' + '\nbetween windows', 
            beta_factor=300, y_offset=max_y)

    plt.tight_layout()

    if save_cfg is not None:
        fname = os.path.join(save_cfg['savepath'], 'eeg_intro')
        fig.savefig(fname + '.' + save_cfg['format'], **save_cfg)

    return ax
