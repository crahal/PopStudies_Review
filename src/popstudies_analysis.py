import pandas as pd
import matplotlib.pyplot as plt
from scipy import sparse as sp
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.lines as mlines
from collections import defaultdict
import seaborn as sns
import matplotlib.dates as mdates
import geopandas as gpd
import matplotlib as mpl
import os
import matplotlib.patches as patches
import matplotlib.ticker as mtick
from matplotlib import cm
from matplotlib.colors import ListedColormap,LinearSegmentedColormap
import numpy as np
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
import re
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.probability import FreqDist
import warnings
from matplotlib import colors
from matplotlib import rcParams
rcParams['font.family'] = 'Helvetica'

warnings.simplefilter(action='ignore', category=FutureWarning)
mpl.rc('font', family='Helvetica')
csfont = {'fontname': 'Helvetica'}
hfont = {'fontname': 'Helvetica'}


def open_access_analysis(main_df, figure_path):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    color = ['#377eb8', '#ffb94e']

    OA_df = pd.DataFrame(index=main_df['year'].sort_values(ascending=True).unique(),
                         columns=['number_articles', 'number_oa'])
    for year in main_df['year'].sort_values(ascending=True).unique():
        temp = main_df[main_df['year'] ==year]
        OA_df.at[year, 'number_articles'] = len(temp)
        OA_df.at[year, 'number_oa'] = len(temp[temp['openaccess']==1])
    OA_df['percent_OA'] = (OA_df['number_oa']/OA_df['number_articles'])*100
    ax1.plot(OA_df.index.to_list(), OA_df['number_articles'], color=color[0])
    ax2.plot(OA_df.index.to_list(), OA_df['number_oa'], color=color[1])
    ax3.plot(OA_df.index.to_list(), OA_df['percent_OA'], color='r')
    sns.despine()
    from matplotlib.ticker import MaxNLocator
    ax1.xaxis.set_major_locator(MaxNLocator(5))
    ax2.xaxis.set_major_locator(MaxNLocator(5))
    ax3.xaxis.set_major_locator(MaxNLocator(5))
    ax1.set_ylabel('Number of Papers Published')
    ax2.set_ylabel('Number of Papers Open Access')
    ax3.set_ylabel('Percent of Papers Open Access')
    plt.savefig(os.path.join(figure_path, 'number_of_papers.pdf'),
                bbox_inches='tight')
    print(OA_df)

def uncited_ratios(all_papers, topic_list):
    uncited_comp = pd.DataFrame(index=topic_list, columns = ['number_papers', 'uncited'])
    for topic in topic_list:
        uncited_comp.at[topic, 'number_papers'] = len(all_papers[(all_papers['Topic'].str.contains(str(topic))) &
                                                                 (all_papers['citedbycount']>0)])
        uncited_comp.at[topic, 'uncited'] = len(all_papers[(all_papers['Topic'].str.contains(str(topic))) &
                                                           (all_papers['citedbycount']==0)])
    uncited_comp['uncited_ratio'] = uncited_comp['uncited']/uncited_comp['number_papers']
    print(uncited_comp.sort_values(by = 'uncited_ratio'))


def uncited_papers(main_df_noabs, figure_path):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    color = ['#377eb8', '#ffb94e']
    uncited = main_df_noabs[main_df_noabs['citedbycount']==0]
    print(len(uncited))
    uncited_count = pd.DataFrame(index=main_df_noabs['year'].sort_values(ascending=True).unique(),
                                 columns=['number_articles', 'number_uncited'])
    for year in main_df_noabs['year'].sort_values(ascending=True).unique():
        temp = main_df_noabs[main_df_noabs['year'] ==year]
        uncited_count.at[year, 'number_articles'] = len(temp)
        uncited_count.at[year, 'number_uncited'] = len(temp[temp['citedbycount']==0])
    uncited_count['percent_uncited'] = (uncited_count['number_uncited']/uncited_count['number_articles'])*100
    ax1.plot(uncited_count.index.to_list(), uncited_count['number_articles'], color=color[0])
    ax2.plot(uncited_count.index.to_list(), uncited_count['number_uncited'], color=color[1])
    ax3.plot(uncited_count.index.to_list(), uncited_count['percent_uncited'], color='r')
    sns.despine()
    from matplotlib.ticker import MaxNLocator
    ax1.xaxis.set_major_locator(MaxNLocator(5))
    ax2.xaxis.set_major_locator(MaxNLocator(5))
    ax3.xaxis.set_major_locator(MaxNLocator(5))
    ax1.set_ylabel('Number of Papers Published')
    ax2.set_ylabel('Number of Papers Uncited')
    ax3.set_ylabel('Percent of Papers Uncited')
    plt.savefig(os.path.join(figure_path, 'number_of_uncited_papers.pdf'),
                bbox_inches='tight')
    return uncited_count



def title_analysis(main_df):
    main_df['title_length'] = main_df['Title'].str.len()
    main_df = main_df.sort_values(by='prismcoverdate', ascending=True)
    tit_mean = main_df['title_length'].mean()
    print('Mean length of titles: ' + str(tit_mean) + ' characters')
    tit_max = main_df['title_length'].max()
    print('Max length of titles: ' + str(tit_max) + ' characters')
    tit_min = main_df['title_length'].min()
    print('Min length of titles: ' + str(tit_min) + ' characters')
    print('The shortest title is: ' +
          main_df.sort_values(by='title_length',
                              ascending=True).reset_index().at[0,
                                                               'Title'])
    print('The longest title is: ' +
          main_df.sort_values(by='title_length',
                              ascending=False).reset_index().at[0,
                                                                'Title'])
    tit_mean_firsthalf = main_df[0:int(len(main_df)/2)]['title_length'].mean()
    print('Mean length of titles, first half of period: ' +\
          str(tit_mean_firsthalf) + ' characters')
    tit_mean_secondhalf = main_df[int(len(main_df)/2):]['title_length'].mean()
    print('Mean length of titles, second half of period: ' +\
          str(tit_mean_secondhalf) + ' characters')


def plot_all_G(G, authors_df, author_papers, fig_path):
    fig = plt.figure(figsize=(14, 14))
    ax = plt.subplot2grid((20, 33), (0, 0), rowspan=10, colspan=33)
    ax1 = plt.subplot2grid((20, 33), (10, 0), rowspan=10, colspan=16)
    ax2 = plt.subplot2grid((20, 33), (10, 16), rowspan=10, colspan=16)
    legend_elements = [mlines.Line2D([], [], color=(215 / 255, 48 / 255, 39 / 255, 0.65),
                                     marker='o', linestyle='None',
                                     markersize=8, label='1 Node', markeredgecolor='k',
                                     markeredgewidth=0.5),
                       mlines.Line2D([], [], color=(44 / 255, 162 / 255, 95 / 255, 0.65),
                                     marker='o', linestyle='None',
                                     markersize=8, label='2 Nodes', markeredgecolor='k',
                                     markeredgewidth=0.5),
                       mlines.Line2D([], [], color=(244 / 255, 109 / 255, 67 / 255, 0.65),
                                     marker='o', linestyle='None',
                                     markersize=8, label='3 Nodes', markeredgecolor='k',
                                     markeredgewidth=0.5),
                       mlines.Line2D([], [], color=(253 / 255, 174 / 255, 97 / 255, 0.65),
                                     marker='o', linestyle='None',
                                     markersize=8, label='4 Nodes', markeredgecolor='k',
                                     markeredgewidth=0.5),
                       mlines.Line2D([], [], color=(171 / 255, 217 / 255, 233 / 255, 0.65),
                                     marker='o', linestyle='None',
                                     markersize=8, label='5-7 Nodes', markeredgecolor='k',
                                     markeredgewidth=0.5),
                       mlines.Line2D([], [], color=(116 / 255, 173 / 255, 209 / 255, 0.65),
                                     marker='o', linestyle='None',
                                     markersize=8, label='8-422 Nodes', markeredgecolor='k',
                                     markeredgewidth=0.5),
                       mlines.Line2D([], [], color=(69 / 255, 117 / 255, 180 / 255, 0.65),
                                     marker='o', linestyle='None',
                                     markersize=8, label='433 Nodes', markeredgecolor='k',
                                     markeredgewidth=0.5)]
    pos = graphviz_layout(G, prog="neato")
    C = (G.subgraph(c) for c in nx.connected_components(G))
    colors1 = ['#d73027', '#2ca25f', '#f46d43', '#fdae61',
               '#abd9e9', '#74add1', '#4575b4']
    for g in C:
        if nx.number_of_nodes(g) == 1:
            c = colors1[0]
        elif nx.number_of_nodes(g) == 2:
            c = colors1[1]
        elif nx.number_of_nodes(g) == 3:
            c = colors1[2]
        elif nx.number_of_nodes(g) == 4:
            c = colors1[3]
        elif (nx.number_of_nodes(g) > 4) and nx.number_of_nodes(g) < 8:
            c = colors1[4]
        elif (nx.number_of_nodes(g) > 7) and nx.number_of_nodes(g) < 400:
            c = colors1[5]
        elif (nx.number_of_nodes(g) > 400):
            c = colors1[6]
        nx.draw(g, pos, node_size=28, node_color=c, vmin=0.0,
                vmax=1.0, with_labels=False, ax=ax, alpha=0.65)
    ax.set_title('A.', fontsize=24, loc='left', y=0.9, x=0, **csfont)
    ax.legend(handles=legend_elements, loc='lower center', frameon=True,
              fontsize=10, framealpha=1, edgecolor='k', ncol=7,
              bbox_to_anchor=(0.5, -0.04))

    # Subfigures B and C
    legend_elements = [mlines.Line2D([], [], color=(215 / 255, 48 / 255, 39 / 255, 0.8),
                                     marker='o', linestyle='None',
                                     markersize=9, label='Female', markeredgecolor='k',
                                     markeredgewidth=0.5),
                       mlines.Line2D([], [], color=(69 / 255, 117 / 255, 180 / 255, 0.8),
                                     marker='o', linestyle='None',
                                     markersize=9, label='Male', markeredgecolor='k',
                                     markeredgewidth=0.5),
                       mlines.Line2D([], [], color=(211 / 255, 211 / 255, 211 / 255, 0.8),
                                     marker='o', linestyle='None',
                                     markersize=9, label='Unknown', markeredgecolor='k',
                                     markeredgewidth=0.5)]
    Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
    G0 = G.subgraph(Gcc[0])
    pos = graphviz_layout(G0, prog="neato", args='-Gnodesep=0.5')
    C = (G.subgraph(c) for c in nx.connected_components(G0))
    gend_df = pd.merge(authors_df[['authorid', 'degree']].reset_index(),
                       author_papers[['authorid',
                                      'clean_gender']].drop_duplicates(),
                       how='left', left_on='authorid', right_on='authorid')
    color_list = []
    size_list = []
    for g in C:
        for c in g.nodes:
            if gend_df.loc[c, 'clean_gender'] == 'female':
                color_list.append((215 / 255, 48 / 255, 39 / 255))
            elif gend_df.loc[c, 'clean_gender'] == 'male':
                color_list.append((69 / 255, 117 / 255, 180 / 255))
            else:
                color_list.append((211 / 255, 211 / 255, 211 / 255))
            size_list.append(10 + (3 * gend_df.loc[c, 'degree']))
        nx.draw(g, pos, node_size=size_list, node_color=color_list,
                vmin=0.0, vmax=1.0, with_labels=False, ax=ax1, alpha=0.8,
                width=0.5)

    G1 = G.subgraph(Gcc[1])
    pos = graphviz_layout(G1, prog="neato")
    C = (G.subgraph(c) for c in nx.connected_components(G1))
    color_list = []
    size_list = []
    for g in C:
        for c in g.nodes:
            if gend_df.loc[c, 'clean_gender'] == 'female':
                color_list.append((215 / 255, 48 / 255, 39 / 255))
            elif gend_df.loc[c, 'clean_gender'] == 'male':
                color_list.append((69 / 255, 117 / 255, 180 / 255))
            else:
                color_list.append((211 / 255, 211 / 255, 211 / 255))
            size_list.append(20 + (12 * gend_df.loc[c, 'degree']))
        nx.draw(g, pos, node_size=size_list, node_color=color_list,
                vmin=0.0, vmax=1.0, with_labels=False, ax=ax2, alpha=0.8,
                width=0.5)
    ax1.legend(handles=legend_elements, loc='lower right', frameon=True,
               fontsize=10, framealpha=1, edgecolor='k', bbox_to_anchor=(1, 0.0))
    ax2.legend(handles=legend_elements, loc='lower right', frameon=True,
               fontsize=10, framealpha=1, edgecolor='k', bbox_to_anchor=(0.915, 0.0))
    ax1.set_title('B.', fontsize=24, loc='left', y=0.9, x=0.0, **csfont)
    ax2.set_title('C.', fontsize=24, loc='left', y=0.9, x=0.0, **csfont)
    plt.subplots_adjust(wspace=0.25, hspace=.5)
    plt.savefig(os.path.join(fig_path, 'networks_combined.pdf'),
                bbox_inches='tight')
    plt.savefig(os.path.join(fig_path, 'networks_combined.png'),
                bbox_inches='tight', dpi=600)
    plt.savefig(os.path.join(fig_path, 'networks_combined.svg'),
                bbox_inches='tight')


def value_to_color(val):
    val_position = float((val - color_min)) / (color_max - color_min)
    ind = int(val_position * (n_colors - 1))
    return palette[ind]

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def visualise_mallet(mallet_topics):
    fig = plt.figure(figsize=(16, 14), tight_layout=True)
    ax1 = plt.subplot2grid((5, 4), (0, 0), rowspan=5, colspan=2)
    ax2 = plt.subplot2grid((5, 4), (0, 2), rowspan=1, colspan=1)
    ax3 = plt.subplot2grid((5, 4), (0, 3), rowspan=1, colspan=1)
    ax4 = plt.subplot2grid((5, 4), (1, 2), rowspan=1, colspan=1)
    ax5 = plt.subplot2grid((5, 4), (1, 3), rowspan=1, colspan=1)
    ax6 = plt.subplot2grid((5, 4), (2, 2), rowspan=1, colspan=1)
    ax7 = plt.subplot2grid((5, 4), (2, 3), rowspan=1, colspan=1)
    ax8 = plt.subplot2grid((5, 4), (3, 2), rowspan=1, colspan=1)
    ax9 = plt.subplot2grid((5, 4), (3, 3), rowspan=1, colspan=1)
    ax10 = plt.subplot2grid((5, 4), (4, 2), rowspan=1, colspan=1)
    ax11 = plt.subplot2grid((5, 4), (4, 3), rowspan=1, colspan=1)

    mallet_df = mallet_topics
    word_list = []
    for col in mallet_df.columns:
        temp_df = mallet_df.sort_values(by=col, ascending=False)
        for index in temp_df.index[0:3]:
            if index not in word_list:
                word_list.append(index)
    heatmap_df = mallet_df.T[word_list]
    heatmap_df['topic_sum'] = heatmap_df.sum(axis=1)
    heatmap_df = heatmap_df.sort_values(by='topic_sum',ascending=False)
    heatmap_df = heatmap_df.drop('topic_sum', axis=1)#.T
    n_colors = 256
    palette = sns.diverging_palette(220, 20, n=n_colors)
    color_min, color_max = [-heatmap_df.max().max(), heatmap_df.max().max()]
    corr = pd.melt(heatmap_df.reset_index(), id_vars='index')
    corr.columns = ['x', 'y', 'value']
    x = corr['x']
    y = corr['y']
    size = corr['value'].abs().astype(np.float64)
    x_labels = [v for v in x.unique()]
    y_labels = [v for v in y.unique()[::-1]]
    x_to_num = {p[1]: p[0] for p in enumerate(x_labels)}
    y_to_num = {p[1]: p[0] for p in enumerate(y_labels)}
    size_scale = 4000
    cmap = plt.get_cmap('RdYlBu_r')
    new_cmap = truncate_colormap(cmap, 0.15, 0.85)

    blues_cmap = plt.get_cmap('Blues')
    new_blues = truncate_colormap(blues_cmap, 0.25, 0.85)

    # define top and bottom colormaps
#    top = cm.get_cmap('Oranges_r', 128) # r means reversed version
#    bottom = cm.get_cmap('Blues', 128)# combine it all
#    newcolors = np.vstack((top(np.linspace(0, 1, 128)),
#                           bottom(np.linspace(0, 1, 128))))# create a new colormaps with a name of OrangeBlue
#    orange_blue = ListedColormap(newcolors, name='OrangeBlue')

    ax1.scatter(x=x.map(x_to_num), y=y.map(y_to_num), s=((size ** 0.7) * size_scale),
                c=corr['value'], cmap=new_blues, marker='o', alpha=0.99,
                edgecolor='k', linewidth=0.2)
    ax1.set_xticks([x_to_num[v] for v in x_labels])
    ax1.set_xticklabels(x_labels, rotation=0)
    ax1.set_yticks([y_to_num[v] for v in y_labels])
    ax1.set_yticklabels(y_labels)
    ax1.grid(False, 'minor', alpha=0, zorder=100)
#    ax1.xaxis.grid(linestyle='--', alpha=0.1)
    #    ax1.yaxis.grid(linestyle='--', alpha=0.1)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=90)
    sns.despine(ax=ax1, left=False, right=True, bottom=False, top=True)
    for label in ax1.get_yticklabels():
        label.set_fontproperties('Helvetica')
    for label in ax1.get_xticklabels():
        label.set_fontproperties('Helvetica')


    ax1.set_ylim(-2.5, 51)
    ax1.spines['left'].set_bounds(-1, 50)
    ax1.set_xlim(-1, 20)
    ax1.spines['bottom'].set_bounds(0, 19)
    ax1.set_title('A.', fontsize=24, loc='left', y=1, x=-0.05, **csfont)
    ax2.set_title('B.', fontsize=24, loc='left', y=0.979, x=-0.15, **csfont)

    ax2.yaxis.set_label_position("right")
    ax2.set_ylabel(heatmap_df.index[0], rotation=270)
    ax3.yaxis.set_label_position("right")
    ax3.set_ylabel(heatmap_df.index[1], rotation=270)
    ax4.yaxis.set_label_position("right")
    ax4.set_ylabel(heatmap_df.index[2], rotation=270)
    ax5.yaxis.set_label_position("right")
    ax5.set_ylabel(heatmap_df.index[3], rotation=270)
    ax6.yaxis.set_label_position("right")
    ax6.set_ylabel(heatmap_df.index[4], rotation=270)
    ax7.yaxis.set_label_position("right")
    ax7.set_ylabel(heatmap_df.index[5], rotation=270)
    ax8.yaxis.set_label_position("right")
    ax8.set_ylabel(heatmap_df.index[6], rotation=270)
    ax9.yaxis.set_label_position("right")
    ax9.set_ylabel(heatmap_df.index[7], rotation=270)
    ax10.yaxis.set_label_position("right")
    ax10.set_ylabel(heatmap_df.index[8], rotation=270)
    ax11.yaxis.set_label_position("right")
    ax11.set_ylabel(heatmap_df.index[9], rotation=270)


    for ax, topic in zip([ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9,ax10,ax11], range(0,11)):
        topics = mallet_topics[heatmap_df.index[topic]]
        col = heatmap_df.index[topic]
        topics = topics.sort_values(ascending=False)[0:5]
        topics = topics.sort_values(ascending=True)
        topics = pd.DataFrame(topics)
        sns.barplot(x=col, y=topics.index.values, data=topics, ax=ax,
                    palette=new_cmap(topics[col] * 9.5),
                    edgecolor='k', alpha=0.9)
        for p in ax.patches:
            p.set_height(0.625)
            width = p.get_width()  # get bar length
            ax.text(width + ax.get_xlim()[1]/30,  # set the text at 1 unit right of the bar
                    p.get_y() + p.get_height() / 2,  # get Y coordinate + X coordinate / 2
                    '{:1.2f}'.format(width),  # set variable to display, 2 decimals
                    ha='left',  # horizontal alignment
                    va='center')  # vertical alignment
        sns.despine(ax=ax)
        ax.set_xlim(ax.get_xlim()[0], ax.get_xlim()[1] + (ax.get_xlim()[1]/8))
        ax.set_xlabel('')
        ax.yaxis.label.set_size(12)
        ax.set_ylim(-0.7, 5)
        ax.spines['left'].set_bounds(0, 4)

    ax10.set_xlabel('Weight')
    ax11.set_xlabel('Weight')
    plt.tight_layout(True)
    fig_path = os.path.join(os.getcwd(), '..', 'article', 'figures')
    plt.savefig(os.path.join(fig_path, 'mallet_output.pdf'), transparent=False)
    plt.savefig(os.path.join(fig_path, 'mallet_output.png'), transparent=False)
    plt.savefig(os.path.join(fig_path, 'mallet_output.svg'), transparent=False, format='svg')


def describe_norms(main_df, auth_df):
    main_df['pages'] = main_df['pageend'] - main_df['pagestart'] + 1
    colors1 = ['#1f78b4', '#F7B706', '#d73027', '#0B6623']
    fig = plt.figure(figsize=(14, 10), tight_layout=True)
    ax = plt.subplot2grid((8, 14), (0, 0), rowspan=5, colspan=14)
    ax2 = plt.subplot2grid((8, 14), (5, 0), rowspan=3, colspan=7)
    ax3 = plt.subplot2grid((8, 14), (5, 7), rowspan=3, colspan=7)

    main_df['prismcoverdate'] = pd.to_datetime(main_df['prismcoverdate'])
    main_df = main_df.sort_values(by='prismcoverdate', ascending=True)
    main_df[['prismcoverdate', 'pages']].plot(x='prismcoverdate', y='pages',
                                              kind='scatter', color='w',
                                              edgecolor='k', linewidth=.3, ax=ax, s=38)
    glass_mean = main_df[:845]['pages'].mean()
    glass_max = main_df[:845]['pages'].max()
    glass_std = main_df[:845]['pages'].std()
    grebenik_mean = main_df[845:1344]['pages'].mean()
    grebenik_max = main_df[845:1344]['pages'].max()
    grebenik_std = main_df[845:1344]['pages'].std()
    Simons_mean = main_df[1344:1794]['pages'].mean()
    Simons_max = main_df[1344:1794]['pages'].max()
    Simons_std = main_df[1344:1794]['pages'].std()
    ermisch_mean = main_df[1794:]['pages'].mean()
    ermisch_max = main_df[1794:]['pages'].max()
    ermisch_std = main_df[1794:]['pages'].std()
    all_mean = main_df['pages'].mean()
    all_max = main_df['pages'].max()
    all_std = main_df['pages'].std()

    ax.hlines(glass_mean + 2 * glass_std, np.datetime64('1947-01-01'), np.datetime64('1979-03-01'),
              linestyle='--', linewidth=0.75, alpha=0.55, color=colors1[0])
    ax.hlines(glass_mean - 2 * glass_std, np.datetime64('1947-01-01'), np.datetime64('1979-03-01'),
              linestyle='--', linewidth=0.75, alpha=0.55, color=colors1[0])
    ax.fill_between(main_df['prismcoverdate'][0:845],
                    glass_mean - 2 * glass_std, glass_mean + 2 * glass_std,
                    alpha=0.065, color=colors1[0])
    ax.hlines(grebenik_mean + 2 * grebenik_std, np.datetime64('1979-03-01'), np.datetime64('1997-03-01'),
              linestyle='--', linewidth=0.75, alpha=0.55, color=colors1[1])
    ax.hlines(grebenik_mean - 2 * grebenik_std, np.datetime64('1979-03-01'), np.datetime64('1997-03-01'),
              linestyle='--', linewidth=0.75, alpha=0.55, color=colors1[1])
    ax.fill_between(main_df['prismcoverdate'][845:1344],
                    grebenik_mean - 2 * grebenik_std, grebenik_mean + 2 * grebenik_std,
                    alpha=0.065, color=colors1[1])
    ax.hlines(Simons_mean + 2 * Simons_std, np.datetime64('1997-03-01'), np.datetime64('2017-01-01'),
              linestyle='--', linewidth=0.75, alpha=0.55, color=colors1[2])
    ax.hlines(Simons_mean - 2 * Simons_std, np.datetime64('1997-03-01'), np.datetime64('2017-01-01'),
              linestyle='--', linewidth=0.75, alpha=0.55, color=colors1[2])
    ax.fill_between(main_df['prismcoverdate'][1344:1790],
                    Simons_mean - 2 * Simons_std, Simons_mean + 2 * Simons_std,
                    alpha=0.065, color=colors1[2])
    ax.hlines(ermisch_mean + 2 * ermisch_std, np.datetime64('2017-01-01'), np.datetime64('2020-09-01'),
              linestyle='--', linewidth=0.75, alpha=0.55, color=colors1[3])
    ax.hlines(ermisch_mean - 2 * ermisch_std, np.datetime64('2017-01-01'), np.datetime64('2020-09-01'),
              linestyle='--', linewidth=0.75, alpha=0.55, color=colors1[3])
    ax.fill_between(main_df['prismcoverdate'][1790:],
                    ermisch_mean - 2 * ermisch_std, ermisch_mean + 2 * ermisch_std,
                    alpha=0.065, color=colors1[3])
    ax.set_xlabel('')
    ax.set_ylabel('Page Length (pages)', fontsize=12, **csfont)
    ax.set_ylim(-5, 75)
    ax.set_xlim(np.datetime64('1946-01-01'), np.datetime64('2022-01-01'))

    xmin, xmax = (-8250, 3250)
    xspan = xmax - xmin
    ax_xmin, ax_xmax = ax.get_xlim()
    xax_span = ax_xmax - ax_xmin
    ymin, ymax = ax.get_ylim()
    yspan = ymax - ymin
    resolution = int(xspan / xax_span * 100) * 2 + 1  # guaranteed uneven
    beta = 300. / xax_span  # the higher this is, the smaller the radius
    x = np.linspace(xmin, xmax, resolution)
    x_half = x[:resolution // 2 + 1]
    y_half_brace = (1 / (1. + np.exp(-beta * (x_half - x_half[0])))
                    + 1 / (1. + np.exp(-beta * (x_half - x_half[-1]))))
    y = np.concatenate((y_half_brace, y_half_brace[-2::-1]))
    y = ymin + (.05 * y - .01) * yspan + 72.5  # adjust vertical position
    ax.plot(x, y, color='black', lw=1)
    ax.text((xmax + xmin) / 2., ymin + .07 * yspan + 72.5, 'Glass and Grebenik',
            ha='center', va='bottom', **csfont, fontsize=14)

    ax.vlines(3350, -5, 68.4, linewidth=1.2, linestyle='--', color='#d3d3d3')

    xmin, xmax = (3450, 9800)
    xspan = xmax - xmin
    ax_xmin, ax_xmax = ax.get_xlim()
    xax_span = ax_xmax - ax_xmin
    ymin, ymax = ax.get_ylim()
    yspan = ymax - ymin
    resolution = int(xspan / xax_span * 100) * 2 + 1  # guaranteed uneven
    beta = 300. / xax_span  # the higher this is, the smaller the radius
    x = np.linspace(xmin, xmax, resolution)
    x_half = x[:resolution // 2 + 1]
    y_half_brace = (1 / (1. + np.exp(-beta * (x_half - x_half[0])))
                    + 1 / (1. + np.exp(-beta * (x_half - x_half[-1]))))
    y = np.concatenate((y_half_brace, y_half_brace[-2::-1]))
    y = ymin + (.05 * y - .01) * yspan + 72.50  # adjust vertical position
    ax.plot(x, y, color='black', lw=1)
    ax.text((xmax + xmin) / 2., ymin + .07 * yspan + 72.5, 'Grebenik',
            ha='center', va='bottom', **csfont, fontsize=14)

    ax.vlines(9900, -5, 68.4, linewidth=1.2, linestyle='--', color='#d3d3d3')
    xmin, xmax = (10000, 17150)
    xspan = xmax - xmin
    ax_xmin, ax_xmax = ax.get_xlim()
    xax_span = ax_xmax - ax_xmin
    ymin, ymax = ax.get_ylim()
    yspan = ymax - ymin
    resolution = int(xspan / xax_span * 100) * 2 + 1  # guaranteed uneven
    beta = 300. / xax_span  # the higher this is, the smaller the radius
    x = np.linspace(xmin, xmax, resolution)
    x_half = x[:resolution // 2 + 1]
    y_half_brace = (1 / (1. + np.exp(-beta * (x_half - x_half[0])))
                    + 1 / (1. + np.exp(-beta * (x_half - x_half[-1]))))
    y = np.concatenate((y_half_brace, y_half_brace[-2::-1]))
    y = ymin + (.05 * y - .01) * yspan + 72.5  # adjust vertical position
    ax.plot(x, y, color='black', lw=1)
    ax.text((xmax + xmin) / 2., ymin + .07 * yspan + 72.5, 'Simons', ha='center', va='bottom', **csfont, fontsize=14)

    ax.vlines(17250, -5, 68.4, linewidth=1.2, linestyle='--', color='#d3d3d3')
    xmin, xmax = (17350, 18700)
    xspan = xmax - xmin
    ax_xmin, ax_xmax = ax.get_xlim()
    xax_span = ax_xmax - ax_xmin
    ymin, ymax = ax.get_ylim()
    yspan = ymax - ymin
    resolution = int(xspan / xax_span * 100) * 2 + 1  # guaranteed uneven
    beta = 300. / xax_span  # the higher this is, the smaller the radius
    x = np.linspace(xmin, xmax, resolution)
    x_half = x[:resolution // 2 + 1]
    y_half_brace = (1 / (1. + np.exp(-beta * (x_half - x_half[0])))
                    + 1 / (1. + np.exp(-beta * (x_half - x_half[-1]))))
    y = np.concatenate((y_half_brace, y_half_brace[-2::-1]))
    y = ymin + (.05 * y - .01) * yspan + 72.5  # adjust vertical position
    ax.plot(x, y, color='black', lw=1)
    ax.text((xmax + xmin) / 2., ymin + .07 * yspan + 72.5, 'Ermisch', ha='center', va='bottom', **csfont, fontsize=14);

    ax.vlines(18800, -5, 68.4, linewidth=1.2, linestyle='--', color='#d3d3d3')

    temp_auth = auth_df.drop_duplicates(subset=['doi', 'authorid'])
    glass_auth = temp_auth[auth_df['prismcoverdate'] < '1979-03-01']
    grebenik_auth = temp_auth[(temp_auth['prismcoverdate'] >= '1979-03-01') |
                            (temp_auth['prismcoverdate'] < '1997-03-01')]
    simons_auth = temp_auth[temp_auth['prismcoverdate'] >= '1997-03-01']
    glass_mean_auth = glass_auth.groupby(['doi'])['doi'].count().mean()
    glass_std_auth = glass_auth.groupby(['doi'])['doi'].count().std()
    grebenik_mean_auth = grebenik_auth.groupby(['doi'])['doi'].count().mean()
    grebenik_std_auth = grebenik_auth.groupby(['doi'])['doi'].count().std()
    simons_mean_auth = simons_auth.groupby(['doi'])['doi'].count().mean()
    simons_std_auth = simons_auth.groupby(['doi'])['doi'].count().std()
    auth_short = pd.merge(temp_auth .groupby(['doi'])['doi'].count().reset_index(name='count'),
                          temp_auth [['doi', 'prismcoverdate']].drop_duplicates(), how='left', left_on='doi',
                          right_on='doi')
    auth_short = auth_short[['count', 'prismcoverdate']].sort_values(by='prismcoverdate')
    av_auth = pd.DataFrame(index=range(1947, 2021), columns=['av_auth',
                                                             'av_auth_+stdev',
                                                             'av_auth_-stdev',
                                                             'pc_solo'])
    for year in range(1948, 2020):
        auth_year = auth_short[(auth_short['prismcoverdate'].str.contains(str(year))) |
                               (auth_short['prismcoverdate'].str.contains(str(year - 1))) |
                               (auth_short['prismcoverdate'].str.contains(str(year + 1)))
                               ]
        av_auth.loc[year, 'pc_solo'] = len(auth_year[auth_year['count'] == 1]) / len(auth_year)
        av_auth.loc[year, 'av_auth'] = auth_year['count'].mean()

    av_auth[0:33][['av_auth']].plot(ax=ax2, legend=False, color=colors1[0])
    av_auth[32:51][['av_auth']].plot(ax=ax2, legend=False, color=colors1[1])
    av_auth[50:73][['av_auth']].plot(ax=ax2, legend=False, color=colors1[2])

    av_auth[0:33][['pc_solo']].plot(ax=ax3, legend=False, color=colors1[0])
    av_auth[32:51][['pc_solo']].plot(ax=ax3, legend=False, color=colors1[1])
    av_auth[50:73][['pc_solo']].plot(ax=ax3, legend=False, color=colors1[2])

    ax2.set_ylabel("Average Number of Authors", **csfont, fontsize=12)
    #    av_auth[['pc_solo']].plot(ax=ax3, legend=False)
    #    ax3.yaxis.set_label_position("right")
    ax3.set_ylabel("Fraction Solo Authored", **csfont, fontsize=12)

    ax.annotate('$\mu$=' + str(round(glass_mean, 1)) + ', $\sigma$=' + str(round(glass_std, 1)),
                xy=(2800, 47.5), xytext=(2800, 47.5), rotation=270, fontsize=12, **csfont)
    ax.annotate('$\mu$=' + str(round(grebenik_mean, 1)) + ', $\sigma$=' + str(round(grebenik_std, 1)),
                xy=(9350, 47.5), xytext=(9350, 47.5), rotation=270, fontsize=12, **csfont)
    ax.annotate('$\mu$=' + str(round(Simons_mean, 1)) + ', $\sigma$=' + str(round(Simons_std, 1)),
                xy=(16650, 47.5), xytext=(16650, 47.5), rotation=270, fontsize=12, **csfont)
    ax.annotate('$\mu$=' + str(round(ermisch_mean, 1)) + ', $\sigma$=' + str(round(ermisch_std, 1)),
                xy=(18275, 47.5), xytext=(18275, 47.5), rotation=270, fontsize=12, **csfont)
    ax.set_title('A.', fontsize=22, loc='left', y=1.01, x=-.04, **csfont)
    ax2.set_title('B.', fontsize=22, loc='left', y=1.025, x=-.04, **csfont)
    ax3.set_title('C.', fontsize=22, loc='left', y=1.025, x=-.04, **csfont)
    sns.despine(ax=ax)
    sns.despine(ax=ax2)
    sns.despine(ax=ax3)  # , left=True, right=False, top=True, bottom=False)

    ax2.set_ylim(0, 5.15)
    ax2.set_xlim(1946, 2020)

    xmin, xmax = (1947, 1979)
    xspan = xmax - xmin
    ax_xmin, ax_xmax = ax2.get_xlim()
    xax_span = ax_xmax - ax_xmin
    ymin, ymax = ax2.get_ylim()
    yspan = ymax - ymin
    resolution = int(xspan / xax_span * 100) * 2 + 1  # guaranteed uneven
    beta = 300. / xax_span  # the higher this is, the smaller the radius
    x = np.linspace(xmin, xmax, resolution)
    x_half = x[:resolution // 2 + 1]
    y_half_brace = (1 / (1. + np.exp(-beta * (x_half - x_half[0])))
                    + 1 / (1. + np.exp(-beta * (x_half - x_half[-1]))))
    y = np.concatenate((y_half_brace, y_half_brace[-2::-1]))
    y = ymin + (.05 * y - .01) * yspan + 4.5  # adjust vertical position
    ax2.plot(x, y, color='black', lw=1)
    ax2.text((xmax + xmin) / 2., ymin + .07 * yspan + 4.5, 'Glass and Grebenik',
             ha='center', va='bottom', **csfont, fontsize=11)
    ax2.vlines(1979, -0.15, 4.5, linewidth=1.2, linestyle='--', color='#d3d3d3')

    xmin, xmax = (1979, 1997)
    xspan = xmax - xmin
    ax_xmin, ax_xmax = ax2.get_xlim()
    xax_span = ax_xmax - ax_xmin
    ymin, ymax = ax2.get_ylim()
    yspan = ymax - ymin
    resolution = int(xspan / xax_span * 100) * 2 + 1  # guaranteed uneven
    beta = 300. / xax_span  # the higher this is, the smaller the radius
    x = np.linspace(xmin, xmax, resolution)
    x_half = x[:resolution // 2 + 1]
    y_half_brace = (1 / (1. + np.exp(-beta * (x_half - x_half[0])))
                    + 1 / (1. + np.exp(-beta * (x_half - x_half[-1]))))
    y = np.concatenate((y_half_brace, y_half_brace[-2::-1]))
    y = ymin + (.05 * y - .01) * yspan + 4.5  # adjust vertical position
    ax2.plot(x, y, color='black', lw=1)
    ax2.text((xmax + xmin) / 2., ymin + .07 * yspan + 4.5, 'Grebenik',
             ha='center', va='bottom', **csfont, fontsize=11)
    ax2.vlines(1997, -0.15, 4.5, linewidth=1.2, linestyle='--', color='#d3d3d3')

    xmin, xmax = (1997, 2019)
    xspan = xmax - xmin
    ax_xmin, ax_xmax = ax2.get_xlim()
    xax_span = ax_xmax - ax_xmin
    ymin, ymax = ax2.get_ylim()
    yspan = ymax - ymin
    resolution = int(xspan / xax_span * 100) * 2 + 1  # guaranteed uneven
    beta = 300. / xax_span  # the higher this is, the smaller the radius
    x = np.linspace(xmin, xmax, resolution)
    x_half = x[:resolution // 2 + 1]
    y_half_brace = (1 / (1. + np.exp(-beta * (x_half - x_half[0])))
                    + 1 / (1. + np.exp(-beta * (x_half - x_half[-1]))))
    y = np.concatenate((y_half_brace, y_half_brace[-2::-1]))
    y = ymin + (.05 * y - .01) * yspan + 4.5  # adjust vertical position
    ax2.plot(x, y, color='black', lw=1)
    ax2.text((xmax + xmin) / 2., ymin + .07 * yspan + 4.5, 'Simons/Ermisch',
             ha='center', va='bottom', **csfont, fontsize=11)
    ax2.vlines(2019, -0.15, 4.5, linewidth=1.2, linestyle='--', color='#d3d3d3')

    ax2.annotate('$\mu$=' + str(round(glass_mean_auth, 1)) + ' (' + str(round(glass_std_auth, 1)) + ')',
                 xy=(1976.25, 3.75), xytext=(1976.25, 2.8), rotation=270, fontsize=11, **csfont)
    ax2.annotate('$\mu$=' + str(round(grebenik_mean_auth, 1)) + ' (' + str(round(grebenik_std_auth, 1)) + ')',
                 xy=(1994.25, 3.75), xytext=(1994.25, 2.8), rotation=270, fontsize=11, **csfont)
    ax2.annotate('$\mu$=' + str(round(simons_mean_auth, 1)) + ' (' + str(round(simons_std_auth, 1)) + ')',
                 xy=(2016, 0.5), xytext=(2016, 0.5), rotation=270, fontsize=11, **csfont)

    glass_group = glass_auth.groupby(['doi'])['doi'].count().reset_index(name='count')
    glass_solo_auth = len(glass_group[glass_group['count'] == 1]) / len(glass_group)
    grebenik_group = grebenik_auth.groupby(['doi'])['doi'].count().reset_index(name='count')
    grebenik_solo_auth = len(grebenik_group[grebenik_group['count'] == 1]) / len(grebenik_group)
    simons_group = simons_auth.groupby(['doi'])['doi'].count().reset_index(name='count')
    simons_solo_auth = len(simons_group[simons_group['count'] == 1]) / len(simons_group)

    ax3.set_ylim(0.05, 1.1)
    ax3.set_xlim(1946, 2020)

    xmin, xmax = (1947, 1979)
    xspan = xmax - xmin
    ax_xmin, ax_xmax = ax3.get_xlim()
    xax_span = ax_xmax - ax_xmin
    ymin, ymax = ax3.get_ylim()
    yspan = ymax - ymin
    resolution = int(xspan / xax_span * 100) * 2 + 1  # guaranteed uneven
    beta = 300. / xax_span  # the higher this is, the smaller the radius
    x = np.linspace(xmin, xmax, resolution)
    x_half = x[:resolution // 2 + 1]
    y_half_brace = (1 / (1. + np.exp(-beta * (x_half - x_half[0])))
                    + 1 / (1. + np.exp(-beta * (x_half - x_half[-1]))))
    y = np.concatenate((y_half_brace, y_half_brace[-2::-1]))
    y = ymin + (.05 * y - .01) * yspan + .925  # adjust vertical position
    ax3.plot(x, y, color='black', lw=1)
    ax3.text((xmax + xmin) / 2., ymin + .07 * yspan + .925, 'Glass and Grebenik',
             ha='center', va='bottom', **csfont, fontsize=11)
    ax3.vlines(1979, 0.0, 0.925, linewidth=1.2, linestyle='--', color='#d3d3d3')

    xmin, xmax = (1979, 1997)
    xspan = xmax - xmin
    ax_xmin, ax_xmax = ax3.get_xlim()
    xax_span = ax_xmax - ax_xmin
    ymin, ymax = ax3.get_ylim()
    yspan = ymax - ymin
    resolution = int(xspan / xax_span * 100) * 2 + 1  # guaranteed uneven
    beta = 300. / xax_span  # the higher this is, the smaller the radius
    x = np.linspace(xmin, xmax, resolution)
    x_half = x[:resolution // 2 + 1]
    y_half_brace = (1 / (1. + np.exp(-beta * (x_half - x_half[0])))
                    + 1 / (1. + np.exp(-beta * (x_half - x_half[-1]))))
    y = np.concatenate((y_half_brace, y_half_brace[-2::-1]))
    y = ymin + (.05 * y - .01) * yspan + .925  # adjust vertical position
    ax3.plot(x, y, color='black', lw=1)
    ax3.text((xmax + xmin) / 2., ymin + .07 * yspan + .925, 'Grebenik',
             ha='center', va='bottom', **csfont, fontsize=11)
    ax3.vlines(1997, 0, .925, linewidth=1.2, linestyle='--', color='#d3d3d3')

    xmin, xmax = (1997, 2019)
    xspan = xmax - xmin
    ax_xmin, ax_xmax = ax3.get_xlim()
    xax_span = ax_xmax - ax_xmin
    ymin, ymax = ax3.get_ylim()
    yspan = ymax - ymin
    resolution = int(xspan / xax_span * 100) * 2 + 1  # guaranteed uneven
    beta = 300. / xax_span  # the higher this is, the smaller the radius
    x = np.linspace(xmin, xmax, resolution)
    x_half = x[:resolution // 2 + 1]
    y_half_brace = (1 / (1. + np.exp(-beta * (x_half - x_half[0])))
                    + 1 / (1. + np.exp(-beta * (x_half - x_half[-1]))))
    y = np.concatenate((y_half_brace, y_half_brace[-2::-1]))
    y = ymin + (.05 * y - .01) * yspan + .925  # adjust vertical position
    ax3.plot(x, y, color='black', lw=1)
    ax3.text((xmax + xmin) / 2., ymin + .07 * yspan + .925, 'Simons/Ermisch',
             ha='center', va='bottom', **csfont, fontsize=11)
    ax3.vlines(2019, 0.0, 0.925, linewidth=1.2, linestyle='--', color='#d3d3d3')

    ax3.annotate('1947-79: ' + str(int(glass_solo_auth * 100)) + '%',
                 xy=(1976.25, .1), xytext=(1976.25, .09), rotation=270, fontsize=9, **csfont)
    ax3.annotate('1979-97: ' + str(int(grebenik_solo_auth * 100)) + '%',
                 xy=(1994.25, .1), xytext=(1994.25, .09), rotation=270, fontsize=9, **csfont)
    ax3.annotate('1997-2020: ' + str(int(simons_solo_auth * 100)) + '%',
                 xy=(2016, 0.75), xytext=(2016, 0.55), rotation=270, fontsize=9, **csfont)

    # ax3.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    # this takes out tight layout...

    fig_path = os.path.join(os.getcwd(), '..', 'article', 'figures')
    plt.savefig(os.path.join(fig_path, 'norms_over_time.pdf'),
                bbox_inches='tight')
    plt.savefig(os.path.join(fig_path, 'norms_over_time.png'), dpi=400,
                bbox_inches='tight')
    plt.savefig(os.path.join(fig_path, 'norms_over_time.svg'),
                bbox_inches='tight')


def keyword_tabulation(main_df):
    keyword_series = main_df[main_df['keywords'].notnull()]['keywords']
    print('We have keywords for ' + \
          str(len(keyword_series)) + ' papers')
    keyword_list = pd.DataFrame(index=[], columns=['keyword'])
    counter = 0
    for paper in keyword_series:
        for keyword in paper.lower().split(';'):
            counter += 1
            keyword_list.loc[counter, 'keyword'] = keyword.strip()
    print('We have a total of ' + str(len(keyword_list)) + ' keywords')
    grouped = keyword_list.groupby(['keyword'])['keyword'].count()
    print('\nThe top 10 keywords are:')
    print(grouped.sort_values(ascending=False)[0:10])


def simple_continental_analysis(main_df, figure_path):
    topic_cont = pd.DataFrame(index=[],
                              columns=['Simple_Topic', 'Continent'])
    counter = 0
    for index, row in main_df.iterrows():
        for topic in row['Topic'].split(';'):
            topic_cont.loc[counter, 'Simple_Topic'] = topic
            topic_cont.loc[counter, 'Continent'] = row['Continent']
            counter += 1
    topic_cont['Simple_Topic'] = topic_cont['Simple_Topic'].str.extract('(\d+)', expand=False)
    topic_cont['Simple_Topic'] = 'Topic ' + topic_cont['Simple_Topic']
    topic_cont = topic_cont[(topic_cont['Simple_Topic'].notnull()) & (topic_cont['Continent'].notnull())]
    topic_count = pd.DataFrame(index=topic_cont['Simple_Topic'].sort_values().unique(),
                               columns=topic_cont['Continent'].unique())
    for column in topic_count.columns:
        for index in topic_count.index:
            topic_count.loc[index, column] = len(topic_cont[(topic_cont['Simple_Topic'] == index) &
                                                            (topic_cont['Continent'] == column)])
        topic_count[column] = topic_count[column] / topic_count[column].sum()
    topic_count = topic_count.round(1)
    topic_count.to_csv(os.path.join(figure_path, '..', 'tables', 'continental_analysis_simple.csv'))
    print(topic_count.to_string())


def split_continental_analysis(main_df, figure_path):
    topic_cont = pd.DataFrame(index=[],
                              columns=['Topic', 'Continent'])
    counter = 0
    topic_set = set()
    for index, row in main_df.iterrows():
        for topic in row['Topic'].split(';'):
            if len(topic.strip())>0:
                topic_set.add(topic)
                topic_cont.loc[counter, 'Topic'] = topic
                topic_cont.loc[counter, 'Continent'] = row['Continent']
                counter += 1
    topic_cont['Topic'] = 'Topic ' + topic_cont['Topic']
    topic_cont = topic_cont[(topic_cont['Topic'].notnull()) & (topic_cont['Continent'].notnull())]
    topic_count = pd.DataFrame(index=topic_cont['Topic'].sort_values().unique(),
                               columns=topic_cont['Continent'].unique())
    for column in topic_count.columns:
        for index in topic_count.index:
            topic_count.loc[index, column] = len(topic_cont[(topic_cont['Topic'] == index) &
                                                            (topic_cont['Continent'] == column)])
        topic_count[column] = topic_count[column] / topic_count[column].sum()
    topic_count = topic_count.round(1)
    topic_count.to_csv(os.path.join(figure_path, '..', 'tables', 'continental_analysis_split.csv'))
    print(topic_count.to_string())


def describe_lengths(main_df):
    main_df['pages'] = main_df['pageend'] - main_df['pagestart']
    colors1 = ['#1f78b4', '#fcdb81', '#d73027', '#0B6623']
    fig, ax = plt.subplots(figsize=(14, 5))
    main_df['prismcoverdate'] = pd.to_datetime(main_df['prismcoverdate'])
    main_df = main_df.sort_values(by='prismcoverdate', ascending=True)
    main_df[['prismcoverdate', 'pages']].plot(x='prismcoverdate', y='pages',
                                              kind='scatter', color='w',
                                              edgecolor='k', linewidth=.75, ax=ax)
    glass_mean = main_df[:845]['pages'].mean()
    glass_max = main_df[:845]['pages'].max()
    glass_std = main_df[:845]['pages'].std()
    grebenik_mean = main_df[845:1344]['pages'].mean()
    grebenik_max = main_df[845:1344]['pages'].max()
    grebenik_std = main_df[845:1344]['pages'].std()
    Simons_mean = main_df[1344:1794]['pages'].mean()
    Simons_max = main_df[1344:1794]['pages'].max()
    Simons_std = main_df[1344:1794]['pages'].std()
    ermisch_mean = main_df[1794:]['pages'].mean()
    ermisch_max = main_df[1794:]['pages'].max()
    ermisch_std = main_df[1794:]['pages'].std()
    all_mean = main_df['pages'].mean()
    all_max = main_df['pages'].max()
    all_std = main_df['pages'].std()

    ax.hlines(glass_mean + 2 * glass_std, '1947-01-01', '1979-03-01',
              linestyle='--', linewidth=0.75, alpha=0.55, color=colors1[0])
    ax.hlines(glass_mean - 2 * glass_std, '1947-01-01', '1979-03-01',
              linestyle='--', linewidth=0.75, alpha=0.55, color=colors1[0])
    ax.fill_between(main_df['prismcoverdate'][0:845],
                    glass_mean - 2 * glass_std, glass_mean + 2 * glass_std,
                    alpha=0.065, color=colors1[0])
    ax.hlines(grebenik_mean + 2 * grebenik_std, '1979-03-01', '1997-03-01',
              linestyle='--', linewidth=0.75, alpha=0.55, color=colors1[1])
    ax.hlines(grebenik_mean - 2 * grebenik_std, '1979-03-01', '1997-03-01',
              linestyle='--', linewidth=0.75, alpha=0.55, color=colors1[1])
    ax.fill_between(main_df['prismcoverdate'][845:1344],
                    grebenik_mean - 2 * grebenik_std, grebenik_mean + 2 * grebenik_std,
                    alpha=0.065, color=colors1[1])
    ax.hlines(Simons_mean + 2 * Simons_std, '1997-03-01', '2017-01-01',
              linestyle='--', linewidth=0.75, alpha=0.55, color=colors1[2])
    ax.hlines(Simons_mean - 2 * Simons_std, '1997-03-01', '2017-01-01',
              linestyle='--', linewidth=0.75, alpha=0.55, color=colors1[2])
    ax.fill_between(main_df['prismcoverdate'][1344:1790],
                    Simons_mean - 2 * Simons_std, Simons_mean + 2 * Simons_std,
                    alpha=0.065, color=colors1[2])
    ax.hlines(ermisch_mean + 2 * ermisch_std, '2017-01-01', '2020-09-01',
              linestyle='--', linewidth=0.75, alpha=0.55, color=colors1[3])
    ax.hlines(ermisch_mean - 2 * ermisch_std, '2017-01-01', '2020-09-01',
              linestyle='--', linewidth=0.75, alpha=0.55, color=colors1[3])
    ax.fill_between(main_df['prismcoverdate'][1790:],
                    ermisch_mean - 2 * ermisch_std, ermisch_mean + 2 * ermisch_std,
                    alpha=0.065, color=colors1[3])
    ax.set_xlabel('')
    ax.set_ylabel('Page Length (pages)', fontsize=12, **csfont)
    ax.set_ylim(-5, 75)
    ax.set_xlim('1946-01-01', '2022-01-01')

    xmin, xmax = (-8250, 3250)
    xspan = xmax - xmin
    ax_xmin, ax_xmax = ax.get_xlim()
    xax_span = ax_xmax - ax_xmin
    ymin, ymax = ax.get_ylim()
    yspan = ymax - ymin
    resolution = int(xspan / xax_span * 100) * 2 + 1  # guaranteed uneven
    beta = 300. / xax_span  # the higher this is, the smaller the radius
    x = np.linspace(xmin, xmax, resolution)
    x_half = x[:resolution // 2 + 1]
    y_half_brace = (1 / (1. + np.exp(-beta * (x_half - x_half[0])))
                    + 1 / (1. + np.exp(-beta * (x_half - x_half[-1]))))
    y = np.concatenate((y_half_brace, y_half_brace[-2::-1]))
    y = ymin + (.05 * y - .01) * yspan + 72.5  # adjust vertical position
    ax.plot(x, y, color='black', lw=1)
    ax.text((xmax + xmin) / 2., ymin + .07 * yspan + 72.5, 'Glass and Grebenik', ha='center', va='bottom', **csfont,
            fontsize=13)

    xmin, xmax = (3450, 9800)
    xspan = xmax - xmin
    ax_xmin, ax_xmax = ax.get_xlim()
    xax_span = ax_xmax - ax_xmin
    ymin, ymax = ax.get_ylim()
    yspan = ymax - ymin
    resolution = int(xspan / xax_span * 100) * 2 + 1  # guaranteed uneven
    beta = 300. / xax_span  # the higher this is, the smaller the radius
    x = np.linspace(xmin, xmax, resolution)
    x_half = x[:resolution // 2 + 1]
    y_half_brace = (1 / (1. + np.exp(-beta * (x_half - x_half[0])))
                    + 1 / (1. + np.exp(-beta * (x_half - x_half[-1]))))
    y = np.concatenate((y_half_brace, y_half_brace[-2::-1]))
    y = ymin + (.05 * y - .01) * yspan + 72.50  # adjust vertical position
    ax.plot(x, y, color='black', lw=1)
    ax.text((xmax + xmin) / 2., ymin + .07 * yspan + 72.5, 'Grebenik', ha='center', va='bottom', **csfont, fontsize=13)

    xmin, xmax = (10000, 17150)
    xspan = xmax - xmin
    ax_xmin, ax_xmax = ax.get_xlim()
    xax_span = ax_xmax - ax_xmin
    ymin, ymax = ax.get_ylim()
    yspan = ymax - ymin
    resolution = int(xspan / xax_span * 100) * 2 + 1  # guaranteed uneven
    beta = 300. / xax_span  # the higher this is, the smaller the radius
    x = np.linspace(xmin, xmax, resolution)
    x_half = x[:resolution // 2 + 1]
    y_half_brace = (1 / (1. + np.exp(-beta * (x_half - x_half[0])))
                    + 1 / (1. + np.exp(-beta * (x_half - x_half[-1]))))
    y = np.concatenate((y_half_brace, y_half_brace[-2::-1]))
    y = ymin + (.05 * y - .01) * yspan + 72.5  # adjust vertical position
    ax.plot(x, y, color='black', lw=1)
    ax.text((xmax + xmin) / 2., ymin + .07 * yspan + 72.5, 'Simons', ha='center', va='bottom', **csfont, fontsize=13)

    xmin, xmax = (17350, 18700)
    xspan = xmax - xmin
    ax_xmin, ax_xmax = ax.get_xlim()
    xax_span = ax_xmax - ax_xmin
    ymin, ymax = ax.get_ylim()
    yspan = ymax - ymin
    resolution = int(xspan / xax_span * 100) * 2 + 1  # guaranteed uneven
    beta = 300. / xax_span  # the higher this is, the smaller the radius
    x = np.linspace(xmin, xmax, resolution)
    x_half = x[:resolution // 2 + 1]
    y_half_brace = (1 / (1. + np.exp(-beta * (x_half - x_half[0])))
                    + 1 / (1. + np.exp(-beta * (x_half - x_half[-1]))))
    y = np.concatenate((y_half_brace, y_half_brace[-2::-1]))
    y = ymin + (.05 * y - .01) * yspan + 72.5  # adjust vertical position
    ax.plot(x, y, color='black', lw=1)
    ax.text((xmax + xmin) / 2., ymin + .07 * yspan + 72.5, 'Ermisch', ha='center', va='bottom', **csfont, fontsize=13);
    fig_path = os.path.join(os.getcwd(), '..', 'article', 'figures')
    sns.despine()
    plt.savefig(os.path.join(fig_path, 'pagelength_over_time.pdf'),
                bbox_inches='tight')
    plt.savefig(os.path.join(fig_path, 'pagelength_over_time.png'), dpi=400,
                bbox_inches='tight')
    plt.savefig(os.path.join(fig_path, 'pagelength_over_time.svg'),
                bbox_inches='tight')

    print('All-time: mean length was ' + \
          str(round(all_mean, 2)) + \
          ', stdev was ' + str(round(all_std, 2)) + \
          ', max length of ' + str(round(all_max, 2)))
    print('Glass and Grebenik: mean length was ' + \
          str(round(glass_mean, 2)) + \
          ', stdev was ' + str(round(glass_std, 2)) + \
          ', max length of ' + str(round(glass_max, 2)))
    print('Grebenik: mean length was ' + str(round(grebenik_mean, 2)) + \
          ', stdev was ' + str(round(grebenik_std, 2)) + \
          ', max length of ' + str(round(grebenik_max, 2)))
    print('Simons: mean length was ' + str(round(Simons_mean, 2)) + \
          ', stdev was ' + str(round(Simons_std, 2)) + \
          ', max length of ' + str(round(Simons_max, 2)))
    print('Ermisch: mean length was ' + str(round(ermisch_mean, 2)) + \
          ', stdev was ' + str(round(ermisch_std, 2)) + \
          ', max length of ' + str(round(ermisch_max, 2)))


def network_summary(G, Gcc):
    print('Edges in entire network: ' + str(G.number_of_edges()))
    print('Nodes in entire network: ' + str(G.number_of_nodes()))
    print('Density of entire network: ' + str(nx.density(G)))
    print('Edges in Giant Component: ' + str(G.subgraph(Gcc[0]).number_of_edges()))
    print('Nodes in Giant Component: ' + str(G.subgraph(Gcc[0]).number_of_nodes()))
    print('Density of Giant Component: ' + str(nx.density(G.subgraph(Gcc[0]))))


def authorship_country(auth_df, d_path):
    temp_auth = auth_df.drop_duplicates(subset=['doi', 'authorid'])
    cou_cou = auth_df.groupby(['aff_country'])['aff_country'].count()
    cou_cou = cou_cou.sort_values(ascending=False).reset_index(name='count')
    print('Number of unique countries from which authors write from: ' + \
          str(len(cou_cou)))
    holder_string = '\nThese are: '
    for index, row in cou_cou.iterrows():
        holder_string = holder_string + row['aff_country'] + ' (' + str(row['count']) + (')') + ', '
    print(holder_string[:-2])
    cou_cou[['aff_country']].to_csv(os.path.join(d_path, 'support', 'unique_countries.csv'))


def authorship_per_paper(auth_df):
    temp_auth = auth_df.drop_duplicates(subset=['doi', 'authorid'])
    for years in ['194', '195', '196', '197', '198', '199', '200', '201']:
        count_year = temp_auth[temp_auth['prismcoverdate'].str.contains(years)].groupby(['doi'])['doi'].count()
        print('Average number of authors per paper in the ' + years + '0s: ' + str(round(count_year.mean(), 3)))
    grouped_auth_count = temp_auth.groupby(['doi'])['doi'].count()
    grouped_auth_count = grouped_auth_count.reset_index(name='count')
    max_df = grouped_auth_count[grouped_auth_count['count'] == grouped_auth_count['count'].max()]
    max_doi = max_df.loc[max_df.index[0], 'doi']
    print('The most number of authors on one paper: ' +
          str(grouped_auth_count['count'].max()) +
          ' (DOI: ' + str(max_doi) + ')')
    print('The number of solo authored papers: ' + str(
        round(grouped_auth_count.groupby(['count'])['count'].count()[1], 3)))
    print('The number of papers with 2 authors is : ' + str(
        round(grouped_auth_count.groupby(['count'])['count'].count()[2], 3)))
    print('The number of papers with 3 authors is : ' + str(
        round(grouped_auth_count.groupby(['count'])['count'].count()[3], 3)))
    print('The number of papers with more than 3 authors is : ' + str(
        round(grouped_auth_count.groupby(['count'])['count'].count()[3:].sum(), 3)))
    for years in ['194', '195', '196', '197', '198', '199', '200', '201']:
        count_year = temp_auth[temp_auth['prismcoverdate'].str.contains(years)].groupby(['doi'])['doi'].count()
        num_solo = count_year.reset_index(name='count').groupby(['count'])['count'].count()[1]
        print('Percent of solo authored papers in the ' + years + '0s: ' + str(round(num_solo / len(count_year), 3)))


def make_network(auth_df):
    temp_auth = auth_df.drop_duplicates(subset=['doi', 'authorid'])
    temp_auth = temp_auth[temp_auth['doi'].notnull()]
    author_papers = temp_auth[temp_auth['authorid'].notnull()]
    authors_df = author_papers[['authorid', 'indexed_name']].drop_duplicates(subset=['authorid'])
    int_p_id = dict(enumerate(list(author_papers['doi'].unique())))
    int_a_id = dict(enumerate(list(author_papers['authorid'].unique())))
    a_int_id = {authorId: intVal for intVal, authorId in int_a_id.items()}
    p_int_id = {paperId: intVal for intVal, paperId in int_p_id.items()}
    author_paper_tuples = list(zip(author_papers['authorid'], author_papers['doi']))
    author_paper_tuples = [(a_int_id[t[0]], p_int_id[t[1]]) for t in author_paper_tuples]
    AP = sp.csc_matrix((np.ones(len(author_paper_tuples)), zip(*author_paper_tuples)))
    AA = AP.dot(AP.T)
    AA = np.array(AA - np.diag(AA.diagonal()))
    G = nx.from_numpy_matrix(AA, parallel_edges=True)
    deg_measure = nx.degree(G)
    cent_measure = nx.degree_centrality(G)
    bet_measure = nx.betweenness_centrality(G)
    authors_df['degree'] = authors_df['authorid'].apply(lambda l: deg_measure[a_int_id.get(l)])
    authors_df['degree_cent'] = authors_df['authorid'].apply(lambda l: cent_measure[a_int_id.get(l)])
    authors_df['degree_bet'] = authors_df['authorid'].apply(lambda l: bet_measure.get(a_int_id.get(l)))
    Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
    temp_auth['prismcoverdate'] = temp_auth['prismcoverdate'].astype('datetime64[ns]')
    time_df = pd.DataFrame(index=range(1947, 2021),
                           columns=['whole_edges', 'whole_nodes', 'whole_density',
                                    'giant_edges', 'giant_nodes', 'giant_density'])
    for year in range(1, 75):
        year_df = temp_auth[temp_auth['prismcoverdate'].dt.year < 1947 + year]
        year_df = year_df[year_df['doi'].notnull()]
        year_author_papers = year_df[year_df['authorid'].notnull()]
        year_authors_df = year_author_papers[['authorid', 'indexed_name']].drop_duplicates(subset=['authorid'])
        int_p_id = dict(enumerate(list(year_author_papers['doi'].unique())))
        int_a_id = dict(enumerate(list(year_author_papers['authorid'].unique())))
        a_int_id = {authorId: intVal for intVal, authorId in int_a_id.items()}
        p_int_id = {paperId: intVal for intVal, paperId in int_p_id.items()}
        author_paper_tuples = list(zip(year_author_papers['authorid'], year_author_papers['doi']))
        author_paper_tuples = [(a_int_id[t[0]], p_int_id[t[1]]) for t in author_paper_tuples]
        AP = sp.csc_matrix((np.ones(len(author_paper_tuples)), zip(*author_paper_tuples)))
        AA = AP.dot(AP.T)
        AA = np.array(AA - np.diag(AA.diagonal()))
        G = nx.from_numpy_matrix(AA, parallel_edges=True)
        time_df.loc[year + 1946, 'whole_edges'] = G.number_of_edges()
        time_df.loc[year + 1946, 'whole_nodes'] = G.number_of_nodes()
        time_df.loc[year + 1946, 'whole_density'] = nx.density(G)
        time_df.loc[year + 1946, 'giant_edges'] = G.subgraph(Gcc[0]).number_of_edges()
        time_df.loc[year + 1946, 'giant_nodes'] = G.subgraph(Gcc[0]).number_of_nodes()
        time_df.loc[year + 1946, 'giant_density'] = nx.density(G.subgraph(Gcc[0]))
    return G, Gcc, authors_df, author_papers


def plot_G0_and_G1(G, authors_df, author_papers, fig_path):
    fig = plt.figure(figsize=(12, 7), tight_layout=True)
    ax1 = plt.subplot2grid((1, 2), (0, 0), rowspan=1, colspan=1)
    ax2 = plt.subplot2grid((1, 2), (0, 1), rowspan=1, colspan=1)
    legend_elements = [mlines.Line2D([], [], color=(215 / 255, 48 / 255, 39 / 255, 0.8),
                                     marker='o', linestyle='None',
                                     markersize=10, label='Female', markeredgecolor='k',
                                     markeredgewidth=0.5),
                       mlines.Line2D([], [], color=(69 / 255, 117 / 255, 180 / 255, 0.8),
                                     marker='o', linestyle='None',
                                     markersize=10, label='Male', markeredgecolor='k',
                                     markeredgewidth=0.5),
                       mlines.Line2D([], [], color=(211 / 255, 211 / 255, 211 / 255, 0.8),
                                     marker='o', linestyle='None',
                                     markersize=10, label='Unknown', markeredgecolor='k',
                                     markeredgewidth=0.5)]
    Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
    G0 = G.subgraph(Gcc[0])
    pos = graphviz_layout(G0, prog="neato", args='-Gnodesep=0.5')
    C = (G.subgraph(c) for c in nx.connected_components(G0))
    gend_df = pd.merge(authors_df[['authorid', 'degree']].reset_index(),
                       author_papers[['authorid',
                                      'clean_gender']].drop_duplicates(),
                       how='left', left_on='authorid', right_on='authorid')
    color_list = []
    size_list = []
    for g in C:
        for c in g.nodes:
            if gend_df.loc[c, 'clean_gender'] == 'female':
                color_list.append((215 / 255, 48 / 255, 39 / 255))
            elif gend_df.loc[c, 'clean_gender'] == 'male':
                color_list.append((69 / 255, 117 / 255, 180 / 255))
            else:
                color_list.append((211 / 255, 211 / 255, 211 / 255))
            size_list.append(10 + (2 * gend_df.loc[c, 'degree']))
        nx.draw(g, pos, node_size=size_list, node_color=color_list,
                vmin=0.0, vmax=1.0, with_labels=False, ax=ax1, alpha=0.8,
                width=0.5)
    ax1.set_title('A.', fontsize=24, loc='left', y=0.98, x=0.1, **csfont)
    G1 = G.subgraph(Gcc[1])
    pos = graphviz_layout(G1, prog="neato")
    C = (G.subgraph(c) for c in nx.connected_components(G1))
    color_list = []
    size_list = []
    for g in C:
        for c in g.nodes:
            if gend_df.loc[c, 'clean_gender'] == 'female':
                color_list.append((215 / 255, 48 / 255, 39 / 255))
            elif gend_df.loc[c, 'clean_gender'] == 'male':
                color_list.append((69 / 255, 117 / 255, 180 / 255))
            else:
                color_list.append((211 / 255, 211 / 255, 211 / 255))
            size_list.append(20 + (10 * gend_df.loc[c, 'degree']))
        nx.draw(g, pos, node_size=size_list, node_color=color_list,
                vmin=0.0, vmax=1.0, with_labels=False, ax=ax2, alpha=0.8,
                width=0.5)
    ax2.set_title('B.', fontsize=24, loc='left', y=0.98, x=0.1, **csfont)
    ax2.legend(handles=legend_elements, loc='lower right', frameon=True,
               fontsize=12, framealpha=1, edgecolor='k', bbox_to_anchor=(1, 0.1))
    plt.savefig(os.path.join(fig_path, 'sub_networks.pdf'),
                bbox_inches='tight')
    plt.savefig(os.path.join(fig_path, 'sub_networks.png'),
                bbox_inches='tight', dpi=600)
    plt.savefig(os.path.join(fig_path, 'sub_networks.svg'),
                bbox_inches='tight')


def plot_G(G, fig_path):
    fig = plt.figure(figsize=(14, 8))
    ax1 = plt.subplot2grid((1, 1), (0, 0), rowspan=1, colspan=1)
    pos = graphviz_layout(G, prog="neato")
    C = (G.subgraph(c) for c in nx.connected_components(G))
    colors1 = ['#d73027', '#2ca25f', '#f46d43', '#fdae61',
               '#abd9e9', '#74add1', '#4575b4']
    isolate_counter = 0
    for g in C:
        if nx.number_of_nodes(g) == 1:
            c = colors1[0]
            isolate_counter = isolate_counter + 1
        elif nx.number_of_nodes(g) == 2:
            c = colors1[1]
        elif nx.number_of_nodes(g) == 3:
            c = colors1[2]
        elif nx.number_of_nodes(g) == 4:
            c = colors1[3]
        elif (nx.number_of_nodes(g) > 4) and nx.number_of_nodes(g) < 8:
            c = colors1[4]
        elif (nx.number_of_nodes(g) > 7) and nx.number_of_nodes(g) < 400:
            c = colors1[5]
        elif (nx.number_of_nodes(g) > 400):
            c = colors1[6]
        nx.draw(g, pos, node_size=28, node_color=c, vmin=0.0,
                vmax=1.0, with_labels=False, ax=ax1, alpha=0.7)
#    ax1.set_title('A.', fontsize=24, loc='left', y=0.98, x=0.05)
    print('There are a total of ' + str(isolate_counter) + ' isolates in the full network')
    plt.savefig(os.path.join(fig_path, 'networks.pdf'),
                bbox_inches='tight')
    plt.savefig(os.path.join(fig_path, 'networks.png'),
                bbox_inches='tight', dpi=600)
    plt.savefig(os.path.join(fig_path, 'networks.svg'),
                bbox_inches='tight')


def make_mf_topics(auth_df, main_df, d_path):
    temp_auth = auth_df.drop_duplicates(subset=['doi', 'authorid'])
    topic_df_long = pd.DataFrame(columns=['DOI', 'Topic', ])
    counter = 0  # this is bad!
    for index, row in main_df.iterrows():
        if type(row['Topic']) is str:
            for topic in row['Topic'].split(';'):
                topic = topic.strip()
                if len(topic) > 0:
                    topic_df_long.loc[counter, 'Topic'] = topic
                    topic_df_long.loc[counter, 'DOI'] = row['DOI']
                    counter += 1
        else:
            print(row['DOI'])

    topic_df_long = topic_df_long[topic_df_long['Topic'] != 'NaN']
    topic_df_long = topic_df_long[topic_df_long['Topic'] != 'nan']
    topic_df_long = topic_df_long[topic_df_long['Topic'] != np.nan]
    topic_df_long = topic_df_long[topic_df_long['Topic'].notnull()]

    authors = pd.merge(temp_auth, topic_df_long[['DOI', 'Topic']], how='left',
                       left_on='doi', right_on='DOI')
    authors['Simple_Topic'] = authors['Topic'].str.extract('(\d+)',
                                                           expand=False)
    authors = authors[authors['Simple_Topic'].notnull()]

    auth_out = pd.DataFrame(index=['1a', '1b', '2a', '2b', '2c',
                                   '3a', '3b', '3c', '4a', '4b', '4c', '5a',
                                   '6a', '6b', '6c', '7a'],
                            columns=['Topic_M', 'Subtopics_M',
                                     'Topic_F', 'Subtopics_F',
                                     'Topic_Ratio', 'Subtopics_Ratio'])
    for gender in ['male', 'female']:
        temp_df = authors[authors['clean_gender'] == gender]
        for topic in authors['Simple_Topic']:
            topic_df = temp_df[temp_df['Simple_Topic'] == topic]
            if gender == 'male':
                auth_out.loc[str(topic) + 'a', 'Topic_M'] = len(topic_df)
            elif gender == 'female':
                auth_out.loc[str(topic) + 'a', 'Topic_F'] = len(topic_df)
        for topic in authors['Topic']:
            topic_df = temp_df[temp_df['Topic'] == topic]
            if gender == 'male':
                auth_out.loc[str(topic), 'Subtopics_M'] = len(topic_df)
            elif gender == 'female':
                auth_out.loc[str(topic), 'Subtopics_F'] = len(topic_df)

    auth_out['Topic_Ratio'] = auth_out['Topic_M'] / auth_out['Topic_F']
    auth_out['Topic_Ratio'] = auth_out['Topic_Ratio'].astype(float).round(2)
    auth_out['Subtopics_Ratio'] = (auth_out['Subtopics_M'] / auth_out['Subtopics_F'])
    auth_out['Subtopics_Ratio'] = auth_out['Subtopics_Ratio'].astype(float).round(2)
    auth_out = auth_out.astype(str).replace('nan', '')
    print(auth_out)
    auth_out.to_csv(os.path.join(d_path, '..',
                                 'article', 'tables',
                                 'author_topic_gender.csv'))


def make_men_women_child(topic_df_long, constrained=False):
    mwc_df = topic_df_long[['abstract', 'Topic']].copy()
    mwc_df.loc[:, 'men'] = 0
    mwc_df.loc[:, 'women'] = 0
    mwc_df.loc[:, 'children'] = 0
    if constrained == False:
        for index, row in mwc_df.iterrows():
            if re.search(r'[Cc]hildren｜[Cc]hild|[Ii]nfant|[kK]ids?｜[bB]ab(y|ies)', row['abstract']):
                if re.search(r'[Aa]dults?', row['abstract']) == None:
                    mwc_df.loc[index, 'children'] = 1
            if re.search(r' [Mm]ales?| [Mm][ea]n| [bB]oys?', row['abstract']):
                if re.search(r'[Ww]omen|[Ww]oman|[Ff]emales?|[Gg]irls?', row['abstract']) == None:
                    mwc_df.loc[index, 'men'] = 1
            if re.search(r'[Ww]om[ae]n|[Ff]emale|[Gg]irl', row['abstract']):
                if re.search(r' [Mm]ales?| [Mm][ea]n| [bB]oys?', row['abstract']) == None:
                    mwc_df.loc[index, 'women'] = 1
    else:
        for index, row in mwc_df.iterrows():
            if re.search(r'[Cc]hildren｜[Cc]hild|[Ii]nfant|[kK]ids?｜[bB]ab(y|ies)', row['abstract']):
                mwc_df.loc[index, 'children'] = 1
            if re.search(r' [Mm]ales?| [Mm][ea]n| [bB]oys?', row['abstract']):
                mwc_df.loc[index, 'men'] = 1
            if re.search(r'[Ww]om[ae]n|[Ff]emale|[Gg]irl', row['abstract']):
                mwc_df.loc[index, 'women'] = 1
    return mwc_df


def make_mwc(main_df, filename):
    topic_df_long = pd.DataFrame(columns=['DOI', 'Date', 'prismcoverdate',
                                          'citedbycount', 'Topic', 'abstract'])
    counter = 0  # this is bad!
    for index, row in main_df.iterrows():
        if type(row['Topic']) is str:
            for topic in row['Topic'].split(';'):
                topic = topic.strip()
                if len(topic) > 0:
                    topic_df_long.loc[counter, 'Topic'] = topic
                    topic_df_long.loc[counter, 'abstract'] = row['abstract']
                    counter += 1
        else:
            print(row['DOI'])
    topic_df_long['Topic'] = topic_df_long['Topic'].str.replace('5a', '5')
    topic_df_long['Topic'] = topic_df_long['Topic'].str.replace('7a', '7')
    print(topic_df_long)

    topic_df_long = topic_df_long[topic_df_long['Topic'] != 'nan']
    topic_df_long = topic_df_long[topic_df_long['Topic'] != np.nan]
    topic_df_long = topic_df_long[topic_df_long['Topic'].notnull()]
    mwc_df = make_men_women_child(topic_df_long)
    men_df = mwc_df[mwc_df['men'] == 1].groupby(['Topic'])['Topic'].count()
    men_df = pd.DataFrame(men_df)
    men_df = men_df.rename(columns={'Topic': 'Men_Count'})
    men_df = men_df.reset_index()
    women_df = mwc_df[mwc_df['women'] == 1].groupby(['Topic'])['Topic'].count()
    women_df = pd.DataFrame(women_df)
    women_df = women_df.rename(columns={'Topic': 'Women_Count'})
    women_df = women_df.reset_index()
    children_df = mwc_df[mwc_df['children'] == 1].groupby(['Topic'])['Topic'].count()
    children_df = pd.DataFrame(children_df)
    children_df = children_df.rename(columns={'Topic': 'Children_Count'})
    children_df = children_df.reset_index()


    mwc_df_unconstrained = make_men_women_child(topic_df_long, constrained=True)
    men_df_unconstrained = mwc_df_unconstrained[mwc_df_unconstrained['men'] == 1].groupby(['Topic'])['Topic'].count()
    men_df_unconstrained = pd.DataFrame(men_df_unconstrained)
    men_df_unconstrained = men_df_unconstrained.rename(columns={'Topic': 'Men_Count'})
    men_df_unconstrained = men_df_unconstrained.reset_index()
    women_df_unconstrained = mwc_df_unconstrained[mwc_df_unconstrained['women'] == 1].groupby(['Topic'])['Topic'].count()
    women_df_unconstrained = pd.DataFrame(women_df_unconstrained)
    women_df_unconstrained = women_df_unconstrained.rename(columns={'Topic': 'Women_Count'})
    women_df_unconstrained = women_df_unconstrained.reset_index()
    children_df_unconstrained = mwc_df_unconstrained[mwc_df_unconstrained['children'] == 1].groupby(['Topic'])['Topic'].count()
    children_df_unconstrained = pd.DataFrame(children_df_unconstrained)
    children_df_unconstrained = children_df_unconstrained.rename(columns={'Topic': 'Children_Count'})
    children_df_unconstrained = children_df_unconstrained.reset_index()

    # this doesnt come into the unconstrained df
    all_df = mwc_df.groupby(['Topic'])['Topic'].count()
    all_df = pd.DataFrame(all_df)
    all_df = all_df.rename(columns={'Topic': 'All_Count'})
    all_df = all_df.reset_index()


    type_df_unconstrained = pd.merge(all_df, women_df_unconstrained, how='left', left_on='Topic', right_on='Topic')
    type_df_unconstrained = pd.merge(type_df_unconstrained, children_df_unconstrained, how='left', left_on='Topic', right_on='Topic')
    type_df_unconstrained = pd.merge(type_df_unconstrained, men_df_unconstrained, how='left', left_on='Topic', right_on='Topic')
    type_df_unconstrained = type_df_unconstrained.set_index('Topic')
    type_df_unconstrained = type_df_unconstrained[type_df_unconstrained.index != 'nan']
    type_df_unconstrained = type_df_unconstrained[type_df_unconstrained.index != '4s']
    type_df_unconstrained = type_df_unconstrained[type_df_unconstrained.index != '6']
    type_df_unconstrained = type_df_unconstrained.reindex(index=type_df_unconstrained.index[::-1])
    type_df_unconstrained['Men_Count'] = type_df_unconstrained['Men_Count'].fillna(0)
    type_df_unconstrained['Women_Count'] = type_df_unconstrained['Women_Count'].fillna(0)
    type_df_unconstrained['Children_Count'] = type_df_unconstrained['Children_Count'].fillna(0)


    type_df = pd.merge(all_df, women_df, how='left', left_on='Topic', right_on='Topic')
    type_df = pd.merge(type_df, children_df, how='left', left_on='Topic', right_on='Topic')
    type_df = pd.merge(type_df, men_df, how='left', left_on='Topic', right_on='Topic')
    type_df = type_df.set_index('Topic')
    type_df = type_df[type_df.index != 'nan']
    type_df = type_df[type_df.index != '4s']
    type_df = type_df[type_df.index != '6']
    type_df = type_df.reindex(index=type_df.index[::-1])
    type_df['Men_Count'] = type_df['Men_Count'].fillna(0)
    type_df['Women_Count'] = type_df['Women_Count'].fillna(0)
    type_df['Children_Count'] = type_df['Children_Count'].fillna(0)

    type_df['Men_Count'] = (type_df['Men_Count'] / type_df['All_Count']) * 100
    type_df['Women_Count'] = (type_df['Women_Count'] / type_df['All_Count']) * 100
    type_df['Children_Count'] = (type_df['Children_Count'] / type_df['All_Count']) * 100

    type_df_unconstrained['Men_Count'] = (type_df_unconstrained['Men_Count'] / type_df['All_Count']) * 100
    type_df_unconstrained['Women_Count'] = (type_df_unconstrained['Women_Count'] / type_df['All_Count']) * 100
    type_df_unconstrained['Children_Count'] = (type_df_unconstrained['Children_Count'] / type_df['All_Count']) * 100

    if filename == 'full':
        fig = plt.figure(figsize=(17, 14), tight_layout=True)
        ax1 = plt.subplot2grid((26, 39), (0, 0), colspan=8, rowspan=12)
        ax2 = plt.subplot2grid((26, 39), (0, 10), colspan=8, rowspan=12)
        ax3 = plt.subplot2grid((26, 39), (0, 20), colspan=8, rowspan=12)
        ax4 = plt.subplot2grid((26, 39), (0, 30), colspan=8, rowspan=26)
        ax5 = plt.subplot2grid((26, 39), (14, 0), colspan=8, rowspan=12)
        ax6 = plt.subplot2grid((26, 39), (14, 10), colspan=8, rowspan=12)
        ax7 = plt.subplot2grid((26, 39), (14, 20), colspan=8, rowspan=12)

        colors = ['#4575b4',  # in reverse
                  '#91bfdb', '#91bfdb', '#91bfdb',
                  '#e0f3f8',
                  '#ffffbf', '#ffffbf', '#ffffbf',
                  '#fee090', '#fee090', '#fee090',
                  '#fc8d59', '#fc8d59', '#fc8d59',
                  '#d73027', '#d73027']

        type_df['Men_Count'].plot(kind='barh', legend=False, alpha=0.7,
                                  edgecolor='k', ax=ax1,
                                  color=colors, width=0.8, linewidth=1.15)
        type_df['Women_Count'].plot(kind='barh', legend=False, alpha=0.7,
                                    edgecolor='k', ax=ax2,
                                    color=colors, width=0.8, linewidth=1.15)
        type_df['Children_Count'].plot(kind='barh', legend=False, alpha=0.7,
                                       edgecolor='k', ax=ax3,
                                       color=colors, width=0.8, linewidth=1.15)
        type_df['All_Count'].plot(kind='barh', legend=False, alpha=0.7,
                                  edgecolor='k', ax=ax4,
                                  color=colors, width=0.8, linewidth=1.15)
        type_df_unconstrained['Men_Count'].plot(kind='barh', legend=False, alpha=0.7,
                                                edgecolor='k', ax=ax5,
                                                color=colors, width=0.8, linewidth=1.15)
        type_df_unconstrained['Women_Count'].plot(kind='barh', legend=False, alpha=0.7,
                                                  edgecolor='k', ax=ax6,
                                                  color=colors, width=0.8, linewidth=1.15)
        type_df_unconstrained['Children_Count'].plot(kind='barh', legend=False, alpha=0.7,
                                                     edgecolor='k', ax=ax7,
                                                     color=colors, width=0.8, linewidth=1.15)
        for axx in [ax1, ax2, ax3, ax5, ax6, ax7]:
            axx.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))

        for axy in [ax1, ax2, ax3, ax4, ax5, ax6, ax7]:
            axy.yaxis.set_tick_params(labelsize=12)

        for axy in [ax2, ax3, ax4, ax6, ax7]:
            axy.set_ylabel('')

        ax1.set_ylabel('Topic Studied', fontsize=13, **csfont)
        ax5.set_ylabel('Topic Studied', fontsize=13, **csfont)
        ax1.set_xlabel('Focus on: Men Only', fontsize=13, **csfont)
        ax2.set_xlabel('Focus on: Women Only', fontsize=13, **csfont)
        ax3.set_xlabel('Focus on: Children Only', fontsize=13, **csfont)
        ax4.set_xlabel('Total Number of Papers', fontsize=13, **csfont)
        ax5.set_xlabel('Focus on: Men', fontsize=13, **csfont)
        ax6.set_xlabel('Focus on: Women', fontsize=13, **csfont)
        ax7.set_xlabel('Focus on: Children', fontsize=13, **csfont)


        ax1.set_title('A.', fontsize=24, loc='left', y=1.02, x=-.1, **csfont)
        ax2.set_title('B.', fontsize=24, loc='left', y=1.02, x=-.1, **csfont)
        ax3.set_title('C.', fontsize=24, loc='left', y=1.02, x=-.1, **csfont)
        ax4.set_title('D.', fontsize=24, loc='left', y=1.005, x=-.1, **csfont)
        ax5.set_title('E.', fontsize=24, loc='left', y=1.02, x=-.1, **csfont)
        ax6.set_title('F.', fontsize=24, loc='left', y=1.02, x=-.1, **csfont)
        ax7.set_title('G.', fontsize=24, loc='left', y=1.02, x=-.1, **csfont)

        rects = ax1.patches
        labels = type_df['Men_Count'].to_list()
        for rect, label in zip(rects, labels):
            if label < 10:
                rounder = 2
            else:
                rounder = 1
            height = rect.get_width()
            ax1.text(height + ax1.get_xlim()[1]/10, (rect.get_y() + rect.get_height() / 2) - .225,
                     str(round(label, rounder)) + '%', ha='center', va='bottom')

        rects = ax2.patches
        labels = type_df['Women_Count'].to_list()
        for rect, label in zip(rects, labels):
            if label < 10:
                rounder = 2
            else:
                rounder = 1
            height = rect.get_width()
            ax2.text(height + ax2.get_xlim()[1]/10, (rect.get_y() + rect.get_height() / 2) - .225,
                     str(round(label, rounder)) + '%', ha='center', va='bottom')

        rects = ax3.patches
        labels = type_df['Children_Count'].to_list()
        for rect, label in zip(rects, labels):
            if label < 10:
                rounder = 2
            else:
                rounder = 1
            height = rect.get_width()
            ax3.text(height + ax3.get_xlim()[1]/10, (rect.get_y() + rect.get_height() / 2) - .225,
                     str(round(label, rounder)) + '%', ha='center', va='bottom')

        rects = ax4.patches
        labels = type_df['All_Count'].to_list()
        for rect, label in zip(rects, labels):
            if label < 10:
                rounder = 3
            else:
                rounder = 2
            height = rect.get_width()
            ax4.text(height + ax4.get_xlim()[1]/15, (rect.get_y() + rect.get_height() / 2) - .15,
                     str(label), ha='center', va='bottom')

        rects = ax5.patches
        labels = type_df_unconstrained['Men_Count'].to_list()
        for rect, label in zip(rects, labels):
            if label < 10:
                rounder = 2
            else:
                rounder = 1
            height = rect.get_width()
            ax5.text(height + ax5.get_xlim()[1]/10, (rect.get_y() + rect.get_height() / 2) - .225,
                     str(round(label, rounder)) + '%', ha='center', va='bottom')

        rects = ax6.patches
        labels = type_df_unconstrained['Women_Count'].to_list()
        for rect, label in zip(rects, labels):
            if label < 10:
                rounder = 2
            else:
                rounder = 1
            height = rect.get_width()
            ax6.text(height + ax6.get_xlim()[1]/10, (rect.get_y() + rect.get_height() / 2) - .225,
                     str(round(label, rounder)) + '%', ha='center', va='bottom')

        rects = ax7.patches
        labels = type_df_unconstrained['Children_Count'].to_list()
        for rect, label in zip(rects, labels):
            if label < 10:
                rounder = 2
            else:
                rounder = 1
            height = rect.get_width()
            ax7.text(height + ax7.get_xlim()[1]/10, (rect.get_y() + rect.get_height() / 2) - .225,
                     str(round(label, rounder)) + '%', ha='center', va='bottom')


        sns.despine()
        plt.tight_layout(True)
        fig_path = os.path.join(os.getcwd(), '..', 'article', 'figures')
        plt.savefig(os.path.join(fig_path, 'MWC_Topics' + filename + '.pdf'),
                    bbox_inches='tight')
        plt.savefig(os.path.join(fig_path, 'MWC_Topics' + filename + '.png'),
                    bbox_inches='tight', dpi=600)
        plt.savefig(os.path.join(fig_path, 'MWC_Topics' + filename + '.svg'),
                    bbox_inches='tight')
    else:
        print(type_df)
        print(type_df_unconstrained)


def headline_topics(main_df):
    topic_df = main_df[main_df['Topic'] != '']
    topic_df = topic_df[topic_df['Topic'].notnull()]
    topic_df = topic_df[topic_df['Topic'] != 'nan']
    topic_df_long = pd.DataFrame(columns=['DOI', 'Date', 'prismcoverdate',
                                          'citedbycount', 'Topic', 'abstract'])
    counter = 0
    for index, row in topic_df.iterrows():
        if type(row['Topic']) is str:
            for topic in row['Topic'].split(';'):
                topic = topic.strip()
                if len(topic) > 0:
                    topic_df_long.loc[counter, 'Topic'] = topic
                    topic_df_long.loc[counter, 'Date'] = row['Date']
                    topic_df_long.loc[counter, 'DOI'] = row['DOI']
                    topic_df_long.loc[counter, 'prismcoverdate'] = row['prismcoverdate']
                    topic_df_long.loc[counter, 'abstract'] = row['abstract']
                    counter += 1
        else:
            print(row['DOI'])
    topic_df_long = topic_df_long[topic_df_long['Topic'] != np.nan]
    topic_df_long['Simple_Topic'] = topic_df_long['Topic'].str.extract('(\d+)',
                                                                       expand=False)
    pattern = '(\d{4})'
    topic_df_long['Year'] = topic_df_long['Date'].str.extract(pattern)
    colors1 = ['#e31a1c', '#a6cee3', '#6a3d9a',
               '#33a02c', '#ff7f00', '#1f78b4',
               '#b15928']
    fig = plt.figure(figsize=(13, 8), tight_layout=True)
    ax7 = plt.subplot2grid((7, 32), (6, 0), rowspan=1, colspan=16)
    ax1 = plt.subplot2grid((7, 32), (0, 0), rowspan=1, colspan=16)
    ax2 = plt.subplot2grid((7, 32), (1, 0), rowspan=1, colspan=16)
    ax3 = plt.subplot2grid((7, 32), (2, 0), rowspan=1, colspan=16)
    ax4 = plt.subplot2grid((7, 32), (3, 0), rowspan=1, colspan=16)
    ax5 = plt.subplot2grid((7, 32), (4, 0), rowspan=1, colspan=16)
    ax6 = plt.subplot2grid((7, 32), (5, 0), rowspan=1, colspan=16)
    ax7 = plt.subplot2grid((7, 32), (6, 0), rowspan=1, colspan=16)
    ax8 = plt.subplot2grid((7, 32), (0, 16), rowspan=7, colspan=16)

    topic_df_long['prismcoverdate'] = topic_df_long['prismcoverdate'].astype('datetime64[ns]')
    alpha_val = 0.75
    linewidth_val = 0.00
    sns.swarmplot(x='prismcoverdate', y='Simple_Topic',
                  data=topic_df_long[topic_df_long['Simple_Topic'] == '1'],
                  s=2.5, orient='h', ax=ax1,
                  color=colors1[0], linewidth=linewidth_val,
                  alpha=alpha_val)
    sns.swarmplot(x='prismcoverdate', y='Simple_Topic',
                  data=topic_df_long[topic_df_long['Simple_Topic'] == '2'],
                  s=3.5, orient='h', ax=ax2,
                  color=colors1[1], linewidth=linewidth_val,
                  alpha=alpha_val)
    sns.swarmplot(x='prismcoverdate', y='Simple_Topic',
                  data=topic_df_long[topic_df_long['Simple_Topic'] == '3'],
                  s=3.5, orient='h', ax=ax3,
                  color=colors1[2], linewidth=linewidth_val,
                  alpha=alpha_val)
    sns.swarmplot(x='prismcoverdate', y='Simple_Topic',
                  data=topic_df_long[topic_df_long['Simple_Topic'] == '4'],
                  s=2.85, orient='h', ax=ax4,
                  color=colors1[3], linewidth=linewidth_val,
                  alpha=alpha_val)
    sns.swarmplot(x='prismcoverdate', y='Simple_Topic',
                  data=topic_df_long[topic_df_long['Simple_Topic'] == '5'],
                  s=3.5, orient='h', ax=ax5,
                  color=colors1[4], linewidth=linewidth_val,
                  alpha=alpha_val)
    sns.swarmplot(x='prismcoverdate', y='Simple_Topic',
                  data=topic_df_long[topic_df_long['Simple_Topic'] == '6'],
                  s=3.5, orient='h', ax=ax6,
                  color=colors1[5], linewidth=linewidth_val,
                  alpha=alpha_val)
    sns.swarmplot(x='prismcoverdate', y='Simple_Topic',
                  data=topic_df_long[topic_df_long['Simple_Topic'] == '7'],
                  s=3.5, orient='h', ax=ax7,
                  color=colors1[6], linewidth=linewidth_val,
                  alpha=alpha_val)

    for axx in [ax1, ax2, ax3, ax4, ax5, ax6]:
        axx.set_xticklabels([])
        axx.set_xticks([])
        axx.set_xlabel('')
        axx.set_ylabel('')
        sns.despine(ax=axx, top=True, left=False, right=True, bottom=True)
        axx.set_xlim(topic_df_long['prismcoverdate'].min(), topic_df_long['prismcoverdate'].max())
    ax7.set_xlim(topic_df_long['prismcoverdate'].min(), topic_df_long['prismcoverdate'].max())

    for index, row in topic_df.iterrows():
        if ';' in row['Topic']:
            color = '#f1f1f1'
            cites = row['citedbycount']
            date = pd.to_datetime(row['prismcoverdate'])
        elif ('7' in row['Topic']):
            color = colors1[6]
            cites = row['citedbycount']
            date = pd.to_datetime(row['prismcoverdate'])
        elif ('6' in row['Topic']):
            color = colors1[5]
            cites = row['citedbycount']
            date = pd.to_datetime(row['prismcoverdate'])
        elif ('5' in row['Topic']):
            color = colors1[4]
            cites = row['citedbycount']
            date = pd.to_datetime(row['prismcoverdate'])
        elif ('4' in row['Topic']):
            color = colors1[3]
            cites = row['citedbycount']
            date = pd.to_datetime(row['prismcoverdate'])
        elif ('3' in row['Topic']):
            color = colors1[2]
            cites = row['citedbycount']
            date = pd.to_datetime(row['prismcoverdate'])
        elif ('2' in row['Topic']):
            color = colors1[1]
            cites = row['citedbycount']
            date = pd.to_datetime(row['prismcoverdate'])
        elif ('1' in row['Topic']):
            color = colors1[0]
            cites = row['citedbycount']
            date = pd.to_datetime(row['prismcoverdate'])
        else:
            print(row)

        ax8.plot_date(x=date, y=cites, color=color, marker='.',
                      alpha=0.5, markersize=cites / 5,
                      markeredgecolor='k')
    ax8.xaxis.set_major_locator(mdates.MonthLocator(interval=108))
    ax8.set_xlim(pd.Timestamp('1947-01-01'), pd.Timestamp('2020-12-31'))
    ax7.xaxis.set_major_locator(mdates.MonthLocator(interval=108))
    ax7.set_xlim(pd.Timestamp('1947-01-01'), pd.Timestamp('2020-12-31'))
    sns.despine(ax=ax7, top=True, left=False, right=True, bottom=False)
    sns.despine(ax=ax8, top=True, left=True, right=False, bottom=False)
    ax1.set_title('A.', fontsize=24, loc='left', y=1.035, x=0)
    ax8.set_title('B.', fontsize=24, loc='left', y=1.00, x=0);
    ax7.set_xlabel('')
    ax7.set_ylabel('')
    ax8.yaxis.set_label_position("right")
    ax8.set_ylabel('Citation Count', **csfont, fontsize=12)

    legend_elements = [Patch(facecolor=(227 / 255, 26 / 255, 28 / 255, 0.5),
                             lw=0.5, label="Fertility", edgecolor=(0, 0, 0, 1)),
                       Patch(facecolor=(166 / 255, 206 / 255, 227 / 255, 0.5),
                             lw=0.5, label="Mortality", edgecolor=(0, 0, 0, 1)),
                       Patch(facecolor=(106 / 255, 61 / 255, 154 / 255, 0.5),
                             lw=0.5, label="Migration", edgecolor=(0, 0, 0, 1)),
                       Patch(facecolor=(51 / 255, 160 / 255, 44 / 255, 0.5),
                             lw=0.5, label="Macro", edgecolor=(0, 0, 0, 1)),
                       Patch(facecolor=(255 / 255, 127 / 255, 0/255, 0.5),
                             lw=0.5, label="Methods", edgecolor=(0, 0, 0, 1)),
                       Patch(facecolor=(31 / 255, 120 / 255, 180 / 255, 0.5),
                             lw=0.5, label="Family", edgecolor=(0, 0, 0, 1)),
                       Patch(facecolor=(177 / 255, 89 / 255, 40 / 255, 0.5),
                             lw=0.5, label="Other", edgecolor=(0, 0, 0, 1)),
                       Patch(facecolor=(241 / 255, 241 / 255, 241 / 255, 0.5),
                             lw=0.5, label="Multiple", edgecolor=(0, 0, 0, 1))]
    ax8.legend(handles=legend_elements, loc='upper left', frameon=True, edgecolor='k',
               fontsize=11, framealpha=1)
    ax8.annotate("Bumpass and Lu (2000)\nCitation Count: 728\nTopic: 6a",
                 xy=(11000, 728), xycoords='data', **csfont,
                 xytext=(10750, 580), fontsize=9, textcoords='data',
                 bbox=dict(boxstyle="round, pad=1", fc="w", linewidth=0.5, edgecolor=(0,0,0,1)),
                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3, rad=-0.5", linewidth=0.5, edgecolor='k'))
    ax8.annotate("Preston (1975)\nCitation Count: 704\nTopics: 2a, 4a, 4c",
                 xy=(2000, 718), xycoords='data', **csfont,
                 xytext=(2750, 570), fontsize=9, textcoords='data',
                 bbox=dict(boxstyle="round, pad=1", fc="w", linewidth=0.5, edgecolor=(0,0,0,1)),
                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3, rad=-0.5", linewidth=0.5, color='k'))
    ax8.annotate("Cleland and Wilson (1987)\nCitation Count: 433\nTopic: 1b",
                 xy=(6200, 442), xycoords='data', **csfont,
                 xytext=(9050, 350), fontsize=9, textcoords='data',
                 bbox=dict(boxstyle="round, pad=1", fc="w", linewidth=0.5, edgecolor=(0,0,0,1)),
                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3, rad=-0.5", linewidth=0.5, color='k'))
    ax8.annotate("Rogers (1979)\nCitation Count: 310\nTopics: 2a, 4b",
                 xy=(3300, 300), xycoords='data', **csfont,
                 xytext=(-3200, 450), fontsize=9, textcoords='data',
                 bbox=dict(boxstyle="round, pad=1", fc="w", linewidth=0.5, edgecolor=(0,0,0,1)),
                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3, rad=-0.5", linewidth=0.5, color='k'))
    ax8.annotate("McKeown and Record (1962)\nCitation Count: 265\nTopics: 4c, 2a",
                 xy=(-2800, 260), xycoords='data', **csfont,
                 xytext=(-6500, 342), fontsize=9, textcoords='data',
                 bbox=dict(boxstyle="round, pad=1", fc="w", linewidth=0.5, edgecolor=(0,0,0,1)),
                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3, rad=-0.5", linewidth=0.5, color='k'))

    ax1.set_ylabel('Fertility', rotation=90, **csfont)
    ax1.set_yticklabels([''])
    ax1.margins(y=-0.49)
    ax2.set_ylabel('Mortality', rotation=90, **csfont)
    ax2.set_yticklabels([''])
    ax3.set_ylabel('Migration', rotation=90, **csfont)
    ax3.set_yticklabels([''])
    ax4.set_ylabel('Macro', rotation=90, **csfont)
    ax4.set_yticklabels([''])
    ax5.set_ylabel('Methods', rotation=90, **csfont)
    ax5.set_yticklabels([''])
    ax6.set_ylabel('Family', rotation=90, **csfont)
    ax6.set_yticklabels([''])
    ax7.set_ylabel('Other', rotation=90, **csfont)
    ax7.set_yticklabels([''])
    ax8.set_ylim(-25, 800)
    fig_path = os.path.join(os.getcwd(), '..', 'article', 'figures')
    plt.subplots_adjust(wspace=.10)
    plt.savefig(os.path.join(fig_path, 'topics_over_time.pdf'),
                bbox_inches='tight')
    plt.savefig(os.path.join(fig_path, 'topics_over_time.png'),
                bbox_inches='tight', dpi=600)
    plt.savefig(os.path.join(fig_path, 'topics_over_time.svg'),
                bbox_inches='tight')


def gender_over_time_oneplot(auth_df):
    csfont = {'fontname': 'Helvetica'}
    auth_df_g = auth_df.drop_duplicates(subset=['doi', 'authorid']).copy()
    auth_df_g.loc[:, 'prismcoverdate'] = pd.to_datetime(auth_df_g['prismcoverdate'], format='%Y-%m-%d')
    auth_df_g.loc[:, 'Year'] = auth_df_g['prismcoverdate'].dt.year
    gender_time_df = pd.DataFrame(index=auth_df_g['Year'].unique(),
                                  columns=['pc_clean_fem_1', 'pc_clean_unknown_1',
                                           'pc_clean_fem_3', 'pc_clean_unknown_3',
                                           'pc_clean_fem_5', 'pc_clean_unknown_5',
                                           'pc_clean_fem_10', 'pc_clean_unknown_10'])
    for year in range(gender_time_df.index.min(), gender_time_df.index.max()):
        temp = auth_df_g[auth_df_g['Year'] == year]
        try:
            gender_time_df.loc[year, 'pc_clean_fem_1'] = len(temp[temp['clean_gender'] == 'female']) / \
                                                         len(temp[(temp['clean_gender'] == 'male') |
                                                                  (temp['clean_gender'] == 'female')])
            gender_time_df.loc[year, 'pc_clean_unknown_1'] = len(temp[temp['clean_gender'] == 'unknown']) / \
                                                             len(temp)

        except:
            pass

    for year in range(gender_time_df.index.min() + 2, gender_time_df.index.max()):
        temp = auth_df_g[(auth_df_g['Year'] > year - 3) & (auth_df_g['Year'] <= year)]
        try:
            gender_time_df.loc[year, 'pc_clean_fem_3'] = len(temp[temp['clean_gender'] == 'female']) / \
                                                         len(temp[(temp['clean_gender'] == 'male') |
                                                                  (temp['clean_gender'] == 'female')])
            gender_time_df.loc[year, 'pc_clean_unknown_3'] = len(temp[temp['clean_gender'] == 'unknown']) / len(temp)
        except:
            pass

    for year in range(gender_time_df.index.min() + 4, gender_time_df.index.max()):
        temp = auth_df_g[(auth_df_g['Year'] > year - 5) & (auth_df_g['Year'] <= year)]
        try:
            gender_time_df.loc[year, 'pc_clean_fem_5'] = len(temp[temp['clean_gender'] == 'female']) / \
                                                         len(temp[(temp['clean_gender'] == 'male') |
                                                                  (temp['clean_gender'] == 'female')])
            gender_time_df.loc[year, 'pc__clean_unknown_5'] = len(temp[temp['clean_gender'] == 'unknown']) / len(temp)
        except:
            pass
    for year in range(gender_time_df.index.min() + 9, gender_time_df.index.max()):
        temp = auth_df_g[(auth_df_g['Year'] > year - 10) & (auth_df_g['Year'] <= year)]
        try:
            gender_time_df.loc[year, 'pc_clean_fem_10'] = len(temp[temp['clean_gender'] == 'female']) / \
                                                          len(temp[(temp['clean_gender'] == 'male') |
                                                                   (temp['clean_gender'] == 'female')])
            gender_time_df.loc[year, 'pc_clean_unknown_10'] = len(temp[temp['clean_gender'] == 'unknown']) / len(temp)
        except:
            pass
    gender_time_df.loc[:, 'pc_clean_fem_10_upper'] = gender_time_df['pc_clean_fem_10'] + gender_time_df[
        'pc_clean_unknown_10'] / 2
    gender_time_df.loc[:, 'pc_clean_fem_10_lower'] = gender_time_df['pc_clean_fem_10'] - gender_time_df[
        'pc_clean_unknown_10'] / 2

    fig, (ax1) = plt.subplots(1, 1, figsize=(12, 6))
    color = ['#377eb8', '#ffb94e']
    ax1.plot(gender_time_df.index, gender_time_df['pc_clean_fem_10_lower'] * 100, color=color[1],
             linewidth=0.4, linestyle='-', dashes=(12, 6))
    ax1.plot(gender_time_df.index, gender_time_df['pc_clean_fem_10_upper'] * 100, color=color[1],
             linewidth=0.4, linestyle='-', dashes=(12, 6))
    ax1.plot(gender_time_df.index, gender_time_df['pc_clean_fem_10'] * 100, color=color[0],
             linewidth=1.6)
    print(gender_time_df)

    temp = gender_time_df[gender_time_df['pc_clean_fem_10'].notnull()]
    temp = temp[['pc_clean_fem_10']]
    temp = temp.astype(float)
    ax1.set_ylim(7.5, 55)
    ax1.spines['left'].set_bounds(10, 55)
    ax1.set_xlim(1951, 2020)
    ax1.spines['bottom'].set_bounds(1953, 2020)

    ax1.set_ylabel('Female Authorship: Ten Year Rolling Interval', **csfont, fontsize=13)
    ax1.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    ax1.fill_between(gender_time_df.index.to_list(),
                     gender_time_df['pc_clean_fem_10_lower'].astype(float) * 100,
                     gender_time_df['pc_clean_fem_10_upper'].astype(float) * 100,
                     alpha=0.2, color=color[1])
    legend_elements = [Line2D([0], [0], color=color[0], lw=1,
                              label='Gender Inference: F/(M+F)', alpha=1),
                       Patch(facecolor=(255 / 255, 185 / 255, 78 / 255, 0.3),
                             lw=0.5, label="Confidence Intervals", edgecolor=(0, 0, 0, 1))]
    ax1.legend(handles=legend_elements, loc='upper left', frameon=False,
               fontsize=13)

    sns.despine()
    ax1.xaxis.grid(linestyle='--', alpha=0.2)
    ax1.yaxis.grid(linestyle='--', alpha=0.2)
#    ax1.set_title('A.', fontsize=24, loc='left', y=1.035, x=0, **csfont)

    early_annot_clean = len(auth_df_g[(auth_df_g['Year'] <= 1984) &
                                      (auth_df_g['clean_gender'] == 'female')]) / \
                        len(auth_df_g[(auth_df_g['Year'] <= 1984) &
                                      ((auth_df_g['clean_gender'] == 'female') |
                                       (auth_df_g['clean_gender'] == 'male'))])
    late_annot_clean = len(auth_df_g[(auth_df_g['Year'] > 1984) &
                                     (auth_df_g['clean_gender'] == 'female')]) / \
                       len(auth_df_g[(auth_df_g['Year'] > 1984) &
                                     ((auth_df_g['clean_gender'] == 'female') |
                                      (auth_df_g['clean_gender'] == 'male'))])
    ax1.axhline(y=early_annot_clean * 100, xmin = .06, xmax=0.48, color='k',
                linestyle='--', alpha=0.75, linewidth=0.75)
    ax1.axhline(y=late_annot_clean * 100, xmin=0.48, xmax=1, color='k',
                linestyle='--', alpha=0.75, linewidth=0.75)
    ax1.annotate("1947-1984 backward\nlooking average: " + str(round(early_annot_clean, 3) * 100) + '%',
                 xy=(1984, early_annot_clean * 100), xycoords='data', **csfont,
                 xytext=(1995, (early_annot_clean * 100) - 10), fontsize=12, textcoords='data',
                 arrowprops=dict(arrowstyle="->",
                                 connectionstyle="arc3, rad=-0.5", linewidth=0.75, color='k'))
    ax1.annotate("1984-2020 backward\nlooking average: " + str(round(late_annot_clean, 3) * 100) + '%',
                 xy=(1984, late_annot_clean * 100), xycoords='data',
                 xytext=(1957, (late_annot_clean * 100) - 5.5), fontsize=12, textcoords='data', **csfont,
                 arrowprops=dict(arrowstyle="->",
                                 connectionstyle="arc3, rad=-0.5", linewidth=0.75, color='k'))
    ax1.tick_params(axis='both', which='major', labelsize=12)
    ax1.tick_params(axis='both', which='minor', labelsize=10)
    fig_path = os.path.join(os.getcwd(), '..', 'article', 'figures')
    plt.savefig(os.path.join(fig_path, 'gender_over_time_oneplot.pdf'),
                bbox_inches='tight')
    plt.savefig(os.path.join(fig_path, 'gender_over_time_oneplot.png'),
                bbox_inches='tight', dpi=600)
    plt.savefig(os.path.join(fig_path, 'gender_over_time_oneplot.svg'),
                bbox_inches='tight')
    pc_fem = len(auth_df[auth_df['clean_gender'] == 'female'])
    tot_gender = len(auth_df[(auth_df['clean_gender'] == 'female') |
                             (auth_df['clean_gender'] == 'male')])
    print('Percent of female authorships (full sample): ' + \
          str(round(pc_fem / tot_gender * 100, 2)) + '%')
    gender_time_df.to_csv(os.path.join(os.getcwd(), '..', 'data', 'support', 'gender_over_time_oneplot.csv'))


def gender_over_time(auth_df):
    auth_df_g = auth_df.drop_duplicates(subset=['doi', 'authorid']).copy()
    auth_df_g.loc[:, 'prismcoverdate'] = pd.to_datetime(auth_df_g['prismcoverdate'], format='%Y-%m-%d')
    auth_df_g.loc[:, 'Year'] = auth_df_g['prismcoverdate'].dt.year
    gender_time_df = pd.DataFrame(index=auth_df_g['Year'].unique(),
                                  columns=['pc_guess_fem_1', 'pc_guess_unknown_1',
                                           'pc_detect_fem_1', 'pc_detect_unknown_1'
                                                              'pc_guess_fem_3', 'pc_guess_unknown_3',
                                           'pc_detect_fem_3', 'pc_detect_unknown_3',
                                           'pc_guess_fem_5', 'pc_guess_unknown_5',
                                           'pc_detect_fem_5', 'pc_detect_unknown_5',
                                           'pc_guess_fem_10', 'pc_guess_unknown_10',
                                           'pc_detect_fem_10', 'pc_detect_unknown_10'])
    for year in range(gender_time_df.index.min(), gender_time_df.index.max()):
        temp = auth_df_g[auth_df_g['Year'] == year]
        temp = temp[temp['gender_guesser'] != 'andy']
        try:
            gender_time_df.loc[year, 'pc_guess_fem_1'] = len(temp[temp['gender_guesser'] == 'female']) / \
                                                         len(temp[(temp['gender_guesser'] == 'male') |
                                                                  (temp['gender_guesser'] == 'female')])
            gender_time_df.loc[year, 'pc_guess_unknown_1'] = len(temp[temp['gender_guesser'] == 'unknown']) / \
                                                             len(temp)
            gender_time_df.loc[year, 'pc_detect_fem_1'] = len(temp[temp['gender_detector'] == 'female']) / \
                                                          len(temp[(temp['gender_detector'] == 'male') |
                                                                   (temp['gender_detector'] == 'female')])
            gender_time_df.loc[year, 'pc_detect_unknown_1'] = len(temp[(temp['gender_detector'] == 'unknown')]) / len(
                temp)
        except:
            pass

    for year in range(gender_time_df.index.min() + 2, gender_time_df.index.max()):
        temp = auth_df_g[(auth_df_g['Year'] > year - 3) & (auth_df_g['Year'] <= year)]
        temp = temp[temp['gender_guesser'] != 'andy']
        try:
            gender_time_df.loc[year, 'pc_guess_fem_3'] = len(temp[temp['gender_guesser'] == 'female']) / \
                                                         len(temp[(temp['gender_guesser'] == 'male') |
                                                                  (temp['gender_guesser'] == 'female')])
            gender_time_df.loc[year, 'pc_guess_unknown_3'] = len(temp[temp['gender_guesser'] == 'unknown']) / len(temp)
            gender_time_df.loc[year, 'pc_detect_fem_3'] = len(temp[temp['gender_detector'] == 'female']) / \
                                                          len(temp[(temp['gender_detector'] == 'male') |
                                                                   (temp['gender_detector'] == 'female')])
            gender_time_df.loc[year, 'pc_detect_unknown_3'] = len(temp[(temp['gender_detector'] == 'unknown')]) / len(
                temp)
        except:
            pass

    for year in range(gender_time_df.index.min() + 4, gender_time_df.index.max()):
        temp = auth_df_g[(auth_df_g['Year'] > year - 5) & (auth_df_g['Year'] <= year)]
        temp = temp[temp['gender_guesser'] != 'andy']
        try:
            gender_time_df.loc[year, 'pc_guess_fem_5'] = len(temp[temp['gender_guesser'] == 'female']) / \
                                                         len(temp[(temp['gender_guesser'] == 'male') |
                                                                  (temp['gender_guesser'] == 'female')])
            gender_time_df.loc[year, 'pc_guess_unknown_5'] = len(temp[temp['gender_guesser'] == 'unknown']) / len(temp)
            gender_time_df.loc[year, 'pc_detect_fem_5'] = len(temp[temp['gender_detector'] == 'female']) / \
                                                          len(temp[(temp['gender_detector'] == 'male') |
                                                                   (temp['gender_detector'] == 'female')])
            gender_time_df.loc[year, 'pc_detect_unknown_5'] = len(temp[(temp['gender_detector'] == 'unknown')]) / len(
                temp)
        except:
            pass
    for year in range(gender_time_df.index.min() + 9, gender_time_df.index.max()):
        temp = auth_df_g[(auth_df_g['Year'] > year - 10) & (auth_df_g['Year'] <= year)]
        temp = temp[temp['gender_guesser'] != 'andy']
        try:
            gender_time_df.loc[year, 'pc_guess_fem_10'] = len(temp[temp['gender_guesser'] == 'female']) / \
                                                          len(temp[(temp['gender_guesser'] == 'male') |
                                                                   (temp['gender_guesser'] == 'female')])
            gender_time_df.loc[year, 'pc_guess_unknown_10'] = len(temp[temp['gender_guesser'] == 'unknown']) / len(temp)
            gender_time_df.loc[year, 'pc_detect_fem_10'] = len(temp[temp['gender_detector'] == 'female']) / \
                                                           len(temp[(temp['gender_detector'] == 'male') |
                                                                    (temp['gender_detector'] == 'female')])
            gender_time_df.loc[year, 'pc_detect_unknown_10'] = len(temp[(temp['gender_detector'] == 'unknown')]) / len(
                temp)
        except:
            pass
    gender_time_df.loc[:, 'pc_detect_fem_10_upper'] = gender_time_df['pc_detect_fem_10'] + gender_time_df[
        'pc_detect_unknown_10'] / 4
    gender_time_df.loc[:, 'pc_detect_fem_10_lower'] = gender_time_df['pc_detect_fem_10'] - gender_time_df[
        'pc_detect_unknown_10'] / 4
    gender_time_df.loc[:, 'pc_guess_fem_10_upper'] = gender_time_df['pc_guess_fem_10'] + gender_time_df[
        'pc_guess_unknown_10'] / 4
    gender_time_df.loc[:, 'pc_guess_fem_10_lower'] = gender_time_df['pc_guess_fem_10'] - gender_time_df[
        'pc_guess_unknown_10'] / 4

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 7))
    color = ['#377eb8', '#ffb94e']
    ax1.plot(gender_time_df.index, gender_time_df['pc_detect_fem_10_lower'] * 100, color=color[1],
             linewidth=0.3, linestyle='--', dashes=(12, 6))
    ax1.plot(gender_time_df.index, gender_time_df['pc_detect_fem_10_upper'] * 100, color=color[1],
             linewidth=0.3, linestyle='--', dashes=(12, 6))
    ax1.plot(gender_time_df.index, gender_time_df['pc_detect_fem_10'] * 100, color=color[0])

    ax2.plot(gender_time_df.index, gender_time_df['pc_guess_fem_10_lower'] * 100, color=color[1],
             linewidth=0.3, linestyle='--', dashes=(12, 6))
    ax2.plot(gender_time_df.index, gender_time_df['pc_guess_fem_10_upper'] * 100, color=color[1],
             linewidth=0.3, linestyle='--', dashes=(12, 6))
    ax2.plot(gender_time_df.index, gender_time_df['pc_guess_fem_10'] * 100, color=color[0])
    temp = gender_time_df[gender_time_df['pc_guess_fem_10'].notnull()]
    temp = temp[['pc_guess_fem_10', 'pc_detect_fem_10']]
    temp = temp.astype(float)
    print(temp.corr())
    print('The max of Gender Detector is: ' +\
          str(gender_time_df['pc_detect_fem_10'].max()))
    print('The min of Gender Detector is: ' +\
          str(gender_time_df['pc_detect_fem_10'].min()))
    print('The max of Gender Guesser is: ' +\
          str(gender_time_df['pc_guess_fem_10'].max()))
    print('The min of Gender Guesser is: ' +\
          str(gender_time_df['pc_guess_fem_10'].min()))
    ax1.set_ylim(7.5, 55)
    ax1.spines['left'].set_bounds(10, 55)
    ax1.set_xlim(1951, 2020)
    ax1.spines['bottom'].set_bounds(1955, 2020)

    ax2.set_ylim(7.5, 55)
    ax2.spines['left'].set_bounds(10, 55)
    ax2.set_xlim(1951, 2020)
    ax2.spines['bottom'].set_bounds(1955, 2020)

    ax1.set_ylabel('Female Authorship: Ten Year Rolling Interval', **csfont, fontsize=13)
    ax2.set_ylabel('Female Authorship: Ten Year Rolling Interval', **csfont, fontsize=13)
    ax1.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    ax2.fill_between(gender_time_df.index.to_list(),
                     gender_time_df['pc_guess_fem_10_lower'].astype(float) * 100,
                     gender_time_df['pc_guess_fem_10_upper'].astype(float) * 100,
                     alpha=0.2, color=color[1])
    ax1.fill_between(gender_time_df.index.to_list(),
                     gender_time_df['pc_detect_fem_10_lower'].astype(float) * 100,
                     gender_time_df['pc_detect_fem_10_upper'].astype(float) * 100,
                     alpha=0.2, color=color[1])
    legend_elements = [Line2D([0], [0], color=color[0], lw=1,
                              label='Detector: F/(M+F)', alpha=1),
                       Patch(facecolor=(255 / 255, 185 / 255, 78 / 255, 0.3),
                             lw=0.5, label="Detector: CI's", edgecolor=(0, 0, 0, 1))]
    ax1.legend(handles=legend_elements, loc='upper left', frameon=False,
               fontsize=12)
    legend_elements = [Line2D([0], [0], color=color[0], lw=1,
                              label='Guesser: F/(M+F)', alpha=1),
                       Patch(facecolor=(255 / 255, 185 / 255, 78 / 255, 0.3),
                             lw=0.5, label="Guesser: CI's", edgecolor=(0, 0, 0, 1))]
    ax2.legend(handles=legend_elements, loc='upper left', frameon=False,
               fontsize=12)
    sns.despine()
    ax1.xaxis.grid(linestyle='--', alpha=0.2)
    ax1.yaxis.grid(linestyle='--', alpha=0.2)
    ax2.xaxis.grid(linestyle='--', alpha=0.2)
    ax2.yaxis.grid(linestyle='--', alpha=0.2)
    ax1.set_title('A.', fontsize=24, loc='left', y=1.035, x=0, **csfont)
    ax2.set_title('B.', fontsize=24, loc='left', y=1.035, x=0, **csfont);

    early_annot_guesser = len(auth_df_g[(auth_df_g['Year'] <= 1984) & (auth_df_g['gender_guesser'] == 'female')]) / \
                          len(auth_df_g[(auth_df_g['Year'] <= 1984) & ((auth_df_g['gender_guesser'] == 'female') |
                                                                       (auth_df_g['gender_guesser'] == 'male'))])
    late_annot_guesser = len(auth_df_g[(auth_df_g['Year'] > 1984) & (auth_df_g['gender_guesser'] == 'female')]) / \
                         len(auth_df_g[(auth_df_g['Year'] > 1984) & ((auth_df_g['gender_guesser'] == 'female') |
                                                                     (auth_df_g['gender_guesser'] == 'male'))])
    ax2.axhline(y=early_annot_guesser * 100, xmax=0.44, color='k', linestyle='--', alpha=0.75, linewidth=0.75)
    ax2.axhline(y=late_annot_guesser * 100, xmin=0.44, xmax=1, color='k', linestyle='--', alpha=0.75, linewidth=0.75)
    ax2.annotate("1947-1984 backward\nlooking average: " + str(round(early_annot_guesser, 3) * 100) + '%',
                 xy=(1984, early_annot_guesser * 100), xycoords='data', **csfont,
                 xytext=(1995, (early_annot_guesser * 100) - 10), fontsize=10, textcoords='data',
                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3, rad=-0.5", linewidth=0.75, color='k'))
    ax2.annotate("1984-2020 backward\nlooking average: " + str(round(late_annot_guesser, 4) * 100) + '%',
                 xy=(1982, late_annot_guesser * 100), xycoords='data', **csfont,
                 xytext=(1957, (late_annot_guesser * 100) - 5.5), fontsize=10, textcoords='data',
                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3, rad=-0.5", linewidth=0.75, color='k'))

    early_annot_detecter = len(auth_df_g[(auth_df_g['Year'] <= 1984) & (auth_df_g['gender_detector'] == 'female')]) / \
                           len(auth_df_g[(auth_df_g['Year'] <= 1984) & ((auth_df_g['gender_detector'] == 'female') |
                                                                        (auth_df_g['gender_detector'] == 'male'))])
    late_annot_detecter = len(auth_df_g[(auth_df_g['Year'] > 1984) & (auth_df_g['gender_detector'] == 'female')]) / \
                          len(auth_df_g[(auth_df_g['Year'] > 1984) & ((auth_df_g['gender_detector'] == 'female') |
                                                                      (auth_df_g['gender_detector'] == 'male'))])
    ax1.axhline(y=early_annot_detecter * 100, xmax=0.44, color='k', linestyle='--', alpha=0.75, linewidth=0.75)
    ax1.axhline(y=late_annot_detecter * 100, xmin=0.44, xmax=1, color='k', linestyle='--', alpha=0.75, linewidth=0.75)
    ax1.annotate("1947-1984 backward\nlooking average: " + str(round(early_annot_detecter, 4) * 100) + '%',
                 xy=(1984, early_annot_detecter * 100), xycoords='data', **csfont,
                 xytext=(1995, (early_annot_detecter * 100) - 10), fontsize=10, textcoords='data',
                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3, rad=-0.5", linewidth=0.75, color='k'))
    ax1.annotate("1984-2020 backward\nlooking average: " + str(round(late_annot_detecter, 4) * 100) + '%',
                 xy=(1982, late_annot_detecter * 100), xycoords='data',
                 xytext=(1957, (late_annot_detecter * 100) - 5.5), fontsize=10, textcoords='data', **csfont,
                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3, rad=-0.5", linewidth=0.75, color='k'))
    plt.tight_layout(pad=3)
    fig_path = os.path.join(os.getcwd(), '..', 'article', 'figures')
    plt.savefig(os.path.join(fig_path, 'gender_over_time.pdf'),
                bbox_inches='tight')
    plt.savefig(os.path.join(fig_path, 'gender_over_time.png'),
                bbox_inches='tight', dpi=600)
    plt.savefig(os.path.join(fig_path, 'gender_over_time.svg'),
                bbox_inches='tight')
    pc_fem = len(auth_df[auth_df['clean_gender'] == 'female'])
    tot_gender = len(auth_df[(auth_df['clean_gender'] == 'female') |
                             (auth_df['clean_gender'] == 'male')])
    print('Percent of female authorships (full sample): ' + \
          str(round(pc_fem / tot_gender * 100, 2)) + '%')
    gender_time_df.to_csv(os.path.join(os.getcwd(), '..', 'data', 'support', 'gender_over_time.csv'))


def summarize_scrape_and_curate(main_df, auth_df, ref_df, G, Gcc, d_path):
    missing_df = main_df[(main_df['Topic'].isnull()) &
                         (main_df['abstract'] != 'nan. ')]
    if len(missing_df) > 0:
        print('Danger! Papers with abstracts and no topics: ' + str(len(missing_df)))
        cols = ['DOI', 'Title', 'abstract', 'Topic',
                'Subnation_popstudied', 'Regions', 'Nation',
                'Population', 'Dataset', 'Time']
        missing_df[cols].to_csv(os.path.join(d_path, 'manual_review', 'unmatched',
                                             'unmatched_abstracts.csv'))
    print('Total number of papers in our main database: ' + str(len(main_df)))
    print('Total number of papers in our author database: ' + str(len(auth_df['doi'].unique())))
    print('Papers with no abstract: ' + str(len(main_df[main_df['abstract'].str.len()<=5])))
    temp = main_df.groupby(['subtypedescription'])['subtypedescription'].count()
    temp = temp.reset_index(name='Count')
    for index, row in temp.iterrows():
        print('In our main dataframe there are ' + str(row['Count']) + ' ' +
              str(row['subtypedescription']) + 's.')
    print('Total number of authorships: ' + str(len(auth_df.drop_duplicates(subset=['doi', 'authorid']))))
    print('Total number of unique authors: ' + str(len(auth_df['authorid'].unique())))
    print('Average number of references per paper: ' + str(len(ref_df) / len(main_df)))
    print('Average number of authors per paper: ' + str(round(len(auth_df.drop_duplicates(subset=['doi', 'authorid'])) / len(auth_df['doi'].unique()), 3)))
    grouped_ref_count = ref_df.groupby(['doi'])['doi'].count()
    grouped_ref_count = grouped_ref_count.reset_index(name='count')
    max_df = grouped_ref_count[grouped_ref_count['count'] == grouped_ref_count['count'].max()]
    max_doi = max_df.loc[max_df.index[0], 'doi']
    print('The most number of refrences in one paper: ' +
          str(grouped_ref_count['count'].max()) +
          ' (DOI: ' + str(max_doi) + ')')
    print('Date of first article: ' + str(main_df['prismcoverdate'].min()))
    print('Date of most recent article: ' + str(main_df['prismcoverdate'].max()))
    pages_df = main_df[(main_df['pagestart'].notnull()) &
                       (main_df['pageend'].notnull())].copy()
    pages_df.loc[:, 'paper_length'] = pages_df['pageend'] - pages_df['pagestart'] + 1
    print('Average paper length (pages): ' + str(pages_df['paper_length'].sum() / len(pages_df)))
    print('Number of OpenAccess articles: ' + str(main_df['openaccess'].sum()))
    print('The average number of citations is: ' + str(main_df['citedbycount'].mean()))
    max_df = main_df[main_df['citedbycount'] == main_df['citedbycount'].max()]
    max_doi = max_df.loc[max_df.index[0], 'DOI']
    print('The maximum number of citations is: ' + str(main_df['citedbycount'].max()) + \
          '(' + str(max_doi) + ')')
    print('Number of papers with no citations: ' + str(len(main_df[main_df['citedbycount'] == 0])))

    collab_df = auth_df[(auth_df['doi'].duplicated(keep=False)) &
                        (auth_df['aff_id'].notnull()) &
                        (auth_df['prismcoverdate'].str.startswith('198'))]
    collab_df_diff = collab_df[['doi', 'aff_id']].drop_duplicates(keep=False)
    print(
        'Percent of inter-affiliation collaborations in 1980s: ' + str(round(len(collab_df_diff) / len(collab_df), 2)))

    collab_df = auth_df[(auth_df['doi'].duplicated(keep=False)) &
                        (auth_df['aff_id'].notnull()) &
                        (auth_df['prismcoverdate'].str.startswith('199'))]
    collab_df_diff = collab_df[['doi', 'aff_id']].drop_duplicates(keep=False)
    print(
        'Percent of inter-affiliation collaborations in 1990s: ' + str(round(len(collab_df_diff) / len(collab_df), 2)))

    collab_df = auth_df[(auth_df['doi'].duplicated(keep=False)) &
                        (auth_df['aff_id'].notnull()) &
                        (auth_df['prismcoverdate'].str.startswith('200'))]
    collab_df_diff = collab_df[['doi', 'aff_id']].drop_duplicates(keep=False)
    print(
        'Percent of inter-affiliation collaborations in 2000s: ' + str(round(len(collab_df_diff) / len(collab_df), 2)))

    collab_df = auth_df[(auth_df['doi'].duplicated(keep=False)) &
                        (auth_df['aff_id'].notnull()) &
                        (auth_df['prismcoverdate'].str.startswith('201'))]
    collab_df_diff = collab_df[['doi', 'aff_id']].drop_duplicates(keep=False)
    print(
        'Percent of inter-affiliation collaborations in 2010s: ' + str(round(len(collab_df_diff) / len(collab_df), 2)))

    collab_df = auth_df[(auth_df['doi'].duplicated(keep=False)) &
                        (auth_df['aff_id'].notnull())]
    collab_df_diff = collab_df[['doi', 'aff_id']].drop_duplicates(keep=False)
    print('Percent of inter-affiliation collaborations over all time: ' + \
          str(round(len(collab_df_diff) / len(collab_df), 2)))


def make_affil_plot(main_df, auth_df, d_path, figure_path):
    titlesize = 15
    markersize = 10
    region_lookup = pd.read_csv(os.path.join(d_path, 'support', 'region_lookup.csv'))
    country_count = make_country_count(auth_df)
    gdf = make_gdf(country_count, os.path.join(d_path, 'shapefiles',
                                               'global.shp'))
    gdf['count'] = gdf['count'].fillna(0)
    gdf = gdf[gdf['CNTRY_NAME'] != 'Antarctica']
    islands_to_remove = ['Solomon Islands', 'Bouvet Island', 'Cayman Islands', 'Pacific Islands (Palau)',
                         'Cook Islands', 'Paracel Islands', 'Pitcairn Islands', 'Cocos (Keeling) Islands',
                         'Northern Mariana Islands', 'Jarvis Island', 'Falkland Islands (Islas Malvinas)',
                         'Faroe Islands', 'Baker Island', 'Glorioso Islands', 'Heard Island & McDonald Islands',
                         'Howland Island', 'Juan De Nova Island', 'Christmas Island', 'Midway Islands',
                         'Norfolk Island', 'Spratly Islands', 'Marshall Islands', 'Turks and Caicos Islands',
                         'British Virgin Islands', 'Virgin Islands', 'Wake Island']
    for island in islands_to_remove:
        gdf = gdf[gdf['CNTRY_NAME'] != island]
    region_count = make_region(country_count, region_lookup, d_path)
    region_count_short = region_count.groupby(['Region'])['count']. \
                             sum().sort_values(ascending=False)[0:5]
    region_count_short['Other'] = region_count.groupby(['Region'])['count']. \
                                      sum().sort_values(ascending=False)[5:].sum()
    region_count_short = region_count_short.sort_values()
    time_df = make_time(main_df, auth_df)
    aff_count = make_aff_count(auth_df)

    fig = plt.figure(figsize=(14, 10), tight_layout=True)
    ax1 = plt.subplot2grid((10, 11), (0, 1), rowspan=4, colspan=3)
    ax2 = plt.subplot2grid((10, 11), (0, 4), rowspan=4, colspan=3)
    ax4 = plt.subplot2grid((10, 11), (0, 7), rowspan=4, colspan=3)
    ax3 = plt.subplot2grid((10, 11), (4, 0), rowspan=6, colspan=11)

    # make pie
    region_count_short.index = region_count_short.index.str.replace('NORTHERN AMERICA', 'N. America')
    region_count_short.index = region_count_short.index.str.replace('WESTERN EUROPE', 'W. Europe')
    region_count_short.index = region_count_short.index.str.replace('SUB-SAHARAN AFRICA', 'S. Sah. Africa')
    region_count_short.index = region_count_short.index.str.strip().str.title()
    region_count_short = region_count_short.loc[['S. Sah. Africa', 'Oceania',
                                                 'Asia (Ex. Middle East)',
                                                 'W. Europe', 'Other', 'N. America']]
    labels = region_count_short.index
    sizes = region_count_short.tolist()
    colors = ['#f4cae4', '#e6f5c9', '#b3e2cd', '#cbd5e8', '#fff2ae', '#fdcdac']
    colors1 = ['#d73027', '#fc8d59', '#fee090', '#ffffbf', '#a6cee3', '#1f78b4', '#4575b4']
    explode = (0.115, 0.115, 0.115, 0.115, 0.115, 0.115)
    wedges, labels, autopct = ax1.pie(sizes, explode=explode, labels=labels, colors=colors1[0:6],
                                      wedgeprops=dict(width=0.225), autopct='%1.1f%%', shadow=False,
                                      startangle=200, textprops={'fontsize': 12})
    wedges = [patch for patch in ax1.patches if isinstance(patch, patches.Wedge)]
    for w in wedges:
        w.set_linewidth(0.90)
        w.set_edgecolor('k')
        w.set_alpha(0.7)
    centre_circle = plt.Circle((0, 0), 0.75, color='black', fc='white', linewidth=.25)
    ax1.axis('equal')

    # make tsplot
    # old
    #    time_df['affiliations'] = time_df['affiliations'] / time_df['affiliations'][2020]
    #    time_df['countries'] = time_df['countries'] / time_df['countries'][2020]
    #    ax2.step(y=time_df['affiliations'].astype(float), label='Affiliations',
    #             x=time_df.index.astype(int), color='#fc8d59', alpha=0.8)
    #    ax2.step(y=time_df['countries'].astype(float), label='Countries',
    #             x=time_df.index.astype(int), color='#1f78b4', alpha=0.8)
    #    legend = ax2.legend(loc='upper left', edgecolor='k', frameon=True, fontsize=11, ncol=1, framealpha=1)
    #    ax2.xaxis.grid(linestyle='--', alpha=0.5)
    #    ax2.yaxis.grid(linestyle='--', alpha=0.5)
    #    legend.get_frame().set_linewidth(0.5)
    #    ax2.set_ylabel('Cumulative fraction of contributions', fontsize=12, **csfont)

    #   RnR version

    region_lookup['Country'] = region_lookup['Country'].str.strip()
    region_lookup['Region'] = region_lookup['Region'].str.strip()
    auth_df['aff_country'] = auth_df['aff_country'].str.replace('Hong Kong', 'China')
    auth_df_wregion = pd.merge(auth_df, region_lookup, how='left', left_on='aff_country',
                               right_on='Country')
    region_count = make_region(country_count, region_lookup, d_path)
    othersum = region_count.groupby(['Region'])['count']. \
                   sum().sort_values(ascending=False)[5:].sum()
    region_count_short = region_count.groupby(['Region'])['count']. \
                             sum().sort_values(ascending=False)[0:5]
    region_count_short.at['Other'] = othersum
    auth_df_wregion.at[:, 'Region_Short'] = np.where(
        auth_df_wregion['Region'].isin(region_count_short.reset_index()['Region']),
        auth_df_wregion['Region'], 'Other')
    main_df['year'] = main_df.Date.str.extract(r'([0-9][0-9][0-9][0-9])',
                                               expand=True)
    auth_df_wdate = pd.merge(auth_df_wregion, main_df[['DOI', 'year']],
                             how='left', left_on='doi', right_on='DOI')
    auth_df_wdate['Region'] = auth_df_wdate['Region'].str.strip()

    time_df = pd.DataFrame(index=range(1979, 2021),
                           columns=auth_df_wdate['Region'].unique())

    df_region_year = pd.DataFrame(index=auth_df_wdate['year'].unique().tolist(),
                                  columns=region_count_short.index.unique().tolist())

    for region in region_count_short.index.unique():
        for year in auth_df_wdate['year'].unique():
            temp = auth_df_wdate[(auth_df_wdate['Region'] == region) &
                                 (auth_df_wdate['year'] == year)]
            df_region_year.at[year, region] = len(temp)
    df_region_year = df_region_year.reset_index()
    df_region_year = df_region_year.sort_values(by='index', ascending=True)
    df_region_year = df_region_year[df_region_year['index'].astype(int) > 1980]

    df_region_year = df_region_year.rename({'NORTHERN AMERICA': 'N. America'}, axis=1)
    df_region_year = df_region_year.rename({'WESTERN EUROPE': 'W. Europe'}, axis=1)
    df_region_year = df_region_year.rename({'OCEANIA': 'Oceania'}, axis=1)
    df_region_year = df_region_year.rename({'SUB-SAHARAN AFRICA': 'S. Sah. Africa'}, axis=1)
    df_region_year = df_region_year.rename({'ASIA (EX. MIDDLE EAST)': 'Asia (Ex. Mid East)'}, axis=1)
    df_region_year = df_region_year.set_index('index')
    df_region_year.index = df_region_year.index.astype(int)
    colors1 = [colors1[5], colors1[3], colors1[2], colors1[0], colors1[1], colors1[4]]
    df_region_year.plot(kind='area', stacked=True,
                        color=colors1,
                        ax=ax2, alpha=0.55, linewidth=1)

    # ax2.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax2.set_xlim(1980, 2020)
    ax2.legend(frameon=False, loc='upper left')
    ax2.set_xlabel('')
    ax2.set_ylabel('Number of Authors', fontsize=12)

    patch1 = patches.Patch(facecolor=colors1[0], edgecolor='k', label='N. America', linewidth=0.75, alpha=0.55)
    patch2 = patches.Patch(facecolor=colors1[1], edgecolor='k', label='W. Europe', linewidth=0.75, alpha=0.55)
    patch3 = patches.Patch(facecolor=colors1[2], edgecolor='k', label='Asia (Ex. Mid East)', linewidth=0.75, alpha=0.55)
    patch4 = patches.Patch(facecolor=colors1[3], edgecolor='k', label='Oceania', linewidth=0.75, alpha=0.55)
    patch5 = patches.Patch(facecolor=colors1[4], edgecolor='k', label='S. Saharan Africa', linewidth=0.75, alpha=0.55)
    patch6 = patches.Patch(facecolor=colors1[5], edgecolor='k', label='Other', linewidth=0.75, alpha=0.55)
    ax2.legend(handles=[patch1, patch2, patch3, patch4, patch5, patch6],
               frameon=False, framealpha=1, edgecolor='k', loc='upper left')

    # make bargraph
    aff_count['aff_orgs'] = aff_count['aff_orgs'].str.replace('University of Pennsylvania', 'U. Penn')
    aff_count['aff_orgs'] = aff_count['aff_orgs'].str.replace('University of California', 'U. Cal')
    aff_count['aff_orgs'] = aff_count['aff_orgs'].str.replace('University of Wisconsin', 'U. Wisconsin')
    aff_count['aff_orgs'] = aff_count['aff_orgs'].str.replace('The Population Council', 'Pop Council')
    aff_series = aff_count[0:10].set_index('aff_orgs')
    aff_series = aff_series.sort_values(by='count', ascending=True)['count']
    y_pos = np.arange(len(aff_series))
    aa = ax4.barh(y_pos, aff_series.tolist(), edgecolor='k', alpha=0.75)
    aa[0].set_color('#ffffbf')
    aa[0].set_edgecolor('k')
    aa[1].set_color('#1f78b4')
    aa[1].set_edgecolor('k')
    aa[2].set_color('#1f78b4')
    aa[2].set_edgecolor('k')
    aa[3].set_color('#1f78b4')
    aa[3].set_edgecolor('k')
    aa[4].set_color('#fc8d59')
    aa[4].set_edgecolor('k')
    aa[5].set_color('#fc8d59')
    aa[5].set_edgecolor('k')
    aa[6].set_color('#fc8d59')
    aa[6].set_edgecolor('k')
    aa[7].set_color('#ffffbf')
    aa[7].set_edgecolor('k')
    aa[8].set_color('#fc8d59')
    aa[8].set_edgecolor('k')
    aa[9].set_color('#fc8d59')
    aa[9].set_edgecolor('k')
#   ax4.yaxis.grid(linestyle='--', alpha=0.5)
#   ax4.xaxis.grid(linestyle='--', alpha=0.5)

    ax4.yaxis.set_major_locator(mtick.FixedLocator(range(0, len(aff_series))))
    ax4.set_yticklabels(aff_series.index)
    ax = ax4.set_ylabel('')  # ?
    ax4.set_yticks(np.arange(len(aff_series)))

    rects = ax4.patches
    for rect, label in zip(rects, aff_series.tolist()):
        x_value = rect.get_width()
        y_value = rect.get_y() + rect.get_height() / 2
        ax4.text(x_value + 10, y_value - .225, str(round((label / aff_count['count'].sum()) * 100, 2)) + '%',
                 ha='center', va='bottom')
    blue_patch = patches.Patch(facecolor='#1f78b4', edgecolor='k', label='US', linewidth=0.75, alpha=0.75)
    red_patch = patches.Patch(facecolor='#fc8d59', edgecolor='k', label='Europe', linewidth=0.75, alpha=0.75)
    yellow_patch = patches.Patch(facecolor='#ffffbf', edgecolor='k', label='International', linewidth=0.75, alpha=0.75)
    ax4.legend(handles=[red_patch, blue_patch, yellow_patch], frameon=True, framealpha=1, edgecolor='k')

    # make choro
    cmap = LinearSegmentedColormap.from_list('mycmap', colors)
    gdf = gdf.to_crs(epsg=4326)
    gdf['count_cat'] = np.nan
    gdf['count_cat'] = np.where(gdf['count'] == 0, '0', gdf['count_cat'])
    gdf['count_cat'] = np.where((gdf['count'] > 0) & (gdf['count'] <= 1), '1', gdf['count_cat'])
    gdf['count_cat'] = np.where((gdf['count'] > 1) & (gdf['count'] <= 4), '2', gdf['count_cat'])
    gdf['count_cat'] = np.where((gdf['count'] > 4) & (gdf['count'] <= 9), '3', gdf['count_cat'])
    gdf['count_cat'] = np.where((gdf['count'] > 9) & (gdf['count'] <= 15), '4', gdf['count_cat'])
    gdf['count_cat'] = np.where((gdf['count'] > 15) & (gdf['count'] <= 30), '5', gdf['count_cat'])
    gdf['count_cat'] = np.where((gdf['count'] > 30) & (gdf['count'] <= 50), '6', gdf['count_cat'])
    gdf['count_cat'] = np.where(gdf['count'] > 50, '7', gdf['count_cat'])
    gdf.plot(ax=ax3, color='white', edgecolor='black', linewidth=0.35);
    gdf.plot(ax=ax3, color='None', edgecolor='black', alpha=0.2);
    bb = gdf.plot(column='count_cat', cmap='Blues', linewidth=0.00, alpha=0.8, ax=ax3, facecolor='k',
                  legend=True, legend_kwds=dict(loc='lower left',
                                                framealpha=1,
                                                bbox_to_anchor=(0.03, 0.07),
                                                edgecolor='w', ncol=1,
                                                frameon=True, facecolor=(255 / 255, 255 / 255, 255 / 255, 1),
                                                fontsize=titlesize - 3,
                                                # zorder=2
                                                ),
                  markersize=markersize)
    bounds = ['No Studies', '1 Study', '2-4 Studies',
              '5-9 Studies', '10-15 Studies',
              '16-30 Studies', '31-50 Studies',
              'Over 50 Studies']
    legend_labels = ax3.get_legend().get_texts()
    leg = ax3.get_legend()
    leg.set_zorder(5000)
    for bound, legend_label in zip(bounds, legend_labels):
        legend_label.set_text(bound)
    ax3.axis('off')
    for legend_handle in ax3.get_legend().legendHandles:
        legend_handle._legmarker.set_markeredgewidth(1)
        legend_handle._legmarker.set_markeredgecolor('k')
    ax1.set_title('A.', fontsize=titlesize + 6, y=1.025, **csfont, loc='left', x=-.215)
    #    ax1.set_title('Continents', fontsize=16,
    #                  y=1.03, **csfont, loc='center')
    ax2.set_title('B.', fontsize=titlesize + 6, y=1.025, **csfont, loc='left')
    #    ax2.set_title('Time', fontsize=16,
    #                  y=1.03, **csfont, loc='center')
    ax4.set_xlabel('Number of contributions', fontsize=12)
    ax4.set_title('C.', fontsize=titlesize + 6, y=1.025, **csfont, loc='left')
    #    ax4.set_title('Institutions (Top 10)', fontsize=16,
    #                  y=1.03, **csfont, loc='center')
    #    ax3.set_title('Countries', **csfont,
    #                  fontsize=16, y=1)
    ax3.set_title('D.', **csfont, fontsize=titlesize + 6, loc='left', y=1, x=0.025)
    sns.despine(ax=ax2)
    sns.despine(ax=ax4)

    plt.savefig(os.path.join(figure_path, 'author_institutions.svg'),
                bbox_inches='tight')
    plt.savefig(os.path.join(figure_path, 'author_institutions.pdf'),
                bbox_inches='tight')
    plt.savefig(os.path.join(figure_path, 'author_institutions.png'),
                bbox_inches='tight', dpi=600)


def make_country_count(auth_df):
    auth_df['aff_country'] = auth_df['aff_country'].str.replace('Hong Kong', 'China')
    country_count = auth_df.groupby(['aff_country'])['aff_country'].count()
    country_count = country_count.reset_index(name='count')
    country_count = pd.DataFrame(country_count)
    return country_count


def make_aff_count(auth_df):
    aff_count = auth_df.groupby(['aff_orgs'])['aff_orgs'].count()
    aff_count = aff_count.reset_index(name='count')
    aff_count = pd.DataFrame(aff_count)
    aff_count = aff_count.sort_values(by='count', ascending=False)
    return aff_count


def make_time(main_df, auth_df):
    main_df['year'] = main_df.Date.str.extract(r'([0-9][0-9][0-9][0-9])',
                                               expand=True)
    auth_df_wdate = pd.merge(auth_df, main_df[['DOI', 'year']],
                             how='left', left_on='doi', right_on='DOI')
    time_df = pd.DataFrame(index=range(1979, 2021),
                           columns=['countries', 'affiliations', 'contribs'])
    set_affils = set()
    set_countries = set()
    for year in range(1979, 2021):
        temp_df = auth_df_wdate[auth_df_wdate['year'] == str(year)]
        temp_df = temp_df[temp_df['aff_orgs'].notnull()]
        time_df.loc[year, 'contribs'] = len(temp_df)
        set_affils.update((temp_df['aff_orgs'].tolist()))
        time_df.loc[year, 'affiliations'] = len(set_affils)

        set_countries.update((temp_df['aff_country'].tolist()))
        time_df.loc[year, 'countries'] = len(set_countries)
    return time_df


def make_region(df, gdf, d_path):
    gdf['Country'] = gdf['Country'].str.strip()
    gdf['Country'] = gdf['Country'].replace("Gambia, The", "Gambia")
    gdf['Country'] = gdf['Country'].replace("Vietnam", "Viet Nam")
    gdf['Country'] = gdf['Country'].replace("Tanzania, United Republic of", "Tanzania")
    gdf['Country'] = gdf['Country'].replace("Russia", "Russian Federation")
    gdf['Country'] = gdf['Country'].replace("Congo, Dem. Rep.", 'Democratic Republic Congo')
    gdf['Country'] = gdf['Country'].replace("Korea, South", 'South Korea')

    bad_names = list(np.setdiff1d(df['aff_country'].tolist(), gdf['Country'].tolist()))
    if len(bad_names) > 0:
        print('Some countries are incorrectly named!')
        print(bad_names)
    gdf = pd.merge(gdf, df, how='left', left_on='Country',
                   right_on='aff_country')
    return gdf


def make_gdf(df, shapefile):
    gdf = gpd.read_file(shapefile)
    gdf['CNTRY_NAME'] = gdf['CNTRY_NAME'].replace("Gambia, The", "Gambia")
    gdf['CNTRY_NAME'] = gdf['CNTRY_NAME'].replace("Vietnam", "Viet Nam")
    gdf['CNTRY_NAME'] = gdf['CNTRY_NAME'].replace("Tanzania, United Republic of", "Tanzania")
    gdf['CNTRY_NAME'] = gdf['CNTRY_NAME'].replace("Russia", "Russian Federation")
    gdf['CNTRY_NAME'] = gdf['CNTRY_NAME'].replace("Zaire", 'Democratic Republic Congo')
    bad_names = list(np.setdiff1d(df['aff_country'].tolist(), gdf['CNTRY_NAME'].tolist()))
    if len(bad_names) > 0:
        print('Some countries are incorrectly named!')
        print(bad_names)
    gdf = pd.merge(gdf, df, how='left', left_on='CNTRY_NAME',
                   right_on='aff_country')
    return gdf


def make_author_table(auth_df, main_df, d_path, table_filter):
    auth_df1 = auth_df.drop_duplicates(subset=['doi', 'authorid']).copy()
    auth_df1.loc[:, 'authorid'] = auth_df1['authorid']
    auth_papercount = auth_df1.groupby(['authorid'])['authorid'].count().sort_values(ascending=False)
    auth_papercount = auth_papercount.reset_index(name='Papers')
    auth_wcites = pd.merge(auth_df1, main_df[['DOI', 'citedbycount', 'Date']], how='left',
                           left_on='doi', right_on='DOI')
    auth_wcites = auth_wcites[auth_wcites['authorid'].notnull()]
    auth_wcites.loc[:, 'authorid'] = auth_wcites['authorid'].astype(int).astype(str)
    auth_wcites['year'] = auth_wcites.Date.str.extract(r'([0-9][0-9][0-9][0-9])', expand=True)
    auth_wcites_grouped = auth_wcites.groupby(['authorid'])['citedbycount'].sum()
    auth_wcites_grouped = auth_wcites_grouped.reset_index(name='Cites')
    auth_flat = pd.read_csv(os.path.join(d_path, 'scopus',
                                         'author', 'parsed',
                                         'scopus_authors.tsv'), sep='\t')
    auth_flat.loc[:, 'fullname'] = auth_flat['givenname'] + ' ' + auth_flat['surname']
    auth_popstudies_h = auth_wcites[['authorid', 'doi', 'citedbycount']]
    auth_popstudies_h = auth_popstudies_h[auth_popstudies_h['authorid'].notnull()]
    auth_popstudies_h = auth_popstudies_h.set_index('authorid')
    auth_flat['H-Index'] = 0
    auth_flat = auth_flat[auth_flat['author'].notnull()]
    auth_flat.loc[:, 'author'] = auth_flat['author'].astype(str)
    auth_flat = auth_flat.set_index('author')
    counter = 0

    ### make h index
    for author in auth_flat.index:
        author = str(int(author))
        counter += 1
        temp = auth_wcites[auth_wcites['authorid'] ==
                           author].sort_values(by='citedbycount',
                                               ascending=False)
        temp = temp.reset_index()
        temp = temp.drop('index', 1)
        for pubnumber in range(0, len(temp)):
            if pubnumber + 1 > temp.loc[pubnumber, 'citedbycount']:
                break
            else:
                auth_flat.loc[author, 'H-Index'] = int(pubnumber + 1)

    ## make hm-index
    auth_wcites_papercount = auth_wcites.groupby(['doi'])['doi'].count()
    auth_wcites_papercount = auth_wcites_papercount.reset_index(name='NumberAuthors')
    auth_flat['HM-Index'] = 0
    auth_wcites = pd.merge(auth_wcites, auth_wcites_papercount,
                           how='left', left_on='DOI_x', right_on='doi')
    auth_wcites['citedbycount_adjusted_by_authors'] = auth_wcites['citedbycount'] / auth_wcites['NumberAuthors']
    for author in auth_flat.index:
        author = str(int(author))
        counter += 1
        temp = auth_wcites[auth_wcites['authorid'] ==
                           author].sort_values(by='citedbycount_adjusted_by_authors',
                                               ascending=False)
        temp = temp.reset_index()
        temp = temp.drop('index', 1)
        for pubnumber in range(0, len(temp)):
            if pubnumber + 1 > temp.loc[pubnumber, 'citedbycount_adjusted_by_authors']:
                break
            else:
                auth_flat.loc[author, 'HM-Index'] = int(pubnumber + 1)

    auth_flat['First'] = np.nan
    auth_flat['Last'] = np.nan
    for auth in auth_flat.index:
        temp = auth_wcites[auth_wcites['authorid'] == auth]
        auth_flat.loc[auth, 'First'] = temp['year'].min()
        auth_flat.loc[auth, 'Last'] = temp['year'].max()
    auth_name = auth_flat[['fullname', 'H-Index', 'HM-Index', 'First', 'Last']].reset_index()
    auth_name = auth_name.rename({'author': 'authorid'}, axis=1)
    auth_name['authorid'] = auth_name['authorid'].astype(int).astype(str)
    auth_papercount['authorid'] = auth_papercount['authorid'].astype(int).astype(str)
    auth_wcites_grouped['authorid'] = auth_wcites_grouped['authorid'].astype(int).astype(str)
    author_df = pd.merge(auth_papercount, auth_wcites_grouped, how='left',
                         left_on='authorid', right_on='authorid')
    author_df['authorid'] = author_df['authorid'].astype(int).astype(str)
    author_df = pd.merge(author_df, auth_name, how='left',
                         left_on='authorid', right_on='authorid')
    author_df = author_df.set_index('fullname')
    auth_out = author_df[['Papers', 'Cites', 'H-Index', 'HM-Index', 'First', 'Last']]
    auth_out = auth_out.sort_values(by=table_filter, ascending=False)
    print(auth_out[0:10])
    auth_out.to_csv(os.path.join(d_path, '..',
                                 'article', 'tables',
                                 'authors_' + table_filter + '.csv'))


def make_word_vis(main_df, figure_path, d_path):
    stop_list = pd.read_csv(os.path.join(d_path, 'support',
                                         'custom_stopwords.txt'))
    custom_stop = stop_list['words'].to_list()
    df_abs, words_abs = freq_dist(main_df, 'english', 'lemmatize_token_str_abstract')
    df_tit, words_tit = freq_dist(main_df, 'english', 'lemmatize_token_str_title')
    en_stemmer = SnowballStemmer('english')
    wordlist = []
    for elem in words_abs:
        wordlist.append(' '.join([en_stemmer.stem(w) for w in elem.split(' ') if w.isalnum()]))
    words_abs_mat = co_occurrence(wordlist, 5)
    wordlist = []
    for elem in words_tit:
        wordlist.append(' '.join([en_stemmer.stem(w) for w in elem.split(' ') if w.isalnum()]))
    words_tit_mat = co_occurrence(wordlist, 5)
    matsize = 25
    abs_sum = words_abs_mat.sum().sum()
    tit_sum = words_tit_mat.sum().sum()
    tot_row_abs = pd.DataFrame(words_abs_mat.sum())
    tot_row_abs = tot_row_abs.sort_values(by=0, ascending=False)[0:matsize]
    words_abs_mat = words_abs_mat[tot_row_abs.index.to_list()]
    words_abs_mat = words_abs_mat.reindex(index=tot_row_abs.index.to_list())

    tot_row_tit = pd.DataFrame(words_tit_mat.sum())
    tot_row_tit = tot_row_tit.sort_values(by=0, ascending=False)[0:matsize]
    words_tit_mat = words_tit_mat[tot_row_tit.index.to_list()]
    words_tit_mat = words_tit_mat.reindex(index=tot_row_tit.index.to_list())
    mask_abs = np.zeros_like(words_abs_mat.iloc[0:matsize, 0:matsize], dtype=np.int16)
    mask_abs[np.triu_indices_from(mask_abs)] = True
    mask_tit = np.zeros_like(words_tit_mat.iloc[0:matsize, 0:matsize], dtype=np.int16)
    mask_tit[np.triu_indices_from(mask_tit)] = True

    pd.DataFrame.set_diag = set_diag
    words_abs_mat.astype(float).set_diag(np.nan)
    words_tit_mat.astype(float).set_diag(np.nan)
    sns.set_style('ticks')

    fig = plt.figure(figsize=(16, 13))
    ax1 = plt.subplot2grid((49, 48), (0, 0), rowspan=21, colspan=24)
    ax2 = plt.subplot2grid((49, 48), (0, 26), rowspan=21, colspan=20, projection='polar')
    ax3 = plt.subplot2grid((49, 48), (28, 0), rowspan=21, colspan=20, projection='polar')
    ax4 = plt.subplot2grid((49, 48), (28, 22), rowspan=21, colspan=24)
    ax1.set_title('A.', loc='left', y=1.01, **hfont, fontsize=21, x=-.05)
#    ax1.set_title("Word Co-occurrence: Abstracts",
#                  loc='center', y=1.01, **hfont, fontsize=17, x=0.55)
    ax2.set_title('B.', loc='left', y=1.01, **hfont, fontsize=21, x=-.35)
#    ax2.set_title("Frequency distribution: Abstracts",
#                  loc='center', y=1.01, **hfont, fontsize=17, x=.40)
    ax3.set_title('C.', loc='left', y=1.01, **hfont, fontsize=22, x=-.20)
#    ax3.set_title("Frequency distribution: Titles",
#                  loc='center', y=1.01, **hfont, fontsize=17, x=0.55)
    ax4.set_title('D.', loc='left', y=1.01, **hfont, fontsize=21, x=0)
#    ax4.set_title("Word Co-occurrence: Titles",
#                  loc='center', y=1.01, **hfont, fontsize=17, x=0.575)



    import matplotlib.ticker as tkr
    from matplotlib.colors import LogNorm
    import math
    formatter = tkr.ScalarFormatter(useMathText=True)
    formatter.set_scientific(False)
    cmap = plt.get_cmap('RdBu_r')
    cmap = truncate_colormap(cmap, 0.0, 1)


    ax_abs = sns.heatmap(words_abs_mat.astype(float),
                         norm=LogNorm(1 + words_abs_mat.astype(float).min().min(),
                                      words_abs_mat.astype(float).max().max()),
                         cbar_kws={'ticks': [1, 10, 50, 100, 150, 200, 300, 400],
                                   "shrink": 1, 'use_gridspec': True,
                                   "format": formatter},
                         mask=mask_abs,
                         vmin=1,
                         vmax=250,
                         cmap=cmap,
                         linewidths=.25,
                         ax=ax1)
    ax_abs.collections[0].colorbar.outline.set_edgecolor('k')
    ax_abs.collections[0].colorbar.outline.set_linewidth(1)
    ax_abs.collections[0].colorbar.ax.yaxis.set_ticks_position('left')
    iN = len(df_abs[0:matsize]['count'])
    labs = df_abs[0:matsize].index
    arrCnts = np.array(df_abs[0:matsize]['count']) + .2
    theta = np.arange(0, 2 * np.pi, 2 * np.pi / (iN))
    width = (5 * np.pi) / iN
    bottom = 0.5
    ax2.set_theta_zero_location('W')
    ax2.plot(theta, len(theta) * [0.55], alpha=0.5, color='k', linewidth=1, linestyle='--')
    bars = ax2.plot(theta, arrCnts, alpha=1, linestyle='-', marker='o',
                    color='#377eb8', markersize=7, markerfacecolor='w',
                    markeredgecolor='#ff5148')
    ax2.axis('off')
    rotations = np.rad2deg(theta)
    y0, y1 = ax2.get_ylim()
    for x, bar, rotation, label in zip(theta, arrCnts, rotations, labs):
        offset = (bottom + bar) / (y1 - y0)
        lab = ax2.text(0, 0, label, transform=None,
                       ha='center', va='center')
        renderer = ax2.figure.canvas.get_renderer()
        bbox = lab.get_window_extent(renderer=renderer)
        invb = ax2.transData.inverted().transform([[0, 0], [bbox.width, 0]])
        lab.set_position((x, offset + (invb[1][0] - invb[0][0]) + .05))
        lab.set_transform(ax2.get_xaxis_transform())
        lab.set_rotation(rotation)
    ax2.fill_between(theta, arrCnts, alpha=0.075, color='#4e94ff')
    ax2.fill_between(theta, len(theta) * [0.55], alpha=1, color='w')
    circle = plt.Circle((0.0, 0.0), 0.1, transform=ax2.transData._b, color="k", alpha=0.3)
    ax2.add_artist(circle)
    ax2.plot((0, theta[0]), (0, arrCnts[0]),
             color='k', linewidth=1, alpha=0.5, linestyle='--')
    ax2.plot((0, theta[-1]), (0, arrCnts[-1]),
             color='k', linewidth=1, alpha=0.5, linestyle='--')

    iN = len(df_tit[0:matsize]['count'])
    labs = df_tit[0:matsize].index
    arrCnts = np.array(df_tit[0:matsize]['count']) + 2
    theta = np.arange(0, 2 * np.pi, 2 * np.pi / (iN))
    width = (5 * np.pi) / iN
    bottom = 0.4
    ax3.set_theta_zero_location('W')
    ax3.plot(theta, len(theta) * [2.35], alpha=0.5, color='k', linewidth=1, linestyle='--')
    bars = ax3.plot(theta, arrCnts, alpha=1, linestyle='-', marker='o',
                    color='#377eb8', markersize=7, markerfacecolor='w',
                    markeredgecolor='#ff5148')
    ax3.axis('off')
    rotations = np.rad2deg(theta)
    y0, y1 = ax3.get_ylim()
    for x, bar, rotation, label in zip(theta, arrCnts, rotations, labs):
        offset = (bottom + bar) / (y1 - y0)
        lab = ax3.text(0, 0, label, transform=None,
                       ha='center', va='center')
        renderer = ax3.figure.canvas.get_renderer()
        bbox = lab.get_window_extent(renderer=renderer)
        invb = ax3.transData.inverted().transform([[0, 0], [bbox.width, 0]])
        lab.set_position((x, offset + (invb[1][0] - invb[0][0]) + .11))
        lab.set_transform(ax3.get_xaxis_transform())
        lab.set_rotation(rotation)
    ax3.fill_between(theta, arrCnts, alpha=0.075, color='#4e94ff')
    ax3.fill_between(theta, len(theta) * [2.35], alpha=1, color='w')
    circle = plt.Circle((0.0, 0.0), 0.35, transform=ax3.transData._b, color="k", alpha=0.3)
    ax3.add_artist(circle)
    ax3.plot((0, theta[0]), (0, arrCnts[0]),
             color='k', linewidth=1, alpha=0.5, linestyle='--')
    ax3.plot((0, theta[-1]), (0, arrCnts[-1]),
             color='k', linewidth=1, alpha=0.5, linestyle='--')

    formatter = tkr.ScalarFormatter(useMathText=True)
    formatter.set_scientific(False)
    cmap = plt.get_cmap('RdBu_r')
    cmap = truncate_colormap(cmap, 0.0, 1)
    words_tit_mat = words_tit_mat.replace(0, 0.00001)
    ax_tit = sns.heatmap(words_tit_mat.astype(float),
                         norm=LogNorm(1+ words_tit_mat.astype(float).min().min(),
                                      words_tit_mat.astype(float).max().max()),
                         cbar_kws={'ticks': [0, 5, 10, 25, 50],
                                   "shrink": 1, 'use_gridspec': True,
                                   "format": formatter},
                         mask=mask_tit,
                         cmap=cmap,
                         linewidths=.25,
                         vmin=0.0000, vmax=50,
                         ax=ax4)
    ax_tit.collections[0].colorbar.ax.yaxis.set_ticks_position('left')
    ax_tit.collections[0].colorbar.outline.set_edgecolor('k')
    ax_tit.collections[0].colorbar.outline.set_linewidth(1)
    for _, spine in ax1.spines.items():
        spine.set_visible(True)
    for _, spine in ax4.spines.items():
        spine.set_visible(True)
    sns.despine(ax=ax1)
    sns.despine(ax=ax4)
    plt.savefig(os.path.join(figure_path, 'word_freq_cooc.svg'),
                bbox_inches='tight')
    plt.savefig(os.path.join(figure_path, 'word_freq_cooc.pdf'),
                bbox_inches='tight')
    plt.savefig(os.path.join(figure_path, 'word_freq_cooc.png'),
                bbox_inches='tight', dpi=600)


def set_diag(self, values):
    n = min(len(self.index), len(self.columns))
    self.values[[np.arange(n)] * 2] = values


def freq_dist(df, language, fieldname):
    sentences = '\n'.join(df[fieldname].astype(str).str.lower().tolist())
    en_stemmer = SnowballStemmer(language)
    words = word_tokenize(sentences, language=language)
    counts = FreqDist(en_stemmer.stem(w) for w in words if w.isalnum())
    df_fdist = pd.DataFrame.from_dict(counts, orient='index', columns=['count'])
    df_fdist = df_fdist.sort_values(by='count', ascending=False)
    df_fdist['count'] = (df_fdist['count'] / df_fdist['count'].sum()) * 100
    df_fdist.index = df_fdist.index.str.title() + ': ' + \
                     df_fdist['count'].round(1).astype(str) + '%'
    return df_fdist, df[fieldname].astype(str).str.lower().tolist()


def co_occurrence(sentences, window_size):
    d = defaultdict(int)
    vocab = set()
    for text in sentences:
        # preprocessing (use tokenizer instead)
        text = text.lower().split()
        # iterate over sentences
        for i in range(len(text)):
            token = text[i]
            vocab.add(token)  # add to vocab
            next_token = text[i + 1: i + 1 + window_size]
            for t in next_token:
                key = tuple(sorted([t, token]))
                d[key] += 1

    # formulate the dictionary into dataframe
    vocab = sorted(vocab)  # sort vocab
    df = pd.DataFrame(data=np.zeros((len(vocab), len(vocab)), dtype=np.int16),
                      index=vocab,
                      columns=vocab)
    for key, value in d.items():
        df.at[key[0], key[1]] = value
        df.at[key[1], key[0]] = value
    return df
