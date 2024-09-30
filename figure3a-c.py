import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.cluster import hierarchy

def generate_palette(values, pal='RdBu_r'):
    uniq_val = np.sort(values.loc[~values.isna()].unique())
    pal = sns.color_palette(pal, len(uniq_val))
    palette = dict(zip(uniq_val, pal))
    return values.map(palette)


### Read dataset
SOURCE_FILE = r'Supplementary Table S6.xlsx'
df = pd.read_excel(SOURCE_FILE, index_col='Gene Name', skiprows=1)

### Plot clustered heatmap
CLUSTERING_COLS = ['Healthy', 'CB', 'NCB']
CORR_PVAL_THRESHOLD = 0.05
CLUSTER_COLORS = ['C0', 'C3', 'C1', 'C2', 'C4', 'cyan', 'C6', 'C5', 'C7', 'blue', 'yellow', 'olive', 'pink', 'black', 'green', 'wheat', 'yellowgreen', 'fuchsia', 'turquoise', 'maroon']

# Define palettes
norm = matplotlib.colors.Normalize(-1,1)
colors1 = [[norm(-1.00), 'blue'],
          [norm(-0.00), 'white'],
          [norm( 1.00), 'red']] # Dark blue
colors2 = [[0, 'cyan'],
           [0.5, 'white'],
           [1, 'darkred']]
white_cyan_blue = matplotlib.colors.LinearSegmentedColormap.from_list('WhiteCyanBlue', colors1)
red_white_blue = matplotlib.colors.LinearSegmentedColormap.from_list('RedBlue', colors2)

prot_lnkg = hierarchy.linkage(df[CLUSTERING_COLS].values, method='ward', optimal_ordering=True)
dend = hierarchy.dendrogram(prot_lnkg, no_plot=True, color_threshold=-np.inf)
clust_clr = generate_palette(df['Cluster'], CLUSTER_COLORS)

corr_values = np.unique(df[['BMI Spearman r', 'LDH Spearman r', 'CRP Spearman r']].values.flatten())
norm = matplotlib.colors.Normalize(vmin=-np.abs(corr_values).max(),vmax=np.abs(corr_values).max())
corr_colors = red_white_blue(norm(corr_values))
corr_map = dict(zip(corr_values, corr_colors))
bmi_cor_clr = df['BMI Spearman r'].map(corr_map)
ldh_cor_clr = df['LDH Spearman r'].map(corr_map)
crp_cor_clr = df['CRP Spearman r'].map(corr_map)
bmi_cor_clr.loc[df['BMI p-value'] > CORR_PVAL_THRESHOLD] = 'lightgray'
ldh_cor_clr.loc[df['LDH p-value'] > CORR_PVAL_THRESHOLD] = 'lightgray'
crp_cor_clr.loc[df['CRP p-value'] > CORR_PVAL_THRESHOLD] = 'lightgray'
row_color_df = pd.DataFrame(zip(clust_clr, bmi_cor_clr, ldh_cor_clr, crp_cor_clr),index=df.index,
                            columns=['Cluster', 'BMI Corr', 'LDH Corr', 'CRP Corr'])
row_color_df = pd.DataFrame(zip(clust_clr, bmi_cor_clr, ldh_cor_clr, crp_cor_clr), index=df.index,
                            columns=['Cluster', 'BMI Corr', 'LDH Corr', 'CRP Corr'])
cbar_kws={'orientation': 'horizontal'}
tree_kws={}
clim = np.max(np.abs([df[CLUSTERING_COLS].min().min(), df[CLUSTERING_COLS].max().max()]))
plt.rcParams.update({'font.size': 10})
cg = sns.clustermap(df[CLUSTERING_COLS], 
                row_linkage=prot_lnkg, col_cluster=False,
                row_colors = row_color_df,
                linewidths=8.0, figsize=(16.66, 50), cmap=white_cyan_blue, cbar_kws=cbar_kws, cbar_pos=(.193, .95, .71, .019), # 'Blues', white_cyan_blue
                dendrogram_ratio=(.3, .1), vmin=-clim, vmax=clim)
cg.ax_row_colors.xaxis.tick_top()
cg.ax_row_colors.set_xticklabels(cg.ax_row_colors.get_xticklabels(), rotation = 90, fontsize=30)
cg.ax_heatmap.xaxis.tick_top()
cg.ax_heatmap.set_xticklabels(cg.ax_heatmap.get_xticklabels(), rotation = 45, fontsize=40)
cg.ax_heatmap.set_yticklabels(cg.ax_heatmap.get_yticklabels(), rotation = 0, fontsize=20)
cg.ax_cbar.set_xticklabels(cg.ax_cbar.get_xticklabels(), rotation = 0, fontsize=40)
cg.ax_cbar.set_title('norm(mean(log(RFU)))', fontsize=40)
plt.show()

# Plot grouped cluster and pattern expression level medians
plt.rcParams.update({'font.size': 20})
GROUP_BY_COLS = ['Cluster', 'Pattern']
for group_by_col in GROUP_BY_COLS:
    n_patterns = len(df[group_by_col].unique())
    N_ROWS = 5
    n_cols = (n_patterns-1) // N_ROWS + 1
    fig, axes = plt.subplots(N_ROWS, n_cols, figsize=(3.5 * n_cols, 4 * N_ROWS), constrained_layout=True)
    for ptrn in range(N_ROWS * n_cols):
        if group_by_col == 'Cluster':
            ptrn_val = ptrn+1
            title = f'Cluster {ptrn_val}'
        elif group_by_col == 'Pattern':
            ptrn_val = chr(65+ptrn)
            title = f'Pattern {ptrn_val}'
        else:
            raise ValueError('Invalid groupby column')
        r = ptrn % N_ROWS
        c = ptrn//N_ROWS
        if n_cols == 1:
            ax = axes[r]
        else:
            ax = axes[r, c]
        
        if ptrn >= n_patterns:
            ax.remove()
        else:
            ax.plot(df.loc[df[group_by_col] == ptrn_val, CLUSTERING_COLS].transpose(), c=CLUSTER_COLORS[ptrn], marker='D')
            ax.set_ylim([-1.4,1.2])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_title(title)
            for tick in ax.get_xticklabels():
                tick.set_rotation(45)
            ax.text(1, 1, f'n = {(df[group_by_col] == ptrn).sum()}',
                      horizontalalignment='center', verticalalignment='top')
    plt.show()