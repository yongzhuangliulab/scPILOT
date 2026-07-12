import os
import scanpy as sc
import anndata as ad
import seaborn as sns
from matplotlib import pyplot as plt, transforms


def ensure_dirs(*paths):
    for path in paths:
        os.makedirs(path, exist_ok=True)


def save_figure_jpg_pdf(path_to_save, dpi=300, bbox_inches='tight'):
    root, _ = os.path.splitext(path_to_save)
    plt.savefig(root + '.jpg', dpi=dpi, bbox_inches=bbox_inches)
    plt.savefig(root + '.pdf', bbox_inches=bbox_inches)


def perturbation_prediction_dotplot(
    experiment_name = 'across_cell_types',
    model_name = 'scPILOT',
    seeds=(1327, 1337, 1347),
    data_file = 'pbmc',
    file_type = '.h5ad',
    cond_key = 'condition',
    ctrl_key = 'control',
    stim_key = 'stimulated',
    cell_label_key = 'cell_type',
):
    sns.set_theme(style = 'white', font = 'Arial', font_scale = 2)
    ensure_dirs(f'../Figures/{experiment_name}')
    adata = sc.read_h5ad(f'../Data/{experiment_name}/{data_file}{file_type}')
    adata = adata[(adata.obs[cond_key] == ctrl_key) | (adata.obs[cond_key] == stim_key)].copy()
    print('adata:')
    print(adata)
    query_keys = sorted(adata.obs[cell_label_key].unique().tolist())
    adata_query_eval_list = []

    for query_no, query_key in enumerate(query_keys):
        truth_added = False

        for seed in seeds:
            file_path = (
                f'../Result_anndata/{experiment_name}/'
                f'{experiment_name}_{model_name}_{data_file}_{query_key}_seed{seed}{file_type}'
            )

            print(f'Reading: {file_path}', flush=True)
            adata_query_eval = ad.read_h5ad(file_path)

            if not truth_added:
                adata_truth = adata_query_eval[
                    adata_query_eval.obs[cond_key].isin([ctrl_key, stim_key])
                ].copy()
                adata_truth.obs['seed'] = 'truth'
                adata_truth.obs_names = [
                    f'{idx}_{query_key}_truth'
                    for idx in adata_truth.obs_names.astype(str)
                ]
                adata_query_eval_list.append(adata_truth)
                truth_added = True

            adata_pred = adata_query_eval[
                adata_query_eval.obs[cond_key] == 'pred'
            ].copy()
            adata_pred.obs['seed'] = seed
            adata_pred.obs_names = [
                f'{idx}_{query_key}_seed{seed}_pred'
                for idx in adata_pred.obs_names.astype(str)
            ]
            adata_query_eval_list.append(adata_pred)

    adata_eval = ad.concat(adata_query_eval_list)
    adata_eval.obs_names_make_unique()
    gene_groups = {
        'Cell-type markers': [
            'CD3D', 'CCL5', 'GNLY',
            'CD79A',
            'FCGR3A', 'S100A9',
            'HLA-DQA1',
        ],
        'Common IFN-β markers': [
            'ISG15', 'IFI6', 'IFIT1',
        ],
        'Cell-type-specific IFN-β markers': [
            'CXCL10', 'CXCL11',
            'APOBEC3A', 'DEFB1',
            'CCL8', 'TARBP1',
        ],
    }
    group_names = list(gene_groups.keys())
    group_sizes = [len(v) for v in gene_groups.values()]
    gene_list = [gene for genes in gene_groups.values() for gene in genes]
    adata_eval.obs['condition'] = adata_eval.obs['condition'].replace('stimulated', 'stim').replace('control', 'ctrl')
    adata_eval.obs['cell_type_condition'] = (
        adata_eval.obs['cell_type'].astype(str) + '_' + adata_eval.obs['condition'].astype(str)
    ).astype('category')
    categories_order = [
        'B_ctrl', 'B_stim', 'B_pred',
        'CD4T_ctrl', 'CD4T_stim', 'CD4T_pred',
        'CD8T_ctrl', 'CD8T_stim', 'CD8T_pred',
        'CD14+Mono_ctrl', 'CD14+Mono_stim', 'CD14+Mono_pred',
        'Dendritic_ctrl', 'Dendritic_stim', 'Dendritic_pred',
        'FCGR3A+Mono_ctrl', 'FCGR3A+Mono_stim', 'FCGR3A+Mono_pred',
        'NK_ctrl', 'NK_stim', 'NK_pred',
    ]
    dp = sc.pl.dotplot(
        adata_eval,
        var_names=gene_list,
        groupby='cell_type_condition',
        categories_order=categories_order,
        show=False,
        return_fig=True,
    )
    dp.style(cmap = 'Purples', dot_edge_color = None, dot_edge_lw = 0)
    dp_ax_dict = dp.get_axes()
    print(dp_ax_dict)
    mainplot_ax = dp_ax_dict['mainplot_ax']
    mainplot_ax.tick_params(axis='x', labelsize=13, rotation=90)
    mainplot_ax.tick_params(axis='y', labelsize=13)
    mainplot_ax.set_xlabel('')
    mainplot_ax.set_ylabel('')

    # ------------------------------------------------------------------
    # Draw horizontal separator lines between cell-type blocks.
    # Robustly compute the boundary positions from actual y tick locations.
    # Each cell type has 3 rows: ctrl, stim, pred.
    # ------------------------------------------------------------------
    yticks = mainplot_ax.get_yticks()
    n_conditions_per_celltype = 3
    n_celltypes = len(categories_order) // n_conditions_per_celltype

    for block_idx in range(1, n_celltypes):
        prev_last = block_idx * n_conditions_per_celltype - 1
        next_first = block_idx * n_conditions_per_celltype

        y_boundary = 0.5 * (yticks[prev_last] + yticks[next_first])

        mainplot_ax.axhline(
            y=y_boundary,
            color='black',
            linewidth=0.7,
            alpha=0.30,
            zorder=0,
        )


    # ------------------------------------------------------------------
    # Draw bottom gene-group brackets and group labels.
    # Brackets are placed below rotated gene labels.
    # ------------------------------------------------------------------
    trans = transforms.blended_transform_factory(
        mainplot_ax.transData,   # x in gene-index coordinates
        mainplot_ax.transAxes,   # y in axes-relative coordinates
    )

    group_boundaries = []
    start_idx = 0

    # Move all brackets and group labels slightly to the right.
    bracket_x_shift = 0.5

    # Shrink each bracket inward to create visible gaps between adjacent groups.
    bracket_gap = 0.12

    for group_name, genes in gene_groups.items():
        end_idx = start_idx + len(genes) - 1

        # True gene-group boundary.
        x0 = start_idx - 0.5
        x1 = end_idx + 0.5

        # Displayed bracket boundary:
        # shifted to the right and slightly shortened at both ends.
        bracket_x0 = x0 + bracket_gap + bracket_x_shift
        bracket_x1 = x1 - bracket_gap + bracket_x_shift
        bracket_xc = 0.5 * (bracket_x0 + bracket_x1)

        group_boundaries.append(
            (group_name, start_idx, end_idx, bracket_x0, bracket_x1, bracket_xc)
        )

        start_idx = end_idx + 1

    # Bottom brackets.
    for group_name, start_idx, end_idx, x0, x1, xc in group_boundaries:
        # Negative y values place annotations below the main axis.
        # The bracket is below the vertical gene labels.
        y_line = -0.23
        y_tick_top = -0.21
        y_text = -0.27

        # Horizontal bracket line.
        mainplot_ax.plot(
            [x0, x1],
            [y_line, y_line],
            color='black',
            linewidth=1.4,
            transform=trans,
            clip_on=False,
            zorder=20,
        )

        # Left upward tick.
        mainplot_ax.plot(
            [x0, x0],
            [y_line, y_tick_top],
            color='black',
            linewidth=1.4,
            transform=trans,
            clip_on=False,
            zorder=20,
        )

        # Right upward tick.
        mainplot_ax.plot(
            [x1, x1],
            [y_line, y_tick_top],
            color='black',
            linewidth=1.4,
            transform=trans,
            clip_on=False,
            zorder=20,
        )

        if group_name == 'Cell-type markers':
            label_text = 'Cell-type\nmarkers'
        elif group_name == 'Common IFN-β markers':
            label_text = 'Common\nIFN-β markers'
        elif group_name == 'Cell-type-specific IFN-β markers':
            label_text = 'Cell-type-specific\nIFN-β markers'
        else:
            label_text = group_name

        mainplot_ax.text(
            xc,
            y_text,
            label_text,
            ha='center',
            va='top',
            rotation=0,
            fontsize=11,
            transform=trans,
            clip_on=False,
            zorder=20,
        )


    size_legend_ax = dp_ax_dict['size_legend_ax']
    size_legend_ax.set_title('Fraction of cells\nin group (%)', fontsize = 14)
    size_legend_ax.tick_params(axis = 'x', labelsize = 12)
    color_legend_ax = dp_ax_dict['color_legend_ax']
    color_legend_ax.set_title('Mean expression\nin group', fontsize = 14)
    color_legend_ax.tick_params(axis = 'x', labelsize = 12)
    save_figure_jpg_pdf(
        f'../Figures/{experiment_name}/{model_name}_dotplot.jpg',
        dpi=300,
        bbox_inches='tight',
    )
    plt.close('all')
if __name__ == '__main__':
    perturbation_prediction_dotplot()
    print('Done')