import numpy as np
import pandas as pd
import anndata as ad
import seaborn as sns
from matplotlib import pyplot as plt
def datasets_assess(
    datasets = [
        dict(
            author = 'Kang et al.\n(7 cell types)',
            experiment_name = 'across_cell_types',
            data_file = 'pbmc',
            file_type = '.h5ad',
            cond_key = 'condition',
            ctrl_key = 'control',
            stim_key = 'stimulated',
            cell_label_key = 'cell_type',
        ),
        dict(
            author = 'Kang et al.\n(8 patients)',
            experiment_name = 'across_patients',
            data_file = 'pbmc_patients',
            file_type = '.h5ad',
            cond_key = 'condition',
            ctrl_key = 'ctrl',
            stim_key = 'stim',
            cell_label_key = 'sample_id',
        ),
        dict(
            author = 'Hagai et al.',
            experiment_name = 'across_species',
            data_file = 'species',
            file_type = '.h5ad',
            cond_key = 'condition',
            ctrl_key = 'unst',
            stim_key = 'LPS6',
            cell_label_key = 'species',
        ),
        dict(
            author = 'Jiang et al.',
            experiment_name = 'across_cell_lines',
            data_file = 'IFNGR2',
            file_type = '.h5ad',
            cond_key = 'target_gene',
            ctrl_key = 'non-targeting',
            stim_key = 'IFNGR2',
            cell_label_key = 'cell_type',
        ),
    ],
    model_names: list[str] = ['scPILOT', 'scGen', 'CellOT', 'biolord', 'identity', 'VAEGAN'],
):
    def distance(x: np.ndarray, y: np.ndarray):
        return ((x - y) ** 2).sum()
    sns.set_theme(style = 'white', font = 'Arial', font_scale = 1.5)
    authors = []
    affiliations = []
    values = []
    values_r2 = []
    values_mmd = []
    for dataset in datasets:
        author = dataset['author']
        experiment_name = dataset['experiment_name']
        data_file = dataset['data_file']
        file_type = dataset['file_type']
        cond_key = dataset['cond_key']
        ctrl_key = dataset['ctrl_key']
        stim_key = dataset['stim_key']
        cell_label_key = dataset['cell_label_key']
        adata = ad.read_h5ad(f'../Data/{experiment_name}/{data_file}{file_type}')
        query_keys = sorted(adata.obs[cell_label_key].unique().tolist())
        learnability_dict = {}
        for query_key in query_keys:
            adata_query_ctrl = adata[(adata.obs[cell_label_key] == query_key) & (adata.obs[cond_key] == ctrl_key)].copy()
            adata_query_stim = adata[(adata.obs[cell_label_key] == query_key) & (adata.obs[cond_key] == stim_key)].copy()
            adata_other_ctrl = adata[(adata.obs[cell_label_key] != query_key) & (adata.obs[cond_key] == ctrl_key)].copy()
            mean_query_ctrl = np.mean(
                adata_query_ctrl.X.toarray()
                if hasattr(adata_query_ctrl.X, 'toarray')
                else adata_query_ctrl.X,
                axis = 0,
            )
            mean_query_stim = np.mean(
                adata_query_stim.X.toarray()
                if hasattr(adata_query_stim.X, 'toarray')
                else adata_query_stim.X,
                axis = 0,
            )
            mean_other_ctrl = np.mean(
                adata_other_ctrl.X.toarray()
                if hasattr(adata_other_ctrl.X, 'toarray')
                else adata_other_ctrl.X,
                axis = 0,
            )
            learnability = distance(
                mean_query_ctrl, mean_query_stim
            ) / (distance(
                mean_query_ctrl, mean_query_stim
            ) + distance(
                mean_query_ctrl, mean_other_ctrl
            ))
            learnability_dict.update(
                {query_key: learnability}
            )
        learnability_array = np.fromiter(learnability_dict.values(), dtype = np.float32)
        authors.append(author)
        affiliations.append('Learnability')
        values.append(learnability_array.mean())
        values_r2.append(learnability_array.mean())
        values_mmd.append(learnability_array.mean())
        r2mean_final_df: pd.DataFrame = pd.read_pickle(f'../DataFrames/{experiment_name}/r2mean_final.pkl')
        mmd_final_df: pd.DataFrame = pd.read_pickle(f'../DataFrames/{experiment_name}/mmd_final.pkl')
        for model_name in sorted(model_names):
            baseline_r2 = r2mean_final_df[
                (r2mean_final_df['Gene set'] == 'All genes') & (r2mean_final_df['Model'] == 'identity')
            ]['R2mean means'].to_numpy().mean()
            model_r2 = r2mean_final_df[
                (r2mean_final_df['Gene set'] == 'All genes') & (r2mean_final_df['Model'] == model_name)
            ]['R2mean means'].to_numpy().mean()
            baseline_mmd = mmd_final_df[mmd_final_df['Model'] == 'identity']['MMD means'].to_numpy().mean()
            model_mmd = mmd_final_df[mmd_final_df['Model'] == model_name]['MMD means'].to_numpy().mean()
            score = (model_r2 / (model_r2 + baseline_r2) + baseline_mmd / (baseline_mmd + model_mmd)) / 2
            authors.append(author)
            affiliations.append(model_name)
            values.append(score)
            values_r2.append(model_r2)
            values_mmd.append(model_mmd)
    learnabilityNscore_df = pd.DataFrame({
        'Dataset': authors,
        'Affiliations': affiliations,
        'Score': values,
    })
    print(f'learnabilityNscore_df:\n{learnabilityNscore_df}')
    learnabilityNscore_df = learnabilityNscore_df.replace('biolord', 'Biolord').replace('identity', 'Identity')
    ax = sns.lineplot(data = learnabilityNscore_df, x = 'Dataset', y = 'Score', hue = 'Affiliations', hue_order = [
        'Learnability', 'Biolord', 'CellOT', 'Identity', 'VAEGAN', 'scGen', 'scPILOT'
    ], style = 'Affiliations', markers = True)
    plt.legend(bbox_to_anchor = (1.01, 0.5), loc = 'center left', borderaxespad = 0)
    plt.savefig(f'../Figures/datasets_assess/learnabilityNscore_lineplot.jpg', dpi = 300, bbox_inches = 'tight')
    plt.close()
    learnabilityNr2_df = pd.DataFrame({
        'Dataset': authors,
        'Affiliations': affiliations,
        'Score_r2': values_r2,
    })
    print(f'learnabilityNr2_df:\n{learnabilityNr2_df}')
    learnabilityNmmd_df = pd.DataFrame({
        'Dataset': authors,
        'Affiliations': affiliations,
        'Score_mmd': values_mmd,
    })
    print(f'learnabilityNmmd_df:\n{learnabilityNmmd_df}')
if __name__ == '__main__':
    datasets_assess()
    print('Done')