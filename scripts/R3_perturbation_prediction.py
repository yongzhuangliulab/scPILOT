import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Literal
from scgen import SCGEN
from scpilot.egd_model import EGD_model
def predict_perturbation(
    experiment_name = 'R3',
    model_name: Literal['scPILOT', 'scGen'] = 'scPILOT',
    data_file = 'pbmc_patients',
    file_type = '.h5ad',
    cond_key = 'condition',
    ctrl_key = 'ctrl',
    stim_key = 'stim',
    cell_label_key = 'sample_id',
):
    sns.set_theme(font = 'Times New Roman', font_scale = 2)
    adata = sc.read_h5ad(f"../Data/{experiment_name}/{data_file}{file_type}")
    adata = adata[(adata.obs[cond_key] == ctrl_key) | (adata.obs[cond_key] == stim_key)].copy()
    print('adata:')
    print(adata)
    sc.pp.pca(adata)
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)
    sc.pl.umap(adata, color = [cond_key], wspace = 0.4, frameon = False, show = False)
    plt.savefig(f'../Figures/{experiment_name}/temp0.png', dpi = 200, bbox_inches = 'tight')
    plt.close()
    sc.pl.umap(adata, color = [cell_label_key], wspace = 0.4, frameon = False, show = False)
    plt.savefig(f'../Figures/{experiment_name}/temp1.png', dpi = 200, bbox_inches = 'tight')
    plt.close()
    plt.figure(figsize = (6, 12))
    temp_fig = [0, 1]
    for temp_fig_no in range(2):
        plt.subplot2grid((2, 1), (temp_fig[temp_fig_no], 0), colspan = 1, rowspan = 1)
        plt.axis('off')
        plt.imshow(plt.imread(f'../Figures/{experiment_name}/temp{temp_fig[temp_fig_no]}.png'))
    plt.savefig(f'../Figures/{experiment_name}/{data_file}_UMAP.png', dpi = 200, bbox_inches = 'tight')
    plt.close()
    query_keys = sorted(adata.obs[cell_label_key].unique().tolist())
    for query_no, query_key in enumerate(query_keys):
        print(f'======Predicting {query_no + 1}: {query_key}======')
        train = adata[~((adata.obs[cell_label_key] == query_key) &
                            (adata.obs[cond_key] == stim_key))].copy()
        print('train:')
        print(train)
        # ======SCGEN======
        # SCGEN.setup_anndata(train, batch_key = cond_key, labels_key = cell_label_key)
        # model = SCGEN(train)
        # model.train(
        #     max_epochs=100,
        #     batch_size=32,
        #     early_stopping=True,
        #     early_stopping_patience=25
        # )
        # model.save(f'../model_trained/{experiment_name}/SCGEN_trained_on_{data_file}_{cell_label}.model', overwrite=True, save_anndata=True)
        # model = SCGEN.load(f'../model_trained/{experiment_name}/SCGEN_trained_on_{data_file}_{cell_label}.model')
        # ======SCGEN======
        # ======scPILOT======
        EGD_model.setup_anndata(train, batch_key = cond_key, labels_key = cell_label_key)
        model = EGD_model(train)
        model.train(
            max_epochs = 400,
            batch_size = 32,
            early_stopping = True,
            early_stopping_patience = 25,
        )
        model.save(f'../model_trained/{experiment_name}/EGD_model_trained_on_{data_file}_{query_key}.model', overwrite = True, save_anndata = True)
        # model = EGD_model.load(f'../model_trained/{experiment_name}/EGD_model_trained_on_{data_file}_{query_key}.model')
        # ======scPILOT======
        latent_z_train = model.get_latent_representation()
        anndata_form_latent_z_train = ad.AnnData(
            X = latent_z_train,
            obs = train.obs.copy(),
        )
        sc.pp.neighbors(anndata_form_latent_z_train, use_rep = 'X')
        sc.tl.umap(anndata_form_latent_z_train)
        sc.pl.umap(anndata_form_latent_z_train, color = [cond_key], wspace = 0.4, frameon = False, show = False)
        plt.savefig(f'../Figures/{experiment_name}/temp2.png', dpi = 200, bbox_inches = 'tight')
        plt.close()
        sc.pl.umap(anndata_form_latent_z_train, color = [cell_label_key], wspace = 0.4, frameon = False, show = False)
        plt.savefig(f'../Figures/{experiment_name}/temp3.png', dpi = 200, bbox_inches = 'tight')
        plt.close()
        plt.figure(figsize = (6, 12))
        temp_fig = [2, 3]
        for temp_fig_no in range(2):
            plt.subplot2grid((2, 1), (temp_fig_no, 0), colspan = 1, rowspan = 1)
            plt.axis('off')
            plt.imshow(plt.imread(f'../Figures/{experiment_name}/temp{temp_fig[temp_fig_no]}.png'))
        plt.savefig(f'../Figures/{experiment_name}/{experiment_name}_{model_name}_on_{data_file}_{query_key}_latent_space.png', dpi = 200, bbox_inches = 'tight')
        plt.close()
        adata_query_ctrl = adata[((adata.obs[cell_label_key] == query_key) & (adata.obs[cond_key] == ctrl_key))].copy()
        adata_query_stim = adata[((adata.obs[cell_label_key] == query_key) & (adata.obs[cond_key] == stim_key))].copy()
        adata_query = adata[adata.obs[cell_label_key] == query_key].copy()
        sc.tl.rank_genes_groups(adata_query, groupby = cond_key, method = 'wilcoxon', n_genes = 100)
        diff_genes = adata_query.uns['rank_genes_groups']['names'][stim_key]
        for ot_flag in range(2):
            if ot_flag == 1 and model_name == 'scGen':
                continue
            if ot_flag == 0:
                adata_query_pred, _ = model.predict(
                    ctrl_key = ctrl_key,
                    stim_key = stim_key,
                    celltype_to_predict = query_key,
                )
            elif ot_flag == 1:
                adata_query_pred, _ = model.predict_new(
                    ctrl_key = ctrl_key,
                    stim_key = stim_key,
                    query_key = query_key,
                )
            adata_query_pred.obs[cond_key] = 'pred'
            adata_query_eval = ad.concat([adata_query_stim, adata_query_pred])
            sc.pp.pca(adata_query_eval)
            sc.pp.neighbors(adata_query_eval)
            sc.tl.umap(adata_query_eval)
            sc.pl.umap(adata_query_eval, color = cond_key, frameon = False, show = False)
            if ot_flag == 0 and model_name == 'scPILOT':
                plt.savefig(f'../Figures/{experiment_name}/VAEGAN_{data_file}_prediction_{query_key}.png', dpi = 200, bbox_inches = 'tight')
                plt.close()
                plt.figure()
                r2mean_all, r2mean_top100 = model.reg_mean_plot(
                    adata_query_eval,
                    axis_keys = {"x": "pred", "y": stim_key},
                    gene_list = diff_genes[:10],
                    top_100_genes = diff_genes,
                    labels = {"x": "Prediction","y": "Ground truth"},
                    path_to_save = f"../Figures/{experiment_name}/VAEGAN_{data_file}_reg_mean_{query_key}.png",
                    show = False,
                    legend = False,
                )
                plt.figure()
                r2var_all, r2var_top100 = model.reg_var_plot(
                    adata_query_eval,
                    axis_keys = {"x": "pred", "y": stim_key},
                    gene_list = diff_genes[:10],
                    top_100_genes = diff_genes,
                    labels = {"x": "Prediction","y": "Ground truth"},
                    path_to_save = f"../Figures/{experiment_name}/VAEGAN_{data_file}_reg_var_{query_key}.png",
                    show = False,
                    legend = False,
                )
                print(f'VAEGAN:')
                print(f'{query_key}: r2mean_all = {r2mean_all}, r2mean_top100 = {r2mean_top100}')
                print(f'{query_key}: r2var_all = {r2var_all}, r2var_top100 = {r2var_top100}')
                adata_query_eval = ad.concat([adata_query_ctrl, adata_query_eval])
                sc.pl.violin(adata_query_eval, keys = diff_genes[0], groupby = cond_key, rotation = 90, show = False)
                plt.savefig(f'../Figures/{experiment_name}/VAEGAN_{data_file}_diff_genes[0]_violin_{query_key}.png', dpi = 200, bbox_inches = 'tight')
                plt.close()
                adata_query_eval.write_h5ad(f'../Result_anndata/{experiment_name}/{experiment_name}_VAEGAN_{data_file}_{query_key}{file_type}')
            elif ot_flag == 0 and model_name == 'scGen':
                plt.savefig(f'../Figures/{experiment_name}/{model_name}_{data_file}_prediction_{query_key}.png', dpi = 200, bbox_inches = 'tight')
                plt.close()
                plt.figure()
                r2mean_all, r2mean_top100 = model.reg_mean_plot(
                    adata_query_eval,
                    axis_keys = {"x": "pred", "y": stim_key},
                    gene_list = diff_genes[:10],
                    top_100_genes = diff_genes,
                    labels = {"x": "Prediction","y": "Ground truth"},
                    path_to_save = f"../Figures/{experiment_name}/{model_name}_{data_file}_reg_mean_{query_key}.png",
                    show = False,
                    legend = False,
                )
                plt.figure()
                EGD_model.setup_anndata(adata_query_eval, batch_key = cond_key, labels_key = cell_label_key)
                r2var_all, r2var_top100 = EGD_model(adata_query_eval).reg_var_plot(
                    adata_query_eval,
                    axis_keys = {"x": "pred", "y": stim_key},
                    gene_list = diff_genes[:10],
                    top_100_genes = diff_genes,
                    labels = {"x": "Prediction","y": "Ground truth"},
                    path_to_save = f"../Figures/{experiment_name}/{model_name}_{data_file}_reg_var_{query_key}.png",
                    show = False,
                    legend = False,
                )
                print(f'{model_name}:')
                print(f'{query_key}: r2mean_all = {r2mean_all}, r2mean_top100 = {r2mean_top100}')
                print(f'{query_key}: r2var_all = {r2var_all}, r2var_top100 = {r2var_top100}')
                adata_query_eval = ad.concat([adata_query_ctrl, adata_query_eval])
                sc.pl.violin(adata_query_eval, keys = diff_genes[0], groupby = cond_key, rotation = 90, show = False)
                plt.savefig(f'../Figures/{experiment_name}/{model_name}_{data_file}_diff_genes[0]_violin_{query_key}.png', dpi = 200, bbox_inches = 'tight')
                plt.close()
                adata_query_eval.write_h5ad(f'../Result_anndata/{experiment_name}/{experiment_name}_{model_name}_{data_file}_{query_key}{file_type}')
            elif ot_flag == 1 and model_name == 'scPILOT':
                plt.savefig(f'../Figures/{experiment_name}/{model_name}_{data_file}_prediction_{query_key}.png', dpi = 200, bbox_inches = 'tight')
                plt.close()
                plt.figure()
                r2mean_all, r2mean_top100 = model.reg_mean_plot(
                    adata_query_eval,
                    axis_keys = {"x": "pred", "y": stim_key},
                    gene_list = diff_genes[:10],
                    top_100_genes = diff_genes,
                    labels = {"x": "Prediction","y": "Ground truth"},
                    path_to_save = f"../Figures/{experiment_name}/{model_name}_{data_file}_reg_mean_{query_key}.png",
                    show = False,
                    legend = False,
                )
                plt.figure()
                r2var_all, r2var_top100 = model.reg_var_plot(
                    adata_query_eval,
                    axis_keys = {"x": "pred", "y": stim_key},
                    gene_list = diff_genes[:10],
                    top_100_genes = diff_genes,
                    labels = {"x": "Prediction","y": "Ground truth"},
                    path_to_save = f"../Figures/{experiment_name}/{model_name}_{data_file}_reg_var_{query_key}.png",
                    show = False,
                    legend = False,
                )
                print(f'{model_name}:')
                print(f'{query_key}: r2mean_all = {r2mean_all}, r2mean_top100 = {r2mean_top100}')
                print(f'{query_key}: r2var_all = {r2var_all}, r2var_top100 = {r2var_top100}')
                adata_query_eval = ad.concat([adata_query_ctrl, adata_query_eval])
                sc.pl.violin(adata_query_eval, keys = diff_genes[0], groupby = cond_key, rotation = 90, show = False)
                plt.savefig(f'../Figures/{experiment_name}/{model_name}_{data_file}_diff_genes[0]_violin_{query_key}.png', dpi = 200, bbox_inches = 'tight')
                plt.close()
                adata_query_eval.write_h5ad(f'../Result_anndata/{experiment_name}/{experiment_name}_{model_name}_{data_file}_{query_key}{file_type}')
if __name__ == '__main__':
    predict_perturbation()