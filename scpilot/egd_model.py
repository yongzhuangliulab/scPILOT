from typing import Optional, Sequence, Literal
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import ot
from adjustText import adjust_text
from anndata import AnnData
from matplotlib import pyplot as plt
from scipy import stats
from scipy.spatial.distance import cdist
from scvi import REGISTRY_KEYS
from scvi.data import AnnDataManager
from scvi.data.fields import CategoricalObsField, LayerField
from scvi.model.base import BaseModelClass, UnsupervisedTrainingMixin, VAEMixin
from .egd_network import EGD_network
from .egd_training_plan import EGDTrainingPlan
def balancer(
    adata: AnnData,
    cell_label_key: str,
):
    class_names = np.unique(adata.obs[cell_label_key])
    class_sizes = {}
    for cls in class_names:
        class_sizes[cls] = adata[adata.obs[cell_label_key] == cls].shape[0]
    max_size = np.max(list(class_sizes.values()))
    index_all = []
    for cls in class_names:
        cls_bin = np.array(adata.obs[cell_label_key] == cls)
        index_cls = np.nonzero(cls_bin)[0]
        index_cls_r = index_cls[np.random.choice(len(index_cls), max_size)]
        index_all.append(index_cls_r)
    balanced_data = adata[np.concatenate(index_all)].copy()
    return balanced_data
def extractor(
    data: AnnData,
    cell_label: str,
    condition_key: str,
    cell_label_key: str,
    ctrl_key: str,
    stim_key: str
):
    cell_with_both_condition = data[data.obs[cell_label_key] == cell_label]
    condition_1 = data[
        (data.obs[cell_label_key] == cell_label) & (data.obs[condition_key] == ctrl_key)
    ]
    condition_2 = data[
        (data.obs[cell_label_key] == cell_label) & (data.obs[condition_key] == stim_key)
    ]
    training = data[
        ~(
            (data.obs[cell_label_key] == cell_label)
            & (data.obs[condition_key] == stim_key)
        )
    ]
    return [training, condition_1, condition_2, cell_with_both_condition]
def ot_gmm(
    mean1: np.ndarray,
    var1: np.ndarray,
    mean2: np.ndarray,
    var2: np.ndarray,
):
    dis_mtx = ot.dist(mean1, mean2) + ot.dist(np.sqrt(var1), np.sqrt(var2)) + 1e-6
    return ot.emd(np.ones(mean1.shape[0]) / mean1.shape[0], np.ones(mean2.shape[0]) / mean2.shape[0], dis_mtx, numItermax = 1e6)
class EGD_model(VAEMixin, UnsupervisedTrainingMixin, BaseModelClass):
    _training_plan_cls = EGDTrainingPlan
    def __init__(
        self,
        adata: AnnData,
        n_hidden: int = 800,
        n_latent: int = 100,
        n_layers: int = 2,
        dropout_rate: float = 0.2,
        **model_kwargs,
    ):
        super(EGD_model, self).__init__(adata)
        self.module = EGD_network(
            n_input=self.summary_stats.n_vars,
            n_hidden=n_hidden,
            n_latent=n_latent,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            **model_kwargs,
        )
        self._model_summary_string = (
            "EGD Model with the following params: \nn_hidden: {}, n_latent: {}, n_layers: {}, dropout_rate: "
            "{}"
        ).format(
            n_hidden,
            n_latent,
            n_layers,
            dropout_rate,
        )
        self.init_params_ = self._get_init_params(locals())
    def predict(
        self,
        ctrl_key=None,
        stim_key=None,
        adata_to_predict=None,
        celltype_to_predict=None,
        restrict_arithmetic_to="all",
    ):
        cell_label_key = self.adata_manager.get_state_registry(
            REGISTRY_KEYS.LABELS_KEY
        ).original_key
        condition_key = self.adata_manager.get_state_registry(
            REGISTRY_KEYS.BATCH_KEY
        ).original_key
        if restrict_arithmetic_to == "all":
            ctrl_x = self.adata[self.adata.obs[condition_key] == ctrl_key, :]
            stim_x = self.adata[self.adata.obs[condition_key] == stim_key, :]
            ctrl_x = balancer(ctrl_x, cell_label_key)
            stim_x = balancer(stim_x, cell_label_key)
        else:
            key = list(restrict_arithmetic_to.keys())[0]
            values = restrict_arithmetic_to[key]
            subset = self.adata[self.adata.obs[key].isin(values)]
            ctrl_x = subset[subset.obs[condition_key] == ctrl_key, :]
            stim_x = subset[subset.obs[condition_key] == stim_key, :]
            if len(values) > 1:
                ctrl_x = balancer(ctrl_x, cell_label_key)
                stim_x = balancer(stim_x, cell_label_key)
        if celltype_to_predict is not None and adata_to_predict is not None:
            raise Exception("Please provide either a cell type or adata not both!")
        if celltype_to_predict is None and adata_to_predict is None:
            raise Exception("Please provide a cell type name or adata for your unperturbed cells")
        if celltype_to_predict is not None:
            ctrl_pred = extractor(
                self.adata,
                celltype_to_predict,
                condition_key,
                cell_label_key,
                ctrl_key,
                stim_key
            )[1]
        else:
            ctrl_pred = adata_to_predict
        eq = min(ctrl_x.X.shape[0], stim_x.X.shape[0])
        ctrl_ind = np.random.choice(range(ctrl_x.shape[0]), size=eq, replace=False)
        stim_ind = np.random.choice(range(stim_x.shape[0]), size=eq, replace=False)
        ctrl_adata = ctrl_x[ctrl_ind, :]
        stim_adata = stim_x[stim_ind, :]
        latent_ctrl_avg = self._avg_vector(ctrl_adata)
        latent_stim_avg = self._avg_vector(stim_adata)
        delta = latent_stim_avg - latent_ctrl_avg
        ctrl_pred_z = self.get_latent_representation(ctrl_pred)
        stim_pred_z = delta + ctrl_pred_z
        stim_pred_X = (
            self.module.generative(torch.Tensor(stim_pred_z))["px"].cpu().detach().numpy()
        )
        stim_pred = AnnData(
            X=stim_pred_X,
            obs=ctrl_pred.obs.copy(),
            var=ctrl_pred.var.copy(),
            obsm=ctrl_pred.obsm.copy()
        )
        return stim_pred, delta
    def predict_new(
        self,
        ctrl_key = None,
        stim_key = None,
        query_key = None,
    ):
        cell_label_key = self.adata_manager.get_state_registry(
            REGISTRY_KEYS.LABELS_KEY
        ).original_key
        condition_key = self.adata_manager.get_state_registry(
            REGISTRY_KEYS.BATCH_KEY
        ).original_key
        adata_atlas_ctrl = self.adata[
            (self.adata.obs[cell_label_key] != query_key)
            & (self.adata.obs[condition_key] == ctrl_key),
            : ,
        ].copy()
        adata_atlas_stim = self.adata[
            (self.adata.obs[cell_label_key] != query_key)
            & (self.adata.obs[condition_key] == stim_key),
            : ,
        ].copy()
        adata_query_ctrl = self.adata[
            (self.adata.obs[cell_label_key] == query_key)
            & (self.adata.obs[condition_key] == ctrl_key),
            : ,
        ].copy()
        latent_m_query_ctrl, latent_v_query_ctrl = self.get_latent_representation(adata_query_ctrl, return_dist = True)
        latent_z_query_ctrl = self.get_latent_representation(adata_query_ctrl)
        latent_center_query_ctrl = self._avg_vector(adata_query_ctrl)
        class_names = np.unique(adata_atlas_ctrl.obs[cell_label_key])
        delta_query_list = []
        latent_center_atlas_cls_list = []
        for cls in class_names:
            adata_atlas_cls_ctrl = adata_atlas_ctrl[adata_atlas_ctrl.obs[cell_label_key] == cls, : ].copy()
            adata_atlas_cls_stim_original = adata_atlas_stim[adata_atlas_stim.obs[cell_label_key] == cls, : ].copy()
            if adata_atlas_cls_stim_original.shape[0] > 0:
                index_atlas_cls_stim_original = np.random.choice(range(adata_atlas_cls_stim_original.shape[0]), size = adata_atlas_cls_ctrl.shape[0])
                adata_atlas_cls_stim = adata_atlas_cls_stim_original[index_atlas_cls_stim_original, : ].copy()
                latent_m_atlas_cls_ctrl, latent_v_atlas_cls_ctrl = self.get_latent_representation(adata_atlas_cls_ctrl, return_dist = True)
                latent_z_atlas_cls_ctrl = self.get_latent_representation(adata_atlas_cls_ctrl)
                latent_m_atlas_cls_stim, latent_v_atlas_cls_stim = self.get_latent_representation(adata_atlas_cls_stim, return_dist = True)
                latent_z_atlas_cls_stim = self.get_latent_representation(adata_atlas_cls_stim)
                OT_atlas_cls = ot_gmm(latent_m_atlas_cls_ctrl, latent_v_atlas_cls_ctrl, latent_m_atlas_cls_stim, latent_v_atlas_cls_stim)
                delta_atlas_cls = (OT_atlas_cls / np.sum(OT_atlas_cls, axis = 1)[: , None]) @ latent_z_atlas_cls_stim - latent_z_atlas_cls_ctrl
                index_atlas_cls_ctrl = np.random.choice(range(adata_atlas_cls_ctrl.shape[0]), size = adata_query_ctrl.shape[0])
                OT_query2atlas_cls = ot_gmm(
                    latent_m_query_ctrl,
                    latent_v_query_ctrl,
                    latent_m_atlas_cls_ctrl[index_atlas_cls_ctrl, : ].copy(),
                    latent_v_atlas_cls_ctrl[index_atlas_cls_ctrl, : ].copy(),
                )
                delta_query_list.append((OT_query2atlas_cls / np.sum(OT_query2atlas_cls, axis = 1)[: , None]) @ delta_atlas_cls[index_atlas_cls_ctrl, : ].copy())
                latent_center_atlas_cls_list.append(self._avg_vector(adata_atlas_cls_ctrl))
        latent_center_atlas_cls_mtx = np.array(latent_center_atlas_cls_list)
        latent_center_dist = cdist(latent_center_query_ctrl[None, : ], latent_center_atlas_cls_mtx).reshape(-1)
        delta_query_weight = (np.exp(-latent_center_dist) / np.sum(np.exp(-latent_center_dist))).tolist()
        delta_query = np.zeros(delta_query_list[0].shape)
        for i in range(len(delta_query_list)):
            delta_query = delta_query + delta_query_list[i] * delta_query_weight[i]
        latent_z_query_pred = latent_z_query_ctrl + delta_query
        adata_X_query_pred = (
            self.module.generative(torch.Tensor(latent_z_query_pred))["px"].cpu().detach().numpy()
        )
        adata_query_pred = AnnData(
            X = adata_X_query_pred,
            obs = adata_query_ctrl.obs.copy(),
            var = adata_query_ctrl.var.copy(),
            obsm = adata_query_ctrl.obsm.copy()
        )
        return adata_query_pred, delta_query
    def _avg_vector(self, adata) -> np.ndarray:
        return np.mean(self.get_latent_representation(adata), axis=0)
    @torch.no_grad()
    def get_generated_expression(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        if self.is_trained_ is False:
            raise RuntimeError("Please train the model first.")
        adata = self._validate_anndata(adata)
        scdl = self._make_data_loader(
            adata=adata, indices=indices, batch_size=batch_size
        )
        generated = []
        for tensors in scdl:
            _, generative_outputs = self.module(tensors, compute_loss=False)
            px = generative_outputs["px"].cpu()
            generated.append(px)
        return torch.cat(generated).numpy()
    # @torch.no_grad()
    # def batch_removal(self, adata: Optional[AnnData] = None) -> AnnData:
    #     adata = self._validate_anndata(adata)
    #     latent_all = self.get_latent_representation(adata)
    #     cell_label_key = self.adata_manager.get_state_registry(
    #         REGISTRY_KEYS.LABELS_KEY
    #     ).original_key
    #     batch_key = self.adata_manager.get_state_registry(
    #         REGISTRY_KEYS.BATCH_KEY
    #     ).original_key
    #     adata_latent = AnnData(latent_all)
    #     adata_latent.obs = adata.obs.copy(deep=True)
    #     unique_cell_types = np.unique(adata_latent.obs[cell_label_key])
    #     shared_ct = []
    #     not_shared_ct = []
    #     for cell_type in unique_cell_types:
    #         temp_cell = adata_latent[
    #             adata_latent.obs[cell_label_key] == cell_type
    #         ].copy()
    #         if len(np.unique(temp_cell.obs[batch_key])) < 2:
    #             cell_type_ann = adata_latent[
    #                 adata_latent.obs[cell_label_key] == cell_type
    #             ]
    #             not_shared_ct.append(cell_type_ann)
    #             continue
    #         temp_cell = adata_latent[
    #             adata_latent.obs[cell_label_key] == cell_type
    #         ].copy()
    #         batch_list: dict[any, AnnData] = {}
    #         batch_ind: dict[any, pd.Series] = {}
    #         max_batch = 0
    #         max_batch_ind = ""
    #         batches = np.unique(temp_cell.obs[batch_key])
    #         for i in batches:
    #             temp = temp_cell[temp_cell.obs[batch_key] == i]
    #             temp_ind = temp_cell.obs[batch_key] == i # pd.Series(One-dimensional ndarray with axis labels (including time series).)
    #             if max_batch < len(temp):
    #                 max_batch = len(temp)
    #                 max_batch_ind = i
    #             batch_list[i] = temp
    #             batch_ind[i] = temp_ind
    #         max_batch_ann = batch_list[max_batch_ind]
    #         for study in batch_list:
    #             delta = np.average(max_batch_ann.X, axis=0) - np.average(
    #                 batch_list[study].X, axis=0
    #             )
    #             batch_list[study].X = delta + batch_list[study].X
    #             temp_cell[batch_ind[study]].X = batch_list[study].X
    #         shared_ct.append(temp_cell)
    #     all_shared_ann = AnnData.concatenate(
    #         *shared_ct, batch_key="concat_batch", index_unique=None
    #     )
    #     if "concat_batch" in all_shared_ann.obs.columns:
    #         del all_shared_ann.obs["concat_batch"]
    #     if len(not_shared_ct) < 1:
    #         corrected = AnnData(
    #             self.module.generative(torch.Tensor(all_shared_ann.X))["px"].cpu().numpy(),
    #             obs=all_shared_ann.obs,
    #         )
    #         corrected.var_names = adata.var_names.tolist()
    #         corrected = corrected[adata.obs_names]
    #         if adata.raw is not None:
    #             adata_raw = AnnData(X=adata.raw.X, var=adata.raw.var)
    #             adata_raw.obs_names = adata.obs_names
    #             corrected.raw = adata_raw
    #         corrected.obsm["latent"] = all_shared_ann.X
    #         corrected.obsm["corrected_latent"] = self.get_latent_representation(
    #             corrected
    #         )
    #         return corrected
    #     else:
    #         all_not_shared_ann = AnnData.concatenate(
    #             *not_shared_ct, batch_key="concat_batch", index_unique=None
    #         )
    #         all_corrected_data = AnnData.concatenate(
    #             all_shared_ann,
    #             all_not_shared_ann,
    #             batch_key="concat_batch",
    #             index_unique=None,
    #         )
    #         if "concat_batch" in all_shared_ann.obs.columns:
    #             del all_corrected_data.obs["concat_batch"]
    #         corrected = AnnData(
    #             self.module.generative(torch.Tensor(all_corrected_data.X))["px"].cpu().numpy(),
    #             obs=all_corrected_data.obs,
    #         )
    #         corrected.var_names = adata.var_names.tolist()
    #         corrected = corrected[adata.obs_names]
    #         if adata.raw is not None:
    #             adata_raw = AnnData(X=adata.raw.X, var=adata.raw.var)
    #             adata_raw.obs_names = adata.obs_names
    #             corrected.raw = adata_raw
    #         corrected.obsm["latent"] = all_corrected_data.X
    #         corrected.obsm["corrected_latent"] = self.get_latent_representation(
    #             corrected
    #         )
    #         return corrected
    def reg_mean_plot(
        self,
        adata: AnnData,
        axis_keys: dict[Literal['x', 'y'], str],
        labels: dict[Literal['x', 'y'], str],
        path_to_save = "./reg_mean.pdf",
        save = True,
        gene_list: Optional[list] = None,
        show = False,
        top_100_genes = None,
        verbose = False,
        legend = True,
        title: Optional[str] = None,
        x_coeff = 0.30,
        y_coeff = 0.8,
        fontsize = 14,
        **kwargs,
    ):
        import seaborn as sns
        sns.set_theme(font = 'Times New Roman', font_scale = 2)
        condition_key = self.adata_manager.get_state_registry(
            REGISTRY_KEYS.BATCH_KEY
        ).original_key
        diff_genes = top_100_genes
        stim = adata[adata.obs[condition_key] == axis_keys["y"]]
        pred = adata[adata.obs[condition_key] == axis_keys["x"]]
        if diff_genes is not None:
            if hasattr(diff_genes, "tolist"):
                diff_genes = diff_genes.tolist()
            adata_diff = adata[: , diff_genes]
            stim_diff = adata_diff[adata_diff.obs[condition_key] == axis_keys["y"]]
            pred_diff = adata_diff[adata_diff.obs[condition_key] == axis_keys["x"]]
            x_diff: np.ndarray = np.asarray(np.mean(pred_diff.X.toarray(), axis=0)).ravel()
            y_diff: np.ndarray = np.asarray(np.mean(stim_diff.X.toarray(), axis=0)).ravel()
            _, _, r_value_diff, _, _ = stats.linregress(x_diff, y_diff)
            if verbose:
                print("Top 100 DEGs mean: ", r_value_diff**2)
        x: np.ndarray = np.asarray(np.mean(pred.X.toarray(), axis=0)).ravel()
        y: np.ndarray = np.asarray(np.mean(stim.X.toarray(), axis=0)).ravel()
        _, _, r_value, _, _ = stats.linregress(x, y)
        if verbose:
            print("All genes mean: ", r_value**2)
        df = pd.DataFrame({axis_keys["x"]: x, axis_keys["y"]: y})
        ax = sns.regplot(x=axis_keys["x"], y=axis_keys["y"], data=df)
        ax.tick_params(labelsize=fontsize)
        if "range" in kwargs:
            start, stop, step = kwargs.get("range")
            ax.set_xticks(np.arange(start, stop, step))
            ax.set_yticks(np.arange(start, stop, step))
        ax.set_xlabel(labels["x"], fontsize = fontsize)
        ax.set_ylabel(labels["y"], fontsize = fontsize)
        if gene_list is not None:
            texts = []
            for i in gene_list:
                j = adata.var_names.tolist().index(i)
                x_bar = x[j]
                y_bar = y[j]
                texts.append(plt.text(x_bar, y_bar, i, fontsize = 11, color = "black"))
                plt.plot(x_bar, y_bar, "o", color = "red", markersize = 5)
            adjust_text(
                texts,
                x=x,
                y=y,
                arrowprops = dict(arrowstyle = "->", color = "grey", lw = 0.5),
                force_static=(0.0, 0.0),
            )
        if legend:
            plt.legend(loc = "center left", bbox_to_anchor = (1, 0.5))
        if title is None:
            plt.title("", fontsize = fontsize)
        else:
            plt.title(title, fontsize = fontsize)
        ax.text(
            max(x) - max(x) * x_coeff,
            max(y) - y_coeff * max(y),
            r"$\mathrm{R^2_{\mathrm{\mathsf{all\ genes}}}}$= " + f"{r_value ** 2: .2f}",
            fontsize = kwargs.get("textsize", fontsize),
        )
        if diff_genes is not None:
            ax.text(
                max(x) - max(x) * x_coeff,
                max(y) - (y_coeff + 0.15) * max(y),
                r"$\mathrm{R^2_{\mathrm{\mathsf{top\ 100\ DEGs}}}}$= "
                + f"{r_value_diff ** 2: .2f}",
                fontsize = kwargs.get("textsize", fontsize),
            )
        if save:
            plt.savefig(f"{path_to_save}", dpi = 200, bbox_inches = "tight")
        if show:
            plt.show()
        plt.close()
        if diff_genes is not None:
            return r_value**2, r_value_diff**2
        else:
            return r_value**2
    def reg_var_plot(
        self,
        adata: AnnData,
        axis_keys: dict[Literal['x', 'y'], str],
        labels: dict[Literal['x', 'y'], str],
        path_to_save = "./reg_var.pdf",
        save = True,
        gene_list: Optional[list] = None,
        show = False,
        top_100_genes = None,
        verbose = False,
        legend = True,
        title: Optional[str] = None,
        x_coeff = 0.30,
        y_coeff = 0.8,
        fontsize = 14,
        **kwargs,
    ):
        import seaborn as sns
        sns.set_theme(font = 'Times New Roman', font_scale = 2)
        condition_key = self.adata_manager.get_state_registry(
            REGISTRY_KEYS.BATCH_KEY
        ).original_key
        diff_genes = top_100_genes
        stim = adata[adata.obs[condition_key] == axis_keys["y"]]
        pred = adata[adata.obs[condition_key] == axis_keys["x"]]
        if diff_genes is not None:
            if hasattr(diff_genes, "tolist"):
                diff_genes = diff_genes.tolist()
            adata_diff = adata[: , diff_genes]
            stim_diff = adata_diff[adata_diff.obs[condition_key] == axis_keys["y"]]
            pred_diff = adata_diff[adata_diff.obs[condition_key] == axis_keys["x"]]
            x_diff: np.ndarray = np.asarray(np.var(pred_diff.X.toarray(), axis=0)).ravel()
            y_diff: np.ndarray = np.asarray(np.var(stim_diff.X.toarray(), axis=0)).ravel()
            _, _, r_value_diff, _, _ = stats.linregress(x_diff, y_diff)
            if verbose:
                print("Top 100 DEGs var: ", r_value_diff**2)
        x: np.ndarray = np.asarray(np.var(pred.X.toarray(), axis=0)).ravel()
        y: np.ndarray = np.asarray(np.var(stim.X.toarray(), axis=0)).ravel()
        _, _, r_value, _, _ = stats.linregress(x, y)
        if verbose:
            print("All genes var: ", r_value**2)
        df = pd.DataFrame({axis_keys["x"]: x, axis_keys["y"]: y})
        ax = sns.regplot(x=axis_keys["x"], y=axis_keys["y"], data=df)
        ax.tick_params(labelsize=fontsize)
        if "range" in kwargs:
            start, stop, step = kwargs.get("range")
            ax.set_xticks(np.arange(start, stop, step))
            ax.set_yticks(np.arange(start, stop, step))
        ax.set_xlabel(labels["x"], fontsize = fontsize)
        ax.set_ylabel(labels["y"], fontsize = fontsize)
        if gene_list is not None:
            texts = []
            for i in gene_list:
                j = adata.var_names.tolist().index(i)
                x_bar = x[j]
                y_bar = y[j]
                texts.append(plt.text(x_bar, y_bar, i, fontsize = 11, color = "black"))
                plt.plot(x_bar, y_bar, "o", color = "red", markersize = 5)
            adjust_text(
                texts,
                x=x,
                y=y,
                arrowprops = dict(arrowstyle = "->", color = "grey", lw = 0.5),
                force_static=(0.0, 0.0),
            )
        if legend:
            plt.legend(loc = "center left", bbox_to_anchor = (1, 0.5))
        if title is None:
            plt.title("", fontsize = fontsize)
        else:
            plt.title(title, fontsize = fontsize)
        ax.text(
            max(x) - max(x) * x_coeff,
            max(y) - y_coeff * max(y),
            r"$\mathrm{R^2_{\mathrm{\mathsf{all\ genes}}}}$= " + f"{r_value ** 2: .2f}",
            fontsize = kwargs.get("textsize", fontsize),
        )
        if diff_genes is not None:
            ax.text(
                max(x) - max(x) * x_coeff,
                max(y) - (y_coeff + 0.15) * max(y),
                r"$\mathrm{R^2_{\mathrm{\mathsf{top\ 100\ DEGs}}}}$= "
                + f"{r_value_diff ** 2: .2f}",
                fontsize = kwargs.get("textsize", fontsize),
            )
        if save:
            plt.savefig(f"{path_to_save}", dpi = 200, bbox_inches = "tight")
        if show:
            plt.show()
        plt.close()
        if diff_genes is not None:
            return r_value**2, r_value_diff**2
        else:
            return r_value**2
    # def binary_classifier(
    #     self,
    #     adata,
    #     delta,
    #     ctrl_key,
    #     stim_key,
    #     path_to_save,
    #     save=True,
    #     fontsize=14,
    # ):
    #     plt.close("all")
    #     adata = self._validate_anndata(adata)
    #     condition_key = self.adata_manager.get_state_registry(
    #         REGISTRY_KEYS.BATCH_KEY
    #     ).original_key
    #     ctrl = adata[adata.obs[condition_key] == ctrl_key, :]
    #     stim = adata[adata.obs[condition_key] == stim_key, :]
    #     all_latent_ctrl = self.get_latent_representation(ctrl.X)
    #     all_latent_stim = self.get_latent_representation(stim.X)
    #     dot_ctrl = np.zeros((len(all_latent_ctrl)))
    #     dot_stim = np.zeros((len(all_latent_stim)))
    #     for ind, vec in enumerate(all_latent_ctrl):
    #         dot_ctrl[ind] = np.dot(delta, vec)
    #     for ind, vec in enumerate(all_latent_stim):
    #         dot_stim[ind] = np.dot(delta, vec)
    #     plt.hist(dot_ctrl, label=ctrl_key, bins=50)
    #     plt.hist(dot_stim, label=stim_key, bins=50)
    #     # plt.legend(loc=1, prop={'size': 7})
    #     plt.axvline(0, color="k", linestyle="dashed", linewidth=1)
    #     plt.title("  ", fontsize=fontsize)
    #     plt.xlabel("  ", fontsize=fontsize)
    #     plt.ylabel("  ", fontsize=fontsize)
    #     plt.xticks(fontsize=fontsize)
    #     plt.yticks(fontsize=fontsize)
    #     ax = plt.gca()
    #     ax.grid(False)
    #     if save:
    #         plt.savefig(f"{path_to_save}", bbox_inches="tight", dpi=100)
    #     # plt.show()

    @classmethod
    def setup_anndata(
        cls,
        adata: AnnData,
        batch_key: Optional[str] = None,
        labels_key: Optional[str] = None,
        **kwargs,
    ):
        setup_method_args = cls._get_setup_method_args(**locals())
        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, None, is_count_data=False),
            CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key),
            CategoricalObsField(REGISTRY_KEYS.LABELS_KEY, labels_key),
        ]
        adata_manager = AnnDataManager(
            fields=anndata_fields, setup_method_args=setup_method_args
        )
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)