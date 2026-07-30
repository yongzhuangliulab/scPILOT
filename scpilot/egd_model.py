import torch
import ot
import os
import inspect
import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from typing import Optional, Literal, Union, Sequence
from torch.utils.data import DataLoader
from adjustText import adjust_text
from anndata import AnnData
from scipy import stats
from scipy.spatial.distance import cdist
from matplotlib import pyplot as plt
from .egd_network import EGD_network
from .egd_training_plan import EGDTrainingPlan
from .ann_data_splitter import AnnDataSplitter, AnnDataset
from .egd_train_runner import  EGDTrainRunner
def save_figure_jpg_pdf(path_to_save: str, dpi: int = 300, bbox_inches: str = 'tight'):
    """
    Save the current matplotlib figure as both JPG and PDF.

    The input path can end with .jpg, .jpeg, .pdf, or any suffix.
    The function will save both:
        <base>.jpg
        <base>.pdf
    """
    root, _ = os.path.splitext(path_to_save)
    jpg_path = root + '.jpg'
    pdf_path = root + '.pdf'

    plt.savefig(jpg_path, dpi=dpi, bbox_inches=bbox_inches)
    plt.savefig(pdf_path, bbox_inches=bbox_inches)
def balancer(
    adata: AnnData,
    cell_label_key: str,
) -> AnnData:
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
    adata: AnnData,
    cell_label_key: str,
    cell_label: str,
    cond_key: str,
    ctrl_key: str,
    stim_key: str
) -> tuple[AnnData, AnnData, AnnData, AnnData]:
    cells_with_both_conditions = adata[adata.obs[cell_label_key] == cell_label]
    cond_1 = adata[
        (adata.obs[cell_label_key] == cell_label) & (adata.obs[cond_key] == ctrl_key)
    ]
    cond_2 = adata[
        (adata.obs[cell_label_key] == cell_label) & (adata.obs[cond_key] == stim_key)
    ]
    training = adata[
        ~(
            (adata.obs[cell_label_key] == cell_label)
            & (adata.obs[cond_key] == stim_key)
        )
    ]
    return training, cond_1, cond_2, cells_with_both_conditions
def ot_naive(
    z1: np.ndarray,
    z2: np.ndarray,
    numItermax: int = 1000000,
):
    dis_mtx = ot.dist(z1, z2) + 1e-6
    a = ot.utils.unif(z1.shape[0])
    b = ot.utils.unif(z2.shape[0])

    return ot.emd(
        a, b, dis_mtx, numItermax=numItermax
    ), ot.emd2(
        a, b, dis_mtx, numItermax=numItermax
    )
class EGD_model:
    def to_device(self, device: Union[str, int]):
        self.module.to(torch.device(device))
    def _get_init_params(self, locals):
        parameters = inspect.signature(self.__init__).parameters.values()
        init_params = [p.name for p in parameters]
        all_params = {p: locals[p] for p in locals if p in init_params and not isinstance(locals[p], AnnData)}
        non_var_params = [p.name for p in parameters if p.kind != p.VAR_KEYWORD]
        non_var_params = {k: v for k, v in all_params.items() if k in non_var_params}
        var_params = [p.name for p in parameters if p.kind == p.VAR_KEYWORD]
        var_params = {k: v for k, v in all_params.items() if k in var_params}
        user_params = {'kwargs': var_params, 'non_kwargs': non_var_params}
        return user_params
    def __init__(
        self,
        adata: AnnData,
        n_latent: int = 100,
        n_layers: int = 2,
        n_hidden: int = 800,
        dropout_rate: float = 0.2,
        **model_kwargs,
    ):
        if adata.is_view:
            raise ValueError('Please run `adata = adata.copy()`')
        adata.obs_names_make_unique()
        self.adata = adata
        self.n_latent = n_latent
        self.module = EGD_network(
            n_input = adata.n_vars,
            n_hidden = n_hidden,
            n_latent = n_latent,
            n_layers = n_layers,
            dropout_rate = dropout_rate,
            **model_kwargs,
        )
        self.history_: dict = None
        self.train_indices_ = None
        self.validation_indices_ = None
        self.test_indices_ = None
        self.is_trained_ = False
        self.trainer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to_device(self.device)
        self.init_params_ = self._get_init_params(locals())
    def train(
        self,
        max_epochs: int = 400,
        accelerator: str = 'auto',
        devices: Union[int, list[int], str] = 'auto',
        batch_size: int = 32,
        train_size: float = 0.9,
        validation_size: Optional[float] = None,
        shuffle_set_split: bool = True,
        early_stopping: bool = True,
        datasplitter_kwargs: Optional[dict] = None,
        plan_kwargs: Optional[dict] = None,
        **trainer_kwargs,
    ):
        datasplitter_kwargs = datasplitter_kwargs if datasplitter_kwargs is not None else {}
        data_module = AnnDataSplitter(
            self.adata,
            batch_size = batch_size,
            train_size = train_size,
            validation_size = validation_size,
            shuffle_set_split=shuffle_set_split,
            **datasplitter_kwargs,
        )
        plan_kwargs = plan_kwargs if plan_kwargs is not None else {}
        training_plan = EGDTrainingPlan(self.module, **plan_kwargs)
        trainer_kwargs['early_stopping'] = early_stopping
        runner = EGDTrainRunner(
            self,
            training_plan = training_plan,
            data_splitter = data_module,
            max_epochs = max_epochs,
            accelerator = accelerator,
            devices = devices,
            **trainer_kwargs,
        )
        return runner()
    @torch.inference_mode()
    def get_latent_representation(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        batch_size: Optional[int] = 32,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        adata = adata if adata is not None else self.adata
        anndataloader = DataLoader(
            AnnDataset(adata)[indices if indices is not None else np.arange(adata.n_obs)],
            batch_size = batch_size,
        )
        z_list = []
        q_m_list = []
        q_v_list = []
        for x in anndataloader:
            q_m, q_v, z = self.module.encode(x)
            z_list.append(z.cpu())
            q_m_list.append(q_m.cpu())
            q_v_list.append(q_v.cpu())
        return torch.cat(q_m_list).numpy(), torch.cat(q_v_list).numpy(), torch.cat(z_list).numpy()
    def _avg_vector(self, adata) -> np.ndarray:
        return np.mean(self.get_latent_representation(adata)[2], axis=0)
    def predict(
        self,
        cell_label_key: str,
        cond_key: str,
        ctrl_key: str,
        stim_key: str,
        query_key: Optional[str] = None,
        adata_to_predict: Optional[AnnData] = None,
        restrict_arithmetic_to = 'all',
    ):
        if restrict_arithmetic_to == 'all':
            ctrl_x = self.adata[
                (self.adata.obs[cond_key] == ctrl_key)
                & (self.adata.obs[cell_label_key] != query_key)
            ]
            stim_x = self.adata[self.adata.obs[cond_key] == stim_key]
            ctrl_x = balancer(ctrl_x, cell_label_key)
            stim_x = balancer(stim_x, cell_label_key)
        else:
            key = list(restrict_arithmetic_to.keys())[0]
            values = restrict_arithmetic_to[key]
            subset = self.adata[self.adata.obs[key].isin(values)]
            ctrl_x = subset[subset.obs[cond_key] == ctrl_key]
            stim_x = subset[subset.obs[cond_key] == stim_key]
            if len(values) > 1:
                ctrl_x = balancer(ctrl_x, cell_label_key)
                stim_x = balancer(stim_x, cell_label_key)
        if query_key is not None and adata_to_predict is not None:
            raise Exception('Please provide either a cell type or adata not both!')
        if query_key is None and adata_to_predict is None:
            raise Exception('Please provide a cell type name or adata for your unperturbed cells')
        if query_key is not None:
            ctrl_pred = extractor(
                self.adata,
                cell_label_key,
                query_key,
                cond_key,
                ctrl_key,
                stim_key
            )[1]
        else:
            ctrl_pred = adata_to_predict
        eq = min(ctrl_x.X.shape[0], stim_x.X.shape[0])
        ctrl_ind = np.random.choice(range(ctrl_x.shape[0]), size=eq, replace=False)
        stim_ind = np.random.choice(range(stim_x.shape[0]), size=eq, replace=False)
        ctrl_adata = ctrl_x[ctrl_ind]
        stim_adata = stim_x[stim_ind]
        latent_ctrl_avg = self._avg_vector(ctrl_adata)
        latent_stim_avg = self._avg_vector(stim_adata)
        delta = latent_stim_avg - latent_ctrl_avg
        _, _, ctrl_pred_z = self.get_latent_representation(ctrl_pred)
        stim_pred_z = delta + ctrl_pred_z
        stim_pred_X = (
            self.module.generate(torch.Tensor(stim_pred_z)).cpu().detach().numpy()
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
        cell_label_key: str,
        cond_key: str,
        ctrl_key: str,
        stim_key: str,
        query_key: Optional[str] = None,
        adata_to_predict: Optional[AnnData] = None,
        sub_key: Optional[str] = None,
        cov_key: Optional[str] = None,
        cov_weight=1.0,
        use_adaptive_weighting: bool = True,
        use_leiden: bool = True,
        use_latent_ot: bool = True,
    ):
        adata_atlas_stim = self.adata[self.adata.obs[cond_key] == stim_key].copy()
        class_names = sorted(adata_atlas_stim.obs[cell_label_key].unique().tolist())
        adata_atlas_ctrl = self.adata[
            (self.adata.obs[cell_label_key].isin(class_names))
            & (self.adata.obs[cond_key] == ctrl_key),
        ].copy()
        if query_key is not None and adata_to_predict is not None:
            raise Exception('Please provide either a cell type or adata not both!')
        if query_key is None and adata_to_predict is None:
            raise Exception('Please provide a cell type name or adata for your unperturbed cells')
        if query_key is not None:
            adata_query_ctrl = self.adata[
                (self.adata.obs[cell_label_key] == query_key)
                & (self.adata.obs[cond_key] == ctrl_key),
            ].copy()
        else:
            adata_query_ctrl = adata_to_predict
        _, _, latent_z_query_ctrl = self.get_latent_representation(adata_query_ctrl)
        adata_query_ctrl.obsm['latent_z'] = latent_z_query_ctrl
        adata_atlas_ctrl_list = []
        for cls in class_names:
            adata_atlas_cls_ctrl = adata_atlas_ctrl[adata_atlas_ctrl.obs[cell_label_key] == cls].copy()
            adata_atlas_cls_stim = adata_atlas_stim[adata_atlas_stim.obs[cell_label_key] == cls].copy()
            if sub_key is None:
                _, _, latent_z_atlas_cls_ctrl = self.get_latent_representation(adata_atlas_cls_ctrl)
                _, _, latent_z_atlas_cls_stim = self.get_latent_representation(adata_atlas_cls_stim)
                adata_atlas_cls_ctrl.obsm['latent_z'] = latent_z_atlas_cls_ctrl
                if use_latent_ot:
                    OT_atlas_cls, _ = ot_naive(
                        latent_z_atlas_cls_ctrl,
                        latent_z_atlas_cls_stim,
                    )

                    delta_atlas_cls = (
                        OT_atlas_cls
                        / (
                            np.sum(OT_atlas_cls, axis=1)[:, None]
                            + 1e-12
                        )
                    ) @ latent_z_atlas_cls_stim - latent_z_atlas_cls_ctrl

                else:
                    # Ablation: replace cell-level latent OT with
                    # one mean latent shift for the reference cell type.
                    mean_delta_atlas_cls = (
                        np.mean(
                            latent_z_atlas_cls_stim,
                            axis=0,
                        )
                        - np.mean(
                            latent_z_atlas_cls_ctrl,
                            axis=0,
                        )
                    )

                    delta_atlas_cls = np.repeat(
                        mean_delta_atlas_cls[None, :],
                        latent_z_atlas_cls_ctrl.shape[0],
                        axis=0,
                    )
                adata_atlas_cls_ctrl.obsm['delta'] = delta_atlas_cls
                adata_atlas_ctrl_list.append(adata_atlas_cls_ctrl)
            else:
                sub_classes = sorted(adata_atlas_cls_stim.obs[sub_key].unique().tolist())
                for sub_cls in sub_classes:
                    adata_atlas_cls_sub_ctrl = adata_atlas_cls_ctrl[
                        adata_atlas_cls_ctrl.obs[sub_key] == sub_cls
                    ].copy()
                    adata_atlas_cls_sub_stim = adata_atlas_cls_stim[
                        adata_atlas_cls_stim.obs[sub_key] == sub_cls
                    ].copy()
                    _, _, latent_z_atlas_cls_sub_ctrl = self.get_latent_representation(
                        adata_atlas_cls_sub_ctrl
                    )
                    _, _, latent_z_atlas_cls_sub_stim = self.get_latent_representation(
                        adata_atlas_cls_sub_stim
                    )
                    adata_atlas_cls_sub_ctrl.obsm['latent_z'] = latent_z_atlas_cls_sub_ctrl
                    if use_latent_ot:
                        OT_atlas_cls_sub, _ = ot_naive(
                            latent_z_atlas_cls_sub_ctrl,
                            latent_z_atlas_cls_sub_stim,
                        )

                        delta_atlas_cls_sub = (
                            OT_atlas_cls_sub
                            / (
                                np.sum(OT_atlas_cls_sub, axis=1)[:, None]
                                + 1e-12
                            )
                        ) @ latent_z_atlas_cls_sub_stim - latent_z_atlas_cls_sub_ctrl

                    else:
                        # Ablation: one mean latent shift for this
                        # cell-type/subcategory reference group.
                        mean_delta_atlas_cls_sub = (
                            np.mean(
                                latent_z_atlas_cls_sub_stim,
                                axis=0,
                            )
                            - np.mean(
                                latent_z_atlas_cls_sub_ctrl,
                                axis=0,
                            )
                        )

                        delta_atlas_cls_sub = np.repeat(
                            mean_delta_atlas_cls_sub[None, :],
                            latent_z_atlas_cls_sub_ctrl.shape[0],
                            axis=0,
                        )
                    adata_atlas_cls_sub_ctrl.obsm['delta'] = delta_atlas_cls_sub
                    adata_atlas_ctrl_list.append(adata_atlas_cls_sub_ctrl)
        adata_atlas_ctrl = ad.concat(adata_atlas_ctrl_list)

        def transfer_delta_and_matching_loss(
            latent_z_query: np.ndarray,
            adata_reference_group: AnnData,
        ) -> tuple[np.ndarray, float]:
            """
            Transfer a reference perturbation to query cells and return
            the corresponding query-reference matching loss.

            With latent OT:
                - use the OT transport plan for cell-level transfer;
                - use the OT transport cost for adaptive weighting.

            Without latent OT:
                - broadcast the mean reference perturbation vector;
                - use squared centroid distance for adaptive weighting.
            """
            if adata_reference_group.n_obs == 0:
                raise ValueError(
                    'The reference group contains no cells.'
                )

            _, _, latent_z_reference = (
                self.get_latent_representation(
                    adata_reference_group
                )
            )

            reference_delta = np.asarray(
                adata_reference_group.obsm[
                    'delta'
                ]
            )

            if use_latent_ot:
                OT_query2reference, matching_loss = ot_naive(
                    latent_z_query,
                    latent_z_reference,
                )

                transported_delta = (
                    OT_query2reference
                    / (
                        np.sum(OT_query2reference, axis=1)[:, None]
                        + 1e-12
                    )
                ) @ reference_delta

            else:
                # Ablation: do not establish a cell-level OT plan.
                # Every query cell receives the mean perturbation
                # of the current reference group.
                mean_reference_delta = np.mean(
                    reference_delta,
                    axis=0,
                    keepdims=True,
                )

                transported_delta = np.repeat(
                    mean_reference_delta,
                    latent_z_query.shape[0],
                    axis=0,
                )

                query_centroid = np.mean(
                    latent_z_query,
                    axis=0,
                )

                reference_centroid = np.mean(
                    latent_z_reference,
                    axis=0,
                )

                # Non-OT matching loss retained for adaptive weighting.
                matching_loss = float(
                    np.sum(
                        (
                            query_centroid
                            - reference_centroid
                        ) ** 2
                    )
                )

            return transported_delta, matching_loss

        def aggregate_delta_by_cell_type(
            latent_z_query: np.ndarray,
            adata_reference_pool: AnnData,
            empty_group_message: str,
        ) -> np.ndarray:
            """
            Transfer perturbation vectors from reference cells to query cells.

            Reference cells are grouped only by cell_label_key.
            sub_key is ignored during query-reference matching.
            For a different-covariate pool, individual covariate identities
            are also ignored after the pool has been selected.
            """
            if adata_reference_pool.n_obs == 0:
                raise ValueError(empty_group_message)

            reference_cell_types = sorted(
                adata_reference_pool.obs[
                    cell_label_key
                ].unique().tolist()
            )

            transported_delta_list = []
            matching_loss_list = []

            for reference_cell_type in reference_cell_types:
                adata_reference_cell_type = (
                    adata_reference_pool[
                        adata_reference_pool.obs[
                            cell_label_key
                        ] == reference_cell_type
                    ].copy()
                )

                transported_delta, matching_loss = (
                    transfer_delta_and_matching_loss(
                        latent_z_query=latent_z_query,
                        adata_reference_group=(
                            adata_reference_cell_type
                        ),
                    )
                )

                transported_delta_list.append(
                    transported_delta
                )
                matching_loss_list.append(
                    matching_loss
                )

            n_reference_cell_types = len(
                transported_delta_list
            )

            if n_reference_cell_types == 0:
                raise ValueError(empty_group_message)

            if use_adaptive_weighting:
                matching_loss_mtx = np.asarray(
                    matching_loss_list,
                    dtype=np.float64,
                )

                reference_weight_list = (
                    np.exp(-matching_loss_mtx)
                    / np.sum(np.exp(-matching_loss_mtx))
                )
            else:
                reference_weight_list = np.full(
                    n_reference_cell_types,
                    1.0 / n_reference_cell_types,
                    dtype=np.float64,
                )

            delta_query = np.zeros_like(
                transported_delta_list[0]
            )

            for transported_delta, reference_weight in zip(
                transported_delta_list,
                reference_weight_list,
            ):
                delta_query += (
                    transported_delta
                    * reference_weight
                )

            return delta_query


        adata_query_pred_list = []

        if not use_leiden:
            if cov_key is None:
                # The complete query population is treated as one group.
                # Reference cells are grouped only by cell_label_key.
                delta_query_ctrl = aggregate_delta_by_cell_type(
                    latent_z_query=latent_z_query_ctrl,
                    adata_reference_pool=adata_atlas_ctrl,
                    empty_group_message=(
                        'No reference cell types are available for '
                        'the no-Leiden perturbation transfer.'
                    ),
                )

                adata_query_ctrl.obsm[
                    'delta'
                ] = delta_query_ctrl

                latent_z_query_pred = (
                    latent_z_query_ctrl
                    + delta_query_ctrl
                )

                adata_X_query_pred = (
                    self.module.generate(
                        torch.Tensor(
                            latent_z_query_pred
                        )
                    )
                    .cpu()
                    .detach()
                    .numpy()
                )

                adata_query_pred = ad.AnnData(
                    X=adata_X_query_pred,
                    obs=adata_query_ctrl.obs.copy(),
                    var=adata_query_ctrl.var.copy(),
                    obsm=adata_query_ctrl.obsm.copy(),
                )

                adata_query_pred_list.append(
                    adata_query_pred
                )

            else:
                query_cov_classes = sorted(
                    adata_query_ctrl.obs[
                        cov_key
                    ].unique().tolist()
                )

                for query_cov_cls in query_cov_classes:
                    # All query cells from one covariate class form one group.
                    # No query Leiden clustering is performed.
                    adata_query_cov_ctrl = (
                        adata_query_ctrl[
                            adata_query_ctrl.obs[
                                cov_key
                            ] == query_cov_cls
                        ].copy()
                    )

                    _, _, latent_z_query_cov_ctrl = (
                        self.get_latent_representation(
                            adata_query_cov_ctrl
                        )
                    )

                    delta_query_cov = np.zeros_like(
                        latent_z_query_cov_ctrl
                    )

                    # ---------------------------------------------------------
                    # Same-covariate reference pool
                    # ---------------------------------------------------------
                    # First select the same-cov pool, then group it only by
                    # cell_label_key. sub_key does not participate in matching.
                    adata_atlas_same_cov_ctrl = (
                        adata_atlas_ctrl[
                            adata_atlas_ctrl.obs[
                                cov_key
                            ] == query_cov_cls
                        ].copy()
                    )

                    delta_same_cov = (
                        aggregate_delta_by_cell_type(
                            latent_z_query=(
                                latent_z_query_cov_ctrl
                            ),
                            adata_reference_pool=(
                                adata_atlas_same_cov_ctrl
                            ),
                            empty_group_message=(
                                'No same-covariate reference '
                                'cell types are available for '
                                f'{cov_key}={query_cov_cls!r}.'
                            ),
                        )
                    )

                    delta_query_cov += (
                        delta_same_cov
                        * cov_weight
                    )

                    # ---------------------------------------------------------
                    # Different-covariate reference pool
                    # ---------------------------------------------------------
                    # Pool all covariates other than query_cov_cls, then group
                    # the pooled reference cells only by cell_label_key.
                    adata_atlas_diff_cov_ctrl = (
                        adata_atlas_ctrl[
                            adata_atlas_ctrl.obs[
                                cov_key
                            ] != query_cov_cls
                        ].copy()
                    )

                    delta_diff_cov = (
                        aggregate_delta_by_cell_type(
                            latent_z_query=(
                                latent_z_query_cov_ctrl
                            ),
                            adata_reference_pool=(
                                adata_atlas_diff_cov_ctrl
                            ),
                            empty_group_message=(
                                'No different-covariate '
                                'reference cell types are '
                                'available for '
                                f'{cov_key}={query_cov_cls!r}.'
                            ),
                        )
                    )

                    delta_query_cov += (
                        delta_diff_cov
                        * (1.0 - cov_weight)
                    )

                    adata_query_cov_ctrl.obsm[
                        'delta'
                    ] = delta_query_cov

                    latent_z_query_cov_pred = (
                        latent_z_query_cov_ctrl
                        + delta_query_cov
                    )

                    adata_X_query_cov_pred = (
                        self.module.generate(
                            torch.Tensor(
                                latent_z_query_cov_pred
                            )
                        )
                        .cpu()
                        .detach()
                        .numpy()
                    )

                    adata_query_cov_pred = ad.AnnData(
                        X=adata_X_query_cov_pred,
                        obs=(
                            adata_query_cov_ctrl
                            .obs
                            .copy()
                        ),
                        var=(
                            adata_query_cov_ctrl
                            .var
                            .copy()
                        ),
                        obsm=(
                            adata_query_cov_ctrl
                            .obsm
                            .copy()
                        ),
                    )

                    adata_query_pred_list.append(
                        adata_query_cov_pred
                    )

        elif cov_key is None:
            sc.pp.neighbors(
                adata_query_ctrl,
                use_rep='latent_z',
            )
            sc.tl.leiden(adata_query_ctrl)
            query_leidens = adata_query_ctrl.obs['leiden'].unique().tolist()
            sc.pp.neighbors(adata_atlas_ctrl, use_rep = 'latent_z')
            sc.tl.leiden(adata_atlas_ctrl)
            atlas_leidens = adata_atlas_ctrl.obs['leiden'].unique().tolist()
            for query_leiden in query_leidens:
                adata_query_leiden_ctrl = adata_query_ctrl[
                    adata_query_ctrl.obs['leiden'] == query_leiden
                ].copy()
                _, _, latent_z_query_leiden_ctrl = self.get_latent_representation(
                    adata_query_leiden_ctrl
                )
                delta_query_leiden_list = []
                OTloss_list = []
                for atlas_leiden in atlas_leidens:
                    adata_atlas_leiden_ctrl = adata_atlas_ctrl[
                        adata_atlas_ctrl.obs['leiden'] == atlas_leiden
                    ].copy()
                    transported_delta, matching_loss = (
                        transfer_delta_and_matching_loss(
                            latent_z_query=(
                                latent_z_query_leiden_ctrl
                            ),
                            adata_reference_group=(
                                adata_atlas_leiden_ctrl
                            ),
                        )
                    )

                    delta_query_leiden_list.append(
                        transported_delta
                    )

                    OTloss_list.append(
                        matching_loss
                    )
                n_atlas_leidens = len(delta_query_leiden_list)

                if n_atlas_leidens == 0:
                    raise ValueError(
                        'No atlas Leiden clusters are available for perturbation transfer.'
                    )

                if use_adaptive_weighting:
                    OTloss_mtx = np.asarray(
                        OTloss_list,
                        dtype=np.float64,
                    )

                    delta_query_leiden_weight_list = (
                        np.exp(-OTloss_mtx)
                        / np.sum(np.exp(-OTloss_mtx))
                    )
                else:
                    # Ablation: every atlas Leiden cluster contributes equally.
                    delta_query_leiden_weight_list = np.full(
                        n_atlas_leidens,
                        1.0 / n_atlas_leidens,
                        dtype=np.float64,
                    )

                delta_query_leiden = np.zeros_like(
                    delta_query_leiden_list[0],
                    dtype=np.float64,
                )

                for delta_i, weight_i in zip(
                    delta_query_leiden_list,
                    delta_query_leiden_weight_list,
                ):
                    delta_query_leiden += delta_i * weight_i
                adata_query_leiden_ctrl.obsm['delta'] = delta_query_leiden
                latent_z_query_leiden_pred = latent_z_query_leiden_ctrl + delta_query_leiden
                adata_X_query_leiden_pred = (
                    self.module.generate(torch.Tensor(latent_z_query_leiden_pred)).cpu().detach().numpy()
                )
                adata_query_leiden_pred = ad.AnnData(
                    X = adata_X_query_leiden_pred,
                    obs = adata_query_leiden_ctrl.obs.copy(),
                    var = adata_query_leiden_ctrl.var.copy(),
                    obsm = adata_query_leiden_ctrl.obsm.copy(),
                )
                adata_query_pred_list.append(adata_query_leiden_pred)
        else:
            query_cov_classes = sorted(adata_query_ctrl.obs[cov_key].unique().tolist())
            for query_cov_cls in query_cov_classes:
                adata_atlas_same_cov_ctrl = adata_atlas_ctrl[
                    adata_atlas_ctrl.obs[cov_key] == query_cov_cls
                ].copy()
                sc.pp.neighbors(adata_atlas_same_cov_ctrl, use_rep = 'latent_z')
                sc.tl.leiden(adata_atlas_same_cov_ctrl)
                atlas_same_cov_leidens = adata_atlas_same_cov_ctrl.obs['leiden'].unique().tolist()
                adata_atlas_diff_cov_ctrl = adata_atlas_ctrl[
                    adata_atlas_ctrl.obs[cov_key] != query_cov_cls
                ].copy()
                sc.pp.neighbors(adata_atlas_diff_cov_ctrl, use_rep = 'latent_z')
                sc.tl.leiden(adata_atlas_diff_cov_ctrl)
                atlas_diff_cov_leidens = adata_atlas_diff_cov_ctrl.obs['leiden'].unique().tolist()
                adata_query_cov_ctrl = adata_query_ctrl[
                    adata_query_ctrl.obs[cov_key] == query_cov_cls
                ].copy()
                sc.pp.neighbors(adata_query_cov_ctrl, use_rep = 'latent_z')
                sc.tl.leiden(adata_query_cov_ctrl)
                query_cov_leidens = adata_query_cov_ctrl.obs['leiden'].unique().tolist()
                for query_cov_leiden in query_cov_leidens:
                    adata_query_cov_leiden_ctrl = adata_query_cov_ctrl[
                        adata_query_cov_ctrl.obs['leiden'] == query_cov_leiden
                    ].copy()
                    _, _, latent_z_query_cov_leiden_ctrl = self.get_latent_representation(
                        adata_query_cov_leiden_ctrl
                    )
                    delta_query_cov_leiden_list = []
                    OTloss_list = []
                    for atlas_same_cov_leiden in atlas_same_cov_leidens:
                        adata_atlas_same_cov_leiden_ctrl = adata_atlas_same_cov_ctrl[
                            adata_atlas_same_cov_ctrl.obs['leiden'] == atlas_same_cov_leiden
                        ].copy()
                        transported_delta, matching_loss = (
                            transfer_delta_and_matching_loss(
                                latent_z_query=(
                                    latent_z_query_cov_leiden_ctrl
                                ),
                                adata_reference_group=(
                                    adata_atlas_same_cov_leiden_ctrl
                                ),
                            )
                        )

                        delta_query_cov_leiden_list.append(
                            transported_delta
                        )

                        OTloss_list.append(
                            matching_loss
                        )
                    n_same_cov_leidens = len(
                        delta_query_cov_leiden_list
                    )

                    if n_same_cov_leidens == 0:
                        raise ValueError(
                            'No same-covariate atlas Leiden clusters are available.'
                        )

                    if use_adaptive_weighting:
                        OTloss_mtx = np.asarray(
                            OTloss_list,
                            dtype=np.float64,
                        )

                        delta_query_cov_leiden_weight_list = (
                            np.exp(-OTloss_mtx)
                            / np.sum(np.exp(-OTloss_mtx))
                        )
                    else:
                        # Ablation: uniform weighting within the same-covariate group.
                        delta_query_cov_leiden_weight_list = np.full(
                            n_same_cov_leidens,
                            1.0 / n_same_cov_leidens,
                            dtype=np.float64,
                        )

                    delta_query_cov_leiden = np.zeros_like(
                        delta_query_cov_leiden_list[0],
                        dtype=np.float64,
                    )

                    for delta_i, weight_i in zip(
                        delta_query_cov_leiden_list,
                        delta_query_cov_leiden_weight_list,
                    ):
                        delta_query_cov_leiden += (
                            delta_i
                            * weight_i
                            * cov_weight
                        )
                    
                    delta_query_cov_leiden_list = []
                    OTloss_list = []
                    for atlas_diff_cov_leiden in atlas_diff_cov_leidens:
                        adata_atlas_diff_cov_leiden_ctrl = adata_atlas_diff_cov_ctrl[
                            adata_atlas_diff_cov_ctrl.obs['leiden'] == atlas_diff_cov_leiden
                        ].copy()
                        transported_delta, matching_loss = (
                            transfer_delta_and_matching_loss(
                                latent_z_query=(
                                    latent_z_query_cov_leiden_ctrl
                                ),
                                adata_reference_group=(
                                    adata_atlas_diff_cov_leiden_ctrl
                                ),
                            )
                        )

                        delta_query_cov_leiden_list.append(
                            transported_delta
                        )

                        OTloss_list.append(
                            matching_loss
                        )
                    n_diff_cov_leidens = len(
                        delta_query_cov_leiden_list
                    )

                    if n_diff_cov_leidens == 0:
                        raise ValueError(
                            'No different-covariate atlas Leiden clusters are available.'
                        )

                    if use_adaptive_weighting:
                        OTloss_mtx = np.asarray(
                            OTloss_list,
                            dtype=np.float64,
                        )

                        delta_query_cov_leiden_weight_list = (
                            np.exp(-OTloss_mtx)
                            / np.sum(np.exp(-OTloss_mtx))
                        )
                    else:
                        # Ablation: uniform weighting within the different-covariate group.
                        delta_query_cov_leiden_weight_list = np.full(
                            n_diff_cov_leidens,
                            1.0 / n_diff_cov_leidens,
                            dtype=np.float64,
                        )

                    for delta_i, weight_i in zip(
                        delta_query_cov_leiden_list,
                        delta_query_cov_leiden_weight_list,
                    ):
                        delta_query_cov_leiden += (
                            delta_i
                            * weight_i
                            * (1.0 - cov_weight)
                        )

                    adata_query_cov_leiden_ctrl.obsm['delta'] = delta_query_cov_leiden
                    latent_z_query_cov_leiden_pred = latent_z_query_cov_leiden_ctrl + delta_query_cov_leiden
                    adata_X_query_cov_leiden_pred = (
                        self.module.generate(torch.Tensor(latent_z_query_cov_leiden_pred)).cpu().detach().numpy()
                    )
                    adata_query_cov_leiden_pred = ad.AnnData(
                        X = adata_X_query_cov_leiden_pred,
                        obs = adata_query_cov_leiden_ctrl.obs.copy(),
                        var = adata_query_cov_leiden_ctrl.var.copy(),
                        obsm = adata_query_cov_leiden_ctrl.obsm.copy(),
                    )
                    adata_query_pred_list.append(adata_query_cov_leiden_pred)  
                    
        adata_query_pred = ad.concat(adata_query_pred_list)
        return adata_query_pred, adata_query_pred.obsm['delta']
    @staticmethod
    def reg_mean_plot(
        adata: AnnData,
        cond_key: str,
        axis_keys: dict[Literal['x', 'y'], str],
        labels: dict[Literal['x', 'y'], str],
        path_to_save = './reg_mean.pdf',
        save = True,
        gene_list: Optional[list] = None,
        show = False,
        top_genes = None,
        top_gene_label: str = 'T50',
        verbose = False,
        legend = True,
        title: Optional[str] = None,
        x_coeff = 0.30,
        y_coeff = 0.8,
        fontsize = 20,
        **kwargs,
    ):
        import seaborn as sns
        diff_genes = top_genes
        stim = adata[adata.obs[cond_key] == axis_keys['y']]
        pred = adata[adata.obs[cond_key] == axis_keys['x']]
        if diff_genes is not None:
            if hasattr(diff_genes, 'tolist'):
                diff_genes = diff_genes.tolist()
            adata_diff = adata[: , diff_genes]
            stim_diff = adata_diff[adata_diff.obs[cond_key] == axis_keys['y']]
            pred_diff = adata_diff[adata_diff.obs[cond_key] == axis_keys['x']]
            x_diff: np.ndarray = np.asarray(np.mean(pred_diff.X.toarray(), axis=0)).ravel()
            y_diff: np.ndarray = np.asarray(np.mean(stim_diff.X.toarray(), axis=0)).ravel()
            _, _, r_value_diff, _, _ = stats.linregress(x_diff, y_diff)
            if verbose:
                print(f'{top_gene_label} DEGs mean: ', r_value_diff**2)
        x: np.ndarray = np.asarray(np.mean(pred.X.toarray(), axis=0)).ravel()
        y: np.ndarray = np.asarray(np.mean(stim.X.toarray(), axis=0)).ravel()
        _, _, r_value, _, _ = stats.linregress(x, y)
        if verbose:
            print('All genes mean: ', r_value**2)
        df = pd.DataFrame({axis_keys['x']: x, axis_keys['y']: y})
        ax = sns.regplot(x = axis_keys['x'], y = axis_keys['y'], data = df)
        sns.despine(ax = ax, top = True, right = True)
        ax.tick_params(labelsize = fontsize)
        if 'range' in kwargs:
            start, stop, step = kwargs.get('range')
            ax.set_xticks(np.arange(start, stop, step))
            ax.set_yticks(np.arange(start, stop, step))
        ax.set_xlabel(labels['x'], fontsize = fontsize)
        ax.set_ylabel(labels['y'], fontsize = fontsize)
        if gene_list is not None:
            texts = []
            for i in gene_list:
                j = adata.var_names.tolist().index(i)
                x_bar = x[j]
                y_bar = y[j]
                texts.append(plt.text(x_bar, y_bar, i, fontsize = 14, color = 'black'))
                plt.plot(x_bar, y_bar, 'o', color = 'red', markersize = 5)
            adjust_text(
                texts,
                x=x,
                y=y,
                arrowprops = dict(arrowstyle = '->', color = 'grey', lw = 0.5),
                force_static=(0.0, 0.0),
            )
        if legend:
            plt.legend(loc = 'center left', bbox_to_anchor = (1, 0.5))
        if title is None:
            plt.title('', fontsize = fontsize)
        else:
            plt.title(title, fontsize = fontsize)
        ax.text(
            max(x) - max(x) * x_coeff,
            max(y) - y_coeff * max(y),
            r'$\mathrm{R^2_{\mathrm{\mathsf{mean}}}}$= ' + f'{r_value ** 2: .2f}',
            fontsize = kwargs.get('textsize', fontsize),
        )
        if diff_genes is not None:
            ax.text(
                max(x) - max(x) * x_coeff,
                max(y) - (y_coeff + 0.15) * max(y),
                r'$\mathrm{R^2_{\mathrm{\mathsf{mean,\ ' + top_gene_label + r'}}}}$= '
                + f'{r_value_diff ** 2: .2f}',
                fontsize = kwargs.get('textsize', fontsize),
            )
        if save:
            save_figure_jpg_pdf(path_to_save, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        plt.close()
        if diff_genes is not None:
            return r_value**2, r_value_diff**2
        else:
            return r_value**2
    @staticmethod
    def reg_var_plot(
        adata: AnnData,
        cond_key: str,
        axis_keys: dict[Literal['x', 'y'], str],
        labels: dict[Literal['x', 'y'], str],
        path_to_save = './reg_var.pdf',
        save = True,
        gene_list: Optional[list] = None,
        show = False,
        top_genes = None,
        top_gene_label: str = 'T50',
        verbose = False,
        legend = True,
        title: Optional[str] = None,
        x_coeff = 0.30,
        y_coeff = 0.8,
        fontsize = 20,
        **kwargs,
    ):
        import seaborn as sns
        diff_genes = top_genes
        stim = adata[adata.obs[cond_key] == axis_keys['y']]
        pred = adata[adata.obs[cond_key] == axis_keys['x']]
        if diff_genes is not None:
            if hasattr(diff_genes, 'tolist'):
                diff_genes = diff_genes.tolist()
            adata_diff = adata[: , diff_genes]
            stim_diff = adata_diff[adata_diff.obs[cond_key] == axis_keys['y']]
            pred_diff = adata_diff[adata_diff.obs[cond_key] == axis_keys['x']]
            x_diff: np.ndarray = np.asarray(np.var(pred_diff.X.toarray(), axis=0)).ravel()
            y_diff: np.ndarray = np.asarray(np.var(stim_diff.X.toarray(), axis=0)).ravel()
            _, _, r_value_diff, _, _ = stats.linregress(x_diff, y_diff)
            if verbose:
                print(f'{top_gene_label} DEGs var: ', r_value_diff**2)
        x: np.ndarray = np.asarray(np.var(pred.X.toarray(), axis=0)).ravel()
        y: np.ndarray = np.asarray(np.var(stim.X.toarray(), axis=0)).ravel()
        _, _, r_value, _, _ = stats.linregress(x, y)
        if verbose:
            print('All genes var: ', r_value**2)
        df = pd.DataFrame({axis_keys['x']: x, axis_keys['y']: y})
        ax = sns.regplot(x = axis_keys['x'], y = axis_keys['y'], data = df)
        sns.despine(ax = ax, top = True, right = True)
        ax.tick_params(labelsize = fontsize)
        if 'range' in kwargs:
            start, stop, step = kwargs.get('range')
            ax.set_xticks(np.arange(start, stop, step))
            ax.set_yticks(np.arange(start, stop, step))
        ax.set_xlabel(labels['x'], fontsize = fontsize)
        ax.set_ylabel(labels['y'], fontsize = fontsize)
        if gene_list is not None:
            texts = []
            for i in gene_list:
                j = adata.var_names.tolist().index(i)
                x_bar = x[j]
                y_bar = y[j]
                texts.append(plt.text(x_bar, y_bar, i, fontsize = 14, color = 'black'))
                plt.plot(x_bar, y_bar, 'o', color = 'red', markersize = 5)
            adjust_text(
                texts,
                x=x,
                y=y,
                arrowprops = dict(arrowstyle = '->', color = 'grey', lw = 0.5),
                force_static=(0.0, 0.0),
            )
        if legend:
            plt.legend(loc = 'center left', bbox_to_anchor = (1, 0.5))
        if title is None:
            plt.title('', fontsize = fontsize)
        else:
            plt.title(title, fontsize = fontsize)
        ax.text(
            max(x) - max(x) * x_coeff,
            max(y) - y_coeff * max(y),
            r'$\mathrm{R^2_{\mathrm{\mathsf{mean}}}}$= ' + f'{r_value ** 2: .2f}',
            fontsize = kwargs.get('textsize', fontsize),
        )
        if diff_genes is not None:
            ax.text(
                max(x) - max(x) * x_coeff,
                max(y) - (y_coeff + 0.15) * max(y),
                r'$\mathrm{R^2_{\mathrm{\mathsf{var,\ ' + top_gene_label + r'}}}}$= '
                + f'{r_value_diff ** 2: .2f}',
                fontsize = kwargs.get('textsize', fontsize),
            )
        if save:
            save_figure_jpg_pdf(path_to_save, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        plt.close()
        if diff_genes is not None:
            return r_value**2, r_value_diff**2
        else:
            return r_value**2
    def _get_user_attributes(self):
        attributes = inspect.getmembers(self, lambda a: not (inspect.isroutine(a)))
        return [a for a in attributes if not ((a[0].startswith('__') and a[0].endswith('__')) or a[0].startswith('_abc'))]
    def save(
        self,
        dir_path: str,
        prefix: Optional[str] = None,
        overwrite: bool = False,
        save_anndata: bool = False,
        save_kwargs: Optional[dict] = None,
        **anndata_write_kwargs,
    ):
        if not os.path.exists(dir_path) or overwrite:
            os.makedirs(dir_path, exist_ok = overwrite)
        else:
            raise ValueError(
                f'{dir_path} already exists. Please provide another directory for saving.'
            )
        file_name_prefix = prefix if prefix is not None else ''
        save_kwargs = save_kwargs if save_kwargs is not None else {}
        if save_anndata:
            file_suffix = 'adata.h5ad'
            self.adata.write(
                os.path.join(dir_path, f'{file_name_prefix}{file_suffix}'),
                **anndata_write_kwargs,
            )
        model_save_path = os.path.join(dir_path, f'{file_name_prefix}model.pt')
        model_state_dict = self.module.state_dict()
        var_names = self.adata.var_names.astype(str).to_numpy()
        user_attributes = self._get_user_attributes()
        user_attributes = {a[0]: a[1] for a in user_attributes if a[0][-1] == '_'}
        torch.save(
            {
                'model_state_dict': model_state_dict,
                'var_names': var_names,
                'attr_dict': user_attributes,
            },
            model_save_path,
            **save_kwargs,
        )
    @classmethod
    def load(
        cls,
        dir_path: str,
        adata: Optional[AnnData] = None,
        prefix: Optional[str] = None,
    ):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        file_name_prefix = prefix if prefix is not None else ''
        model_file_name = f'{file_name_prefix}model.pt'
        model_path = os.path.join(dir_path, model_file_name)
        model_dict = torch.load(model_path, map_location = device)
        model_state_dict = model_dict['model_state_dict']
        var_names = model_dict['var_names']
        attr_dict = model_dict['attr_dict']
        if adata is None:
            file_suffix = 'adata.h5ad'
            adata_path = os.path.join(dir_path, f'{file_name_prefix}{file_suffix}')
            if os.path.exists(adata_path):
                adata = ad.read_h5ad(adata_path)
            else:
                raise ValueError('Save path contains no saved anndata and no adata was passed.')
        if not np.array_equal(var_names, adata.var_names.astype(str)):
            raise ValueError('adata.var_names is different from saved var_names of adata used to train the model.')
        init_params = attr_dict.pop('init_params_')
        non_kwargs = init_params['non_kwargs']
        kwargs = init_params['kwargs']
        kwargs = {k: v for i, j in kwargs.items() for k, v in j.items()}
        model = cls(adata, **non_kwargs, **kwargs)
        for attr, val in attr_dict.items():
            setattr(model, attr, val)
        model.module.load_state_dict(model_state_dict)
        model.to_device(device)
        model.module.eval()
        return model 