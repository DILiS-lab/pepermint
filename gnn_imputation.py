from typing import Tuple, List
from functools import partial
import random

import numpy as np
import torch
torch.manual_seed(422442)
from torch import nn
from torch.utils.data import DataLoader
import dgl
from dgl.dataloading import GraphCollator
import lightning as L
from dgl.nn.pytorch.conv import GATv2Conv
from dgl.nn.pytorch import HeteroGraphConv
from dgl.dataloading import GraphCollator

from pyproteonet.data import Dataset
from pyproteonet.processing import Standardizer
from pyproteonet.masking import MaskedDataset, MaskedDatasetGenerator
from pyproteonet.lightning import ConsoleLogger
from pyproteonet.lightning.training_early_stopping import TrainingEarlyStopping


class ImputationModule(L.LightningModule):
    def __init__(self, num_samples: int, loss_fn: str = 'mse', skip_connection: bool = True, zero_sequence_embeddings: bool = False):
        super().__init__()
        lat_dim = 128
        dropout = 0.0
        heads = 64
        gat_dim = num_samples // 2
        # Fully connected layers to transform sequence embedding to a low dimension space (bottlenecked to 16 dimensions to prevent overfitting)
        self.emb_transform = nn.Sequential(nn.Dropout(0.0), nn.Linear(1024, 32), nn.LeakyReLU(), nn.Linear(32, 16))
        # Fully connected layers to transform concatenation of abundance vector (across samples) and sequence embedding
        self.pep_fully_connected = nn.Sequential(nn.Linear(num_samples + 16, lat_dim), nn.LeakyReLU(),
                                                 nn.Linear(lat_dim, lat_dim), nn.Dropout(dropout), nn.LeakyReLU(),
                                                 nn.Linear(lat_dim, lat_dim), nn.LeakyReLU(),
                                                 nn.Linear(lat_dim, lat_dim))
        # Graph attention layer to allow each peptide to attend to other peptides of the same protein
        self.pep_pep_gat = HeteroGraphConv({'peptide-peptide': GATv2Conv(in_feats=(lat_dim, lat_dim), out_feats=gat_dim, feat_drop=0, num_heads=heads,)})
        # Final fully connected layers transforming the new latent representation for each peptide
        if skip_connection:
            self.pep_final = nn.Sequential(nn.Linear(gat_dim * heads + lat_dim, lat_dim), nn.LeakyReLU(),
                                           nn.Linear(lat_dim, lat_dim), nn.Dropout(dropout), nn.LeakyReLU())
        else:
            self.pep_final = nn.Sequential(nn.Linear(gat_dim * heads, lat_dim), nn.LeakyReLU(),
                                           nn.Linear(lat_dim, lat_dim), nn.Dropout(dropout), nn.LeakyReLU())
        # Seperate head for prediction of the mean of every peptide value distributions
        self.pep_final_mean = nn.Sequential(nn.Linear(lat_dim, lat_dim),  nn.LeakyReLU(), nn.Linear(lat_dim, num_samples))
        # Seperate head for prediction of the variance/uncertainty of every peptide value
        self.pep_final_var = nn.Sequential(nn.Linear(lat_dim, lat_dim),  nn.LeakyReLU(), nn.Linear(lat_dim, num_samples))
        # value to use for missing and masked abundance values
        self.mask_value = -3
        self.loss_fn = loss_fn
        self.skip_connection = skip_connection
        self.zero_sequence_embeddings = zero_sequence_embeddings

    def forward(self, graph):
        # rplace missing, masked and hidden nodes in the training set with the mask_value
        for key in graph.ntypes:
            data = graph.nodes[key].data
            abundance = data['abundance']
            if 'hidden' in data:
                abundance[data['hidden']] = self.mask_value
            if 'mask' in data:
                abundance[data['mask']] = self.mask_value
            if 'hidden' in data:
                abundance[torch.isnan(abundance)] = self.mask_value
        # transform of initial abundance and sequence embedding vectors
        pep_in = graph.nodes['peptide'].data['abundance']
        pep_emb = graph.nodes['peptide'].data['embedding']
        if self.zero_sequence_embeddings:
            pep_emb = torch.zeros_like(pep_emb)
        pep_emb = self.emb_transform(pep_emb)
        pep_in = torch.cat([pep_in, pep_emb], dim=-1)
        pep_latent = self.pep_fully_connected(pep_in)
        # graph attention to attend to peptides of the same protein
        pep_vec = nn.functional.leaky_relu(self.pep_pep_gat(graph, ({'peptide':pep_latent}, {'peptide':pep_latent}))['peptide'])
        pep_vec = torch.cat(pep_vec.unbind(dim=-2), dim=-1)
        if self.skip_connection:
            pep_vec = torch.cat([pep_vec, pep_latent], dim=-1)
        # final transformation to predict mean and variance of peptide abundance
        pep_vec = self.pep_final(pep_vec)
        pep_mean = self.pep_final_mean(pep_vec)
        pep_var = self.pep_final_var(pep_vec)
        pep_var = torch.e**(pep_var)
        return pep_mean, pep_var

    def compute_loss(self, graph, prefix: str = 'train') -> torch.tensor:
        # we only compute the loss on masked nodes
        pep_mask = graph.nodes['peptide'].data['mask']
        # store abundance before masking (ground truth) for later loss computation
        pep_gt = graph.nodes['peptide'].data['abundance'][pep_mask].detach().clone()
        # forward pass
        pep_vec, pep_var = self(graph)
        # compute loss
        pep_pred = pep_vec[pep_mask]
        pep_var = pep_var[pep_mask]
        pep_loss = nn.functional.mse_loss(pep_pred, pep_gt)
        self.log(f"{prefix}_pep_mse_loss", pep_loss.item(), on_step=False, on_epoch=True, batch_size=1)
        self.log(f"{prefix}_mse_loss", (pep_loss).item(), on_step=False, on_epoch=True, batch_size=1)
        if self.loss_fn == "gnll":
            loss_fn = nn.GaussianNLLLoss()
            pep_loss = loss_fn(pep_pred, pep_gt, pep_var)
        self.log(f"{prefix}_pep_loss", pep_loss.item(), on_step=False, on_epoch=True, batch_size=1)
        loss = pep_loss
        return loss

    def training_step(self, graph, batch_idx):
        loss = self.compute_loss(graph, prefix='train')
        self.log("train_loss", loss.item(), on_step=False, on_epoch=True, batch_size=1)
        return loss
    
    def validation_step(self, graph, batch_idx):
        loss = self.compute_loss(graph, prefix='val')
        self.log("val_loss", loss.item(), on_step=False, on_epoch=True, batch_size=1)

    def predict_step(self, graph, batch_idx, dataloader_idx=0):
        pep_vec, pep_var = self(graph)
        return pep_vec, pep_var

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0005)
    

def impute_gnn(ds: Dataset, skip_connection: bool = True, random_edges: bool = False):
    """
    Imputes missing values in a dataset using Graph Neural Networks (GNN).

    Args:
        ds (Dataset): The dataset to be imputed.
        skip_connection (bool, optional): Whether to use skip connection skipping the GATv2 layer in the GNN model (for ablation study). Defaults to True.
        random_edges (bool, optional): Whether replace edges with random edges in the peptide-peptide graph (for ablation study). Defaults to False.

    Returns:
        None
    """
    # Function implementation goes here
    ...
def impute_gnn(ds: Dataset, skip_connection: bool = True, random_edges: bool = False, zero_sequence_embeddings: bool = False):
    
    # Create a copy of the dataset to use for GNN imputation
    gnn_ds = ds.copy(columns={'peptide':'abundance', 'protein':'abundance'}, copy_molecule_set=True)

    # Create peptide-peptide graph out of protein-peptide graph
    mapping = gnn_ds.mappings['peptide-protein'].df.copy()
    mapping['pep'] = mapping.index.get_level_values('peptide')
    mapping = mapping.merge(mapping, on='protein', suffixes=('tide_a', 'tide_b'))
    mapping = mapping[mapping['peptide_a'] < mapping['peptide_b']].drop_duplicates()
    if random_edges:
        print(f"Mapping shape before random rewiring: {mapping.shape}")
        mapping['peptide_a'] = mapping['peptide_a'].sample(frac=1, random_state=4243444342, replace=False).values
        mapping['peptide_b'] = mapping['peptide_b'].sample(frac=1, random_state=654321, replace=False).values
        mapping = mapping.drop_duplicates()
        print(f"Mapping shape after random rewiring: {mapping.shape}")
    mapping.set_index(['peptide_a', 'peptide_b'], inplace=True, drop=True)
    gnn_ds.molecule_set.add_mapping_pairs(name='peptide-peptide', pairs=mapping, mapping_molecules=('peptide', 'peptide'))
    standardizer = Standardizer(columns=['abundance'])
    gnn_ds = standardizer.standardize(gnn_ds)

    train_ds = gnn_ds.copy()
    # Create a validation set by masking 10% of the non-missing peptide values
    val_ds = gnn_ds.copy()
    val_ids = {}
    ids = ds.values['peptide']['abundance']
    ids = ids[~ids.isna()].sample(frac=0.1, random_state=123456).index
    val_ids['peptide'] = ids
    vals = train_ds.values['peptide']['abundance']
    vals.loc[val_ids['peptide']] = np.nan
    train_ds.values['peptide']['abundance'] = vals

    # For better reproducibility we use a random number generator for masking
    masking_rng = None
    # Defining our custom masking function realizing the self supervised learning approach
    def masking_fn(in_ds):
        # We use this trick to lazy initialize the random number generator for each worker
        nonlocal masking_rng
        if masking_rng is None:
            torch_seed = 424242 #only for debugging with 0 workers
            if torch.utils.data.get_worker_info() is not None:
                torch_seed = torch.utils.data.get_worker_info().seed
            masking_rng = np.random.default_rng(torch_seed)
        pep_ids = in_ds.values['peptide']["abundance"]
        non_missing_peps = pep_ids[~pep_ids.isna()]
        # sample around 10% of non-missing peptides to be masked during every training step
        # we slightly vary the masking fraction for each trainign step to make the training more diverse and robust
        frac = masking_rng.uniform(0.05, 0.15)
        pep_ids = non_missing_peps.sample(frac=frac, random_state=masking_rng).index
        return MaskedDataset.from_ids(dataset=in_ds, mask_ids={'peptide': pep_ids})

    # Create a masked dataset generator with 10 randomly masked datasets per epoch
    mask_ds = MaskedDatasetGenerator(datasets=[train_ds], generator_fn=masking_fn, epoch_size_multiplier=500)
    mask_val_ds = MaskedDataset.from_ids(dataset=val_ds, mask_ids={'peptide': val_ids['peptide']})

    collator = GraphCollator()
    collator_rng = None

    # We use a custom collate function to transform our datasets into dgl graphs
    def collate(masked_datasets_and_samples: List[Tuple[MaskedDataset, List[str]]], train: bool = True):
        nonlocal collator_rng
        assert len(masked_datasets_and_samples) == 1
        res = []
        for md, samples in masked_datasets_and_samples:
            # create a DGL graph from the masked dataset containing binary masks indicating the masked and hidden nodes
            graph = md.to_dgl_graph(
                feature_columns={
                    'peptide': ['abundance'],
                },
                molecule_columns = {'peptide': 'embedding'},
                mappings=['peptide-peptide'],
                make_bidirectional=True,
                samples=samples,
            )
            # subsample the graph to reduce GPU memory requirements while training
            if train:
                if collator_rng is None:
                    if torch.utils.data.get_worker_info() is not None:
                        seed = torch.utils.data.get_worker_info().seed + 356
                    else:
                        seed = 356
                    collator_rng = np.random.default_rng(seed=seed)
                subset_ids = graph.nodes('peptide').int()
                s = subset_ids.shape[0] * 0.5
                mask_ids = graph.nodes['peptide'].data['mask'].any(dim=1)
                subset_ids = subset_ids[mask_ids]
                if subset_ids.shape[0] >= s:
                    subset_ids = subset_ids[collator_rng.choice(subset_ids.shape[0], int(s), replace=False)]
                subset_ids = subset_ids.long()
                graph = dgl.node_subgraph(graph, {'peptide':subset_ids})
            res.append(graph)
        return collator.collate(res)
    
    model = ImputationModule(num_samples=ds.num_samples, loss_fn='mse', skip_connection=skip_connection, zero_sequence_embeddings=zero_sequence_embeddings)
    logger = ConsoleLogger()
    for loss_fn in ['mse', 'gnll']:
        model.loss_fn = loss_fn
        trainer = L.Trainer(
            logger=logger,
            log_every_n_steps=1,
            check_val_every_n_epoch=1,
            max_epochs=1000,
            enable_checkpointing=False,
            callbacks=[TrainingEarlyStopping(monitor="val_pep_mse_loss", mode="min", patience=3)], #Early stopping on validation MSE
        )
        num_workers = 20
        train_dl = DataLoader(mask_ds, batch_size=1, collate_fn=partial(collate, train=True), num_workers=num_workers)
        val_dl = DataLoader([(mask_val_ds, gnn_ds.sample_names)], batch_size=1, collate_fn=partial(collate, train=False), num_workers=0)
        trainer.fit(model=model, train_dataloaders=train_dl, val_dataloaders=val_dl)

    # Generate a impution dataset with missing values masked because those are the values we want to predict/impute
    missing_peps = ds.values['peptide']['abundance']
    missing_peps = missing_peps[missing_peps.isna()].index
    missing_mds = MaskedDataset.from_ids(dataset=gnn_ds, mask_ids={'peptide': missing_peps})
    pred_dl = DataLoader([(missing_mds, ds.sample_names)], batch_size=1, collate_fn=partial(collate, train=False))
    pep_pred, pep_var= trainer.predict(model=model, dataloaders=pred_dl)[0]

    # Write the imputed values back to the MaskedDataset and, therefore, the underlying gnn_ds dataset (We only write back to the masked/missing values)
    missing_mds.set_samples_value_matrix(matrix=pep_pred, molecule='peptide', column="abundance", only_set_masked=True)
    missing_mds.set_samples_value_matrix(matrix=pep_var  * standardizer.stds['peptide']['abundance'], molecule='peptide', column="var", only_set_masked=True)

    # Undo the standardization to tranform the imputed values back to the original scale
    res_ds = standardizer.unstandardize(gnn_ds)

    # Write the result to our original dataset
    ds.values['peptide']['gnn_imp'] = res_ds.values['peptide']['abundance']
    ds.values['peptide']['gnn_imp_var'] = res_ds.values['peptide']['var']