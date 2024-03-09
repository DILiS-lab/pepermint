from typing import Optional
from pathlib import Path
import math

import numpy as np
import pandas as pd
from transformers import T5Tokenizer, T5EncoderModel
import torch
import re
from tqdm.auto import tqdm
from pyproteonet.data import Dataset
import pandas as pd

from datasets import load_breast_cancer_dataset, load_crohns_disease_dataset, load_prostate_cancer_dataset
from datasets import load_human_ecoli_mixture_dda_dia_dataset, load_blood_hiv_dda_dia_dataset
from datasets import load_maxlfq_benchmark_human_ecoli_mixture_dataset, load_crohns_disease_fibrosis_dataset

RANDOM_STATE = 424242 #to making creation of artificaial missing values by random masking reproducible

def create_masked_dataset(dataset: Dataset, masking_fraction: float = 0.1):
    # Incorparate missing values
    dataset = dataset.copy()
    vals = dataset.values['peptide']['abundance']
    dataset.values['peptide']['abundance_gt'] = vals
    vals[vals.dropna().sample(frac=masking_fraction, random_state=RANDOM_STATE).index] = np.nan
    dataset.values['peptide']['abundance'] = vals
    return dataset

def remove_all_missing(dataset: Dataset):
    # Restrict to molecules having at least some non-missing peptide abundance values
    df = dataset.get_wf('peptide', 'abundance')
    pep_mask = df[~df.isna().all(axis=1)].index
    mapping = dataset.mappings['peptide-protein'].df
    prot_mask = mapping[mapping.index.get_level_values('peptide').isin(pep_mask)].index.get_level_values('protein').unique()
    dataset = dataset.copy(molecule_ids={'protein': prot_mask, 'peptide': pep_mask}, copy_molecule_set=True)
    return dataset

def embed_sequences_t5(sequences: pd.Series, batch_size: int = 64):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # Load the tokenizer
    tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False, cache_dir='/hpi/fs00/home/tobias.pietz/fg_data/cache')
    # Load the model
    model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc", cache_dir='/hpi/fs00/home/tobias.pietz/fg_data/cache').to(device)
    # only GPUs support half-precision currently; if you want to run on CPU use full-precision (not recommended, much slower)
    if device==torch.device("cpu"):
        model.to(torch.float32) 
    #sequences = dataset.molecules[partner_molecule][partner_molecule_sequence_column]
    batches = [list(sequences.iloc[i:i+batch_size]) for i in range(0, len(sequences), batch_size)]
    results = []
    for batch in tqdm(batches):
        # replace all rare/ambiguous amino acids by X and introduce white-space between all amino acids
        seq_batch = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in batch]
        # tokenize sequences and pad up to the longest sequence in the batch
        ids = tokenizer(seq_batch, add_special_tokens=True, padding="longest")
        input_ids = torch.tensor(ids['input_ids']).to(device)
        attention_mask = torch.tensor(ids['attention_mask']).to(device)
        # generate embeddings
        with torch.no_grad():
            embedding_repr = model(input_ids=input_ids, attention_mask=attention_mask)
        embedding_repr = embedding_repr.last_hidden_state.to('cpu').numpy()
        for i in range(len(batch)):
            results.append(embedding_repr[i,:len(batch[i])].mean(axis=0))
    peptide_embeddings = pd.Series(results, index=sequences.index)
    return peptide_embeddings

def embed_sequences_t5_cached(sequences, sequence_embedding_cache_path: Path = Path('sequence_embedding_cache.h5'), batch_size: int = 64):
    # Generate sequence embeddings but storing and loading already generated embedding for a .h5 file
    cache = pd.DataFrame({'sequence':[], 'embedding':[]}).set_index('sequence')
    if sequence_embedding_cache_path.exists():
        cache = pd.read_hdf(sequence_embedding_cache_path, key='data')
    in_cache = sequences.isin(cache.index)
    res = pd.Series(index=sequences.index, dtype=object)
    res.loc[sequences[in_cache].index] = cache.loc[sequences[in_cache], 'embedding'].values
    if in_cache.all():
        return res
    embeddings = embed_sequences_t5(sequences=sequences[~in_cache], batch_size=batch_size)
    res.loc[embeddings.index] = embeddings.values
    sequ_embeddings = pd.DataFrame({'sequence':sequences, 'embedding':embeddings}).set_index('sequence')
    cache = pd.concat([cache, sequ_embeddings[~sequ_embeddings.index.isin(cache.index)]])
    if sequence_embedding_cache_path is not None:
        cache.to_hdf(sequence_embedding_cache_path, key='data')
    return res

def main():
    datasets = {}
    #datasets['breast_cancer'] = load_breast_cancer_dataset()
    datasets['breast_cancer_20'] = load_breast_cancer_dataset(num_sub_samples=20)
    datasets['breast_cancer_6'] = load_breast_cancer_dataset(num_sub_samples=6)
    # datasets['crohns_disease'] = load_crohns_disease_dataset()
    # datasets['crohns_fibrosis'] = load_crohns_disease_fibrosis_dataset()
    # datasets['prostate_cancer'] = load_prostate_cancer_dataset()
    # datasets['maxlfqbench'] = load_maxlfq_benchmark_human_ecoli_mixture_dataset()
    # datasets['blood_ddia'] = load_blood_hiv_dda_dia_dataset()
    # datasets['human_ecoli_ddia'] = load_human_ecoli_mixture_dda_dia_dataset()

    preprocesed = {}
    #Masked datasets
    for ds_name in ['breast_cancer', 'breast_cancer_20', 'breast_cancer_6', 'crohns_disease', 'crohns_fibrosis', 'prostate_cancer', 'maxlfqbench']:
        if ds_name not in datasets:
            continue
        print(ds_name)
        dataset = datasets[ds_name]
        dataset = create_masked_dataset(dataset=dataset, masking_fraction=0.1)
        dataset = remove_all_missing(dataset=dataset)
        sequences = dataset.molecules['peptide']['sequence']
        embeddings = embed_sequences_t5_cached(sequences=sequences, batch_size=64)
        dataset.molecules['peptide']['embedding'] = embeddings
        preprocesed[ds_name] = dataset

    #DDA/DIA datasets
    for ds_name in ['blood_ddia', 'human_ecoli_ddia']:
        if ds_name not in datasets:
            continue
        print(ds_name)
        dataset = datasets[ds_name]
        dataset = remove_all_missing(dataset=dataset)
        sequences = dataset.molecules['peptide']['sequence']
        embeddings = embed_sequences_t5_cached(sequences=sequences, batch_size=64)
        dataset.molecules['peptide']['embedding'] = embeddings
        preprocesed[ds_name] = dataset

    #Ratio dataset
    if 'maxlfqbench' in preprocesed:
        print('Normalizing maxlfqbench dataset')
        ds = preprocesed['maxlfqbench'].copy()
        reference_sample = list(ds.sample_names)[0]
        reference_ids = ds.molecules['peptide']
        reference_ids = reference_ids[reference_ids.is_human].index
        # values are already log-transformed, so we transform them back for the normalization
        values = math.e ** ds.get_column_flat(molecule='peptide', column='abundance', ids=reference_ids)
        factors = values.groupby("sample").sum()
        factors = factors[reference_sample] / factors
        for c in ['abundance', 'abundance_gt']:
            values = math.e ** ds.get_column_flat(molecule='peptide', column=c)
            values = values * values.index.get_level_values("sample").map(factors)
            ds.set_column_lf(molecule='peptide', column=c, values=np.log(values))
        preprocesed['maxlfqbench'] = ds

    for ds_name, ds in preprocesed.items():
        ds.save(f'data/datasets_preprocessed/{ds_name}', overwrite=True)

if __name__ == '__main__':
    main()
