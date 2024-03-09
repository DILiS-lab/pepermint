from pathlib import Path
import random

import numpy as np
import pandas as pd
from pyproteonet.io import load_maxquant, read_multiple_mapped_dataframes, read_mapped_dataframe
from pyproteonet.processing import logarithmize
from pyproteonet.normalization import normalize_sum

def load_breast_cancer_dataset(path: Path = Path('data/datasets/PXD035857_breast_cancer'), num_sub_samples: int = 15):
    #load breast cancer dataset
    breast_cancer_ds = load_maxquant(peptides_table=path / 'peptides.txt',
                                     protein_groups_table= path /'proteinGroups.txt')
    #we work on a random subset to keep things fast and manageable
    random.seed(42)
    sample_subset = random.sample(breast_cancer_ds.sample_names, num_sub_samples)
    print('creating subset of dataset containing samples: ' + str(sample_subset))
    print('Loading breast cancer disease dataset with samples: ' + str(sample_subset))
    bc_ds = breast_cancer_ds.copy(samples=sample_subset)
    # Logarithmize the dataset
    bc_ds = logarithmize(bc_ds)
    # Rename molecules and columns to the standard names used for all datasets
    bc_ds.rename_molecule('protein_group', 'protein')
    bc_ds.rename_mapping('peptide-protein_group', 'peptide-protein')
    bc_ds.rename_columns(columns={'peptide':{'Intensity': 'abundance'}}, inplace=True)
    bc_ds.molecules['peptide'].rename(columns={'Sequence': 'sequence'}, inplace=True)
    return bc_ds

def load_crohns_disease_dataset(path = Path('data/datasets/PXD002882_crohns_disease')):
    #load crohns dataset
    peptide_df = pd.read_csv(path / 'peptides.txt', sep='\t')
    protein_df = pd.read_csv(path / 'proteinGroups.txt', sep='\t')
    #use majority protein IDs as protein IDs
    protein_df.rename(columns={'Majority protein IDs': 'protein_ids'}, inplace=True)
    #select samples to load
    samples = [c.replace('Intensity ', '') for c in peptide_df.columns if 'Intensity' in c]
    samples = [c for c in samples if not c.startswith(('L ', 'H '))][3:]
    print('Loading crohns disease dataset with samples: ' + str(samples))
    mask = ~(peptide_df[['Intensity ' + s for s in samples]] == 0).all(axis=1)
    peptide_df = peptide_df[mask]
    crohns_ds = load_maxquant(peptides_table=peptide_df, protein_groups_table=protein_df, samples=samples, protein_group_value_columns=[],
                              protein_group_columns=['protein_ids'])
    # Logarithmize the dataset
    crohns_ds = logarithmize(crohns_ds)
    # Rename molecules and columns to the standard names used for all datasets
    crohns_ds.rename_molecule('protein_group', 'protein')
    crohns_ds.rename_mapping(mapping='peptide-protein_group', new_name='peptide-protein')
    crohns_ds.rename_columns(columns={'peptide':{'Intensity': 'abundance'}}, inplace=True)
    crohns_ds.molecules['peptide'].rename(columns={'Sequence': 'sequence'}, inplace=True)
    return crohns_ds

def load_crohns_disease_fibrosis_dataset(path = Path('data/datasets/PXD022214_crohns_disease_fibrosis')):
    #load prostate cancer dataset
    crohns_ds = load_maxquant(peptides_table=path / 'peptides.txt', protein_groups_table= path /'proteinGroups.txt')
    crohns_ds = logarithmize(crohns_ds)
    # Rename molecules and columns for convenience
    crohns_ds.rename_molecule('protein_group', 'protein')
    crohns_ds.rename_mapping('peptide-protein_group', 'peptide-protein')
    crohns_ds.rename_columns(columns={'peptide':{'Intensity': 'abundance'}}, inplace=True)
    crohns_ds.molecules['peptide'].rename(columns={'Sequence': 'sequence'}, inplace=True)
    return crohns_ds

def load_prostate_cancer_dataset(path: Path = Path('data/datasets/PXD029525_prostate_cancer')):
    #load prostate cancer dataset
    prostate_ds = load_maxquant(peptides_table=path / 'peptides.txt',
                                     protein_groups_table= path /'proteinGroups.txt')
    prostate_ds = logarithmize(prostate_ds)
    # Rename molecules and columns for convenience
    prostate_ds.rename_molecule('protein_group', 'protein')
    prostate_ds.rename_mapping('peptide-protein_group', 'peptide-protein')
    prostate_ds.rename_columns(columns={'peptide':{'Intensity': 'abundance'}}, inplace=True)
    prostate_ds.molecules['peptide'].rename(columns={'Sequence': 'sequence'}, inplace=True)
    return prostate_ds

def load_maxlfq_benchmark_human_ecoli_mixture_dataset(path: Path = Path('data/datasets/PXD000279_maxlfq_benchmark_human_ecoli_mixture')):
    ds = load_maxquant(peptides_table=path/'peptides.txt', protein_groups_table=path/'proteinGroups.txt',
                       protein_group_value_columns=['LFQ intensity'],
                       samples=['L1','L2','L3','H1','H2','H3'], protein_group_columns=['Fasta headers', 'Protein IDs']
                      )
    ds.rename_columns(columns={'peptide':{'Intensity': 'abundance'}}, inplace=True)
    ds.rename_molecule('protein_group', 'protein')
    ds.rename_mapping('peptide-protein_group', 'peptide-protein')
    #Extract organism from fasta headers
    prots = ds.molecules['protein']
    ds.molecules['protein']['is_human'] = prots['Fasta headers'].str.contains('HUMAN')
    ds.molecules['protein']['is_ecoli'] = prots['Fasta headers'].str.contains('ECOLI')
    #Propagate the organism information to peptide level
    mapped = ds.molecule_set.get_mapped(molecule='protein', mapping='peptide-protein', molecule_columns='is_ecoli')
    grouped = mapped.groupby('peptide').is_ecoli.mean()
    #Only set organism for = peptides uniquely mapped to a single organism
    grouped = grouped[(grouped==0) | (grouped==1)]
    ds.molecules['peptide']['is_ecoli'] = False
    ds.molecules['peptide'].loc[grouped.index, 'is_ecoli'] = grouped.astype(bool)
    ds.molecules['peptide']['is_human'] = False
    ds.molecules['peptide'].loc[grouped.index, 'is_human'] = ~grouped.astype(bool)
    #Logarithmize the dataset
    ds = logarithmize(ds)
    ds.molecules['peptide'].rename(columns={'Sequence': 'sequence'}, inplace=True)
    return ds

def load_blood_hiv_dda_dia_dataset(path: Path = Path('data/datasets/blood_hiv_dda_dia')):
    sample_names = ['1_Slot1', '2_Slot1', '3_Slot1', '4_Slot1', '5_Slot1', '6_Slot1', '7_Slot1', '8_Slot1',
                    '9_Slot1', '10_Slot1','11_Slot1', '12_Slot1', '13_Slot1', '14_Slot1', '15_Slot1']    
    #Load DDA/DIA datasets
    dss = {}
    for kind in ['dda', 'dia']:  
        base_path = Path(path) / kind
        proteins = pd.read_csv(base_path / 'proteins.tsv', sep='\t').rename(columns={'ProteinID':'protein'})
        proteins['peptide-protein'] = proteins['protein']
        peptides = pd.read_csv(base_path / 'peptides.tsv', sep='\t').rename(columns={'ProteinID':'protein'})
        peptides['peptide-protein'] = peptides['protein']
        peptide_groups = peptides.groupby('Sequence')
        peptides = peptide_groups[sample_names].sum(min_count=1)
        peptides[['peptide-protein']] = peptide_groups[['peptide-protein']].first()
        peptides['Sequence'] = peptides.index
        ds = read_mapped_dataframe(df=peptides, sample_columns=sample_names, molecule='peptide', mapping_column='peptide-protein',
                                   result_column_name='abundance', molecule_columns=['Sequence'],
                                   mapping_molecule='protein', mapping_name='peptide-protein')
        #Only keep peptides with at least one non-missing value
        ids = ds.get_wf(molecule='peptide', column='abundance')
        ids = ids[~ids.isna().all(axis=1)].index
        ds = ds.copy(molecule_ids={'peptide':ids})
        vals = ds.values['protein']['abundance']
        vals[vals==0]=np.nan
        ds.values['protein']['abundance'] = vals
        vals = ds.values['peptide']['abundance']
        vals[vals==0]=np.nan
        ds.values['peptide']['abundance'] = vals
        dss[kind] = ds
    blood_dda = dss['dda']
    blood_dia = dss['dia']
    steen_pep_dia = blood_dia.values['peptide']['abundance']
    #Scale DIA abundance values to match DDA abundance values mean
    blood_dia.values['peptide']['abundance']  = steen_pep_dia * (blood_dda.values['peptide']['abundance'].mean() / steen_pep_dia.mean())
    #Logarithmize the datasets
    blood_dda = logarithmize(blood_dda)
    blood_dia = logarithmize(blood_dia)
    blood_dda.set_column_lf(molecule='peptide', column='abundance_gt', values=blood_dia.values['peptide']['abundance'], skip_foreign_ids=True)
    blood_dda.molecules['peptide'].rename(columns={'Sequence': 'sequence'}, inplace=True)
    return blood_dda

def load_human_ecoli_mixture_dda_dia_dataset(path: Path = Path('data/datasets/PXD018408_human_ecoli_mixture_dda_dia')):
    #Load DDA data
    maxquant_peps = pd.read_excel(path/'data.xlsx', sheet_name='Table S3 MaxQuant Output')
    maxquant_peps['is_human'] = maxquant_peps['Proteins'].str.contains('HUMAN').fillna(False)
    maxquant_peps['is_ecoli'] = maxquant_peps['Proteins'].str.contains('ECOLI').fillna(False)
    maxquant_peps.rename(columns={'Sequence':'sequence'}, inplace=True)
    dataset_dda = load_maxquant(peptides_table=maxquant_peps, peptide_columns = ['sequence', 'is_human', 'is_ecoli'])
    #Normalize according to non-changing human peptides
    peps = dataset_dda.molecules['peptide']
    ids = peps[peps.is_human & ~peps.is_ecoli].index
    dataset_dda.values['peptide']['intensity_normalized'] = normalize_sum(dataset=dataset_dda, molecule='peptide', column='Intensity',
                                                                          reference_ids=ids, reference_sample='03')
    #Load DIA data and normalize
    spectronaut_data = pd.read_excel(path/'data.xlsx', sheet_name='Table S4 Spectronaut Output', na_values=['Filtered'])
    spectronaut_data['peptide'] = spectronaut_data.loc[:, 'EG.PrecursorId'].str.replace(r'\[.*\]', '', regex=True).str.extract('_([A-Z]+)_')[0]
    spectronaut_samples = [c for c in spectronaut_data.columns if c not in {'EG.PrecursorId', 'PG.FASTAHeader', 'PG.ProteinAccessions', 'peptide'}]
    spectronaut_peptides = spectronaut_data.groupby('peptide')[spectronaut_samples].sum()
    dia_peps = spectronaut_peptides[spectronaut_peptides.index.isin(dataset_dda.molecules['peptide'].sequence)]
    dda_peps = dataset_dda.molecules['peptide']
    dda_peps = dda_peps[dda_peps.sequence.isin(dia_peps.index)]
    dia_peps.loc[dda_peps.sequence, ['is_human', 'is_ecoli']] = dda_peps.loc[:, ['is_human', 'is_ecoli']].values
    dia_samples = dia_peps.columns
    dia_samples = dia_samples[dia_samples.str.contains('.raw')]
    dia_samples = pd.Series(index=dia_samples, data=dia_samples.str.extract(r'.+EA100915_(\d{2})')[0].values)
    dia_peps = dia_peps.rename(columns=dia_samples)
    dia_peps_ds = read_multiple_mapped_dataframes(dfs={'peptide':dia_peps}, molecule_columns={'peptide':['is_human','is_ecoli']}, sample_columns=list(dia_samples))
    peps = dia_peps_ds.molecules['peptide']
    ids = peps[peps.is_human & ~peps.is_ecoli].index
    dia_peps_ds.values['peptide']['abundance_normalized'] = normalize_sum(dataset=dia_peps_ds, molecule='peptide', column='abundance',
                                                                          reference_ids=ids, reference_sample='05')
    #Add ground truth DIA values to DDA dataset
    gt_df = pd.DataFrame({'abundance':dia_peps_ds.values['peptide']['abundance_normalized']})
    dia_1x_samples = ['05','09','13','21','25','29','33','41']
    dia_2x_samples = ['06','10','14','18','26','30','34','38']
    dda_1x_samples = ['03','11','15','19','23','31','35','39']
    dda_2x_samples = ['04','08','16','20','24','28','36','40']
    sample_to_kind = dict([(s,'1x') for s in dia_1x_samples] + [(s,'2x') for s in dia_2x_samples] +
                        [(s,'1x') for s in dda_1x_samples] + [(s,'2x') for s in dda_2x_samples])
    gt_df['kind'] = gt_df.index.get_level_values('sample').map(sample_to_kind)
    gt = gt_df.groupby(['id', 'kind'])['abundance'].median()
    dda_values =dataset_dda.get_values_flat(molecule='peptide', columns=['intensity_normalized'],
                                            molecule_columns='sequence')
    dda_values['kind'] = dda_values.index.get_level_values('sample').map(sample_to_kind)
    indexer = pd.MultiIndex.from_frame(dda_values[['sequence', 'kind']])
    mask = indexer.isin(gt.index)
    dda_values.loc[mask, 'gt_dia'] = gt.loc[indexer[mask]].values
    dda_values.loc[dda_values['gt_dia']==0, 'gt_dia'] = np.nan #0 values are interpreted as missing
    dataset_dda.values['peptide']['gt_dia'] = dda_values['gt_dia']
    #Remove always missing peptides and proteins without any peptides
    pep_intensities = dataset_dda.get_samples_value_matrix(molecule='peptide', column='Intensity')
    peps = pep_intensities[~pep_intensities.isna().all(axis=1)].index
    dataset_dda = dataset_dda.get_molecule_subset(molecule='peptide', ids=peps)
    prots = dataset_dda.molecule_set.get_mapping_degrees(molecule='protein_group', mapping='peptide-protein_group')
    prots = prots[prots > 0].index
    dataset_dda = dataset_dda.get_molecule_subset(molecule='protein_group', ids=prots)
    dataset_dda.rename_molecule('protein_group', new_name='protein')
    dataset_dda.molecule_set.rename_mapping('peptide-protein_group', new_name='peptide-protein')
    #Scale DIA abundance values to match DDA abundance values mean
    dataset_dda.values['peptide']['abundance'] = dataset_dda.values['peptide']['intensity_normalized']
    gt_dia = dataset_dda.values['peptide']['gt_dia']
    dataset_dda.values['peptide']['abundance_gt']  = gt_dia * (dataset_dda.values['peptide']['abundance'].mean() / gt_dia.mean())
    #Logaritmize whole dataset including all abundance value columns
    dataset_dda = logarithmize(dataset_dda)
    return dataset_dda