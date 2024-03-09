from pyproteonet.data import Dataset

from gnn_imputation import impute_gnn

def main():
    for ds_name in ['prostate_cancer']:
        # Remove skip connection
        # print('Imputing ' + ds_name + ' without skip connection')
        # ds = Dataset.load(f'data/datasets_imputed/{ds_name}')
        # impute_gnn(ds, skip_connection=False, random_edges=False)
        # ds.save(f'data/datasets_experiments/{ds_name}_no_skip', overwrite=True)

        # Random rewiring of peptide graph
        # print('Imputing ' + ds_name + ' with random peptide graph')
        # ds = Dataset.load(f'data/datasets_imputed/{ds_name}')
        # impute_gnn(ds, skip_connection=True, random_edges=True)
        # ds.save(f'data/datasets_experiments/{ds_name}_random_edges', overwrite=True)

        # Random rewiring of peptide graph and no skip connection
        # print('Imputing ' + ds_name + ' with random peptide graph and without skip connection')
        # ds = Dataset.load(f'data/datasets_imputed/{ds_name}')
        # impute_gnn(ds, skip_connection=False, random_edges=True)
        # ds.save(f'data/datasets_experiments/{ds_name}_no_skip_random_edges', overwrite=True)

        # Set all sequence embeddings to zero
        print('Imputing ' + ds_name + ' without sequence embeddings')
        ds = Dataset.load(f'data/datasets_imputed/{ds_name}')
        impute_gnn(ds, skip_connection=True, random_edges=False, zero_sequence_embeddings=True)
        ds.save(f'data/datasets_experiments/{ds_name}_zero_sequence_embeddings', overwrite=True)

if __name__ == '__main__':
    main()
    