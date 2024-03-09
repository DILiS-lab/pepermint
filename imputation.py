import json
import time
from pathlib import Path

import numpy as np
from pyproteonet.data import Dataset
from pyproteonet.imputation import impute_molecule

from gnn_imputation import impute_gnn

COMPETITOR_IMPUTATION_METHODS = ['median']#['mindet', 'minprob', 'median', 'knn', 'missforest', 'bpca', 'isvd', 'vae', 'dae', 'cf', 'iterative']

def main():
    runtimes_pep = dict()
    #for ds_name in ['blood_ddia', 'prostate_cancer', 'crohns_disease', 'crohns_fibrosis', 'breast_cancer', 'maxlfqbench', 'human_ecoli_ddia']:
    for ds_name in ['prostate_cancer_zero_sequence_embeddings', 'prostate_cancer_no_skip', 'prostate_cancer_random_edges', 'prostate_cancer_no_skip_random_edges']:
        print('Imputing ' + ds_name)
        ds = Dataset.load(f'data/datasets_experiments/{ds_name}')
        #ds = Dataset.load(f'data/datasets_imputed/{ds_name}')
        runtimes_pep[ds_name] = impute_molecule(dataset=ds, molecule='peptide', column='abundance', methods=COMPETITOR_IMPUTATION_METHODS)
        # start = time.time()
        # impute_gnn(ds)
        # rt = time.time() - start
        # runtimes_pep[ds_name]['gnn_imp']  = rt
        ds.save(f'data/datasets_experiments/{ds_name}', overwrite=True)
    if Path('data/runtimes.json').exists():
        with open('data/runtimes.json', 'r') as f:
            runtimes_loaded = json.load(f)
    else:
        runtimes_loaded = dict()
    for ds_name, runtimes in runtimes_pep.items():
        if ds_name in runtimes_loaded:
            runtimes_loaded[ds_name].update(runtimes)
        else:
            runtimes_loaded[ds_name] = runtimes
        with open('data/runtimes.json', 'w') as f:
            json.dump(runtimes_loaded, f)

if __name__ == '__main__':
    main()
    