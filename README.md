
# Separate and Amplify: Attention's Geometry of Retrieval

![Visualization of embeddings for a $d_k=3$ trained model](_markdown/embed_d003_pca.png)

This repository contains the TSAR synthetic task, minimal model, and training/repro code for the paper ["Separate and Amplify: Attention's Geometry of Retrieval"](https://zenodo.org/records/19359748). DOI: `10.5281/zenodo.19359748`.

## Usage

This software requires Python 3.12+. Setting up an environment (venv, conda, etc) beforehand is recommended.

```
pip install -r requirements.txt
python -m src.repro
```

The script may take several hours to finish, but can be stopped at any time and re-runs will automatically resume from where it left off. The repro script stores its current progress in a `models` folder, and places charts/tables in a `charts` folder.
