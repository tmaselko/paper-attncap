
# Separate, Project, and Amplify: Attention's Geometry of Retrieval
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19422845.svg)](https://doi.org/10.5281/zenodo.19422845)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

![Plots of embeddings for a `d_k = 3` model, before and after training](_markdown/embed_d003_pca.png)
*A `d_k = 3` model rearranges 512 symbol embeddings from a point cloud to an emergent spherical code.*

## Introduction

This repository contains the minimal model, the TSAR synthetic task, and a complete reproduction for the paper ["Separate, Project, and Amplify: Attention's Geometry of Retrieval"](https://zenodo.org/records/19422845).

Without positional confounds, Transformers solve retrieval effortlessly. Using Tuple-Structured Associative Recall (TSAR) to isolate retrieval, a single-layer, single-head Transformer with a head dimension of six can achieve perfect accuracy on 16384-assignment sequences, while training on sequences of no more than 1024 assignments.

Retrieval is geometric, not dimensional. The paper, and thus this repo, unpack the mechanisms that make retrieval work, alongside the possible implications those mechanisms have for real-world models.


## Quickstart & Usage

This software requires Python 3.12+. Setting up an isolated environment (venv, conda) is recommended.

### Installation

```bash
git clone https://github.com/tmaselko/paper-attncap
cd paper-attncap
pip install -r requirements.txt
```

There are several ways to run the reproduction, depending on your time and interest. The script can be stopped at any time and re-runs will automatically resume from where it left off, reusing previous work. The repro script stores its current progress in a `models` folder, and places charts/tables in a `charts` folder.

### Reproduce: Headlines Only

Default mode. Only performs a few of the "most relevant" experiments (trains `d_k=[3, 6, 8, 16]` until successful and then tests all model constructions). This will attempt to generate an embedding rendering like the one seen above and some of the mechinterp charts/tables seen in the paper.

Requirements: About five minutes on a 4090 and a few megabytes of storage.

```bash
python -m src.repro
```

### Reproduce: One of Each

Trains one of each model variant and size, including ablations. Produces all charts and figures in the paper, but without the repeated samples for statistical significance.

Requirements: About two hours on a 4090 and about 15gb of storage.

```bash
python -m src.repro --short
```

### Reproduce: The Entire Paper

Reproduces all the graphs and figures in the paper, including up to 20 training and testing runs for each of the hundreds of model variants.

Requirements: 20-30 hours on a 4090 and about 160gb of storage.

```bash
python -m src.repro --all
```


## Results, Findings, and Interesting Things

### Tuple-Structured Task Design

Tuple-Structured Associative Recall (TSAR, implemented in `tsar.py`) is Multi-Query Associative Recall with all positional confounds removed. Instead of a token sequence, its a token-tuple sequence, where each position in the sequence is either the BOS, an assignment, or a retrieval. Tuple structuring lets you control the exact problem scope and semantics of a task, in a way that propagates through the model for easy post-training analysis.

For example, TSAR distills retrieval to its essentials, and can be solved surprisingly easily by Transformers. Only a single layer and a single head is needed, with no positional encodings, no causal masking, not even an MLP. When studying the trained model, you already know what information each region of the hidden state contains, and can partition the model's weights accordingly to see how it handles each one.

### Retrieval in Three Steps

Transformer models solve retrieval by **separating** its representations into a spherical code, **projecting** the code vectors out of the hidden state, and **amplifying** those vectors to saturate softmax and sharply attend the matching key. All three steps are necessary for robust function, and all three steps are beautifully illustrated in models trained on TSAR (Section 4).

### Spherical Code Representations

The use of a spherical code to represent key and value symbols demonstrates a third category of internal model representation: Dense spherical codes, which we can think of as "hyperposition":

- **Orthogonality** can represent a linear number of features, all totally independent within a single vector.
- **Superposition** can represent an exponential number of features, but interference prevents them all from being present simultaneously.
- **Dense Spherical Codes (Hyperposition)** can represent an unbounded number of features, but they interfere so heavily that only a tiny fraction of them can be simultaneously present in a vector.

In reality, "spherical code" can broadly describe all three of these geometries. Dot products treat every single one of them the same, after all. Each is a geometric interpretation that can be understood in terms of the others, and which one is most applicable changes throughout a model.

Within attention, all `Q` and `K` are hyperpositions (or superpositions, in the case of fuzzy semantic matching), but the `V` they select become combinations of superpositions. An MLP treats a vector as a superposition with `up_proj`, reinterprets it with orthogonality for activation, only to convert it right back into a superposition with `down_proj`.

The hidden state contains all of these at once. It's complete chaos, where every subspace and subset can behave differently, making interpretation a nightmare. Tuple structuring aims to tame this with controlled separation of conceptually distinct subspaces, making each one individually inspectable.

### Capacity

The "unbounded" descriptor for dense spherical codes is not hyperbolic. For a single-key, single-value associative recall task like TSAR, we see this density exploited to encode each possible symbol as a unique feature, which permits their direct comparison in `Q` and `K`.

With real numbers, `d_k = 2` is sufficient for any problem size (`construct_model_unitcircle` in `constructions.py`, Section 5.1 in the paper).
With finite bits spread across any `d_k`, it approaches or reaches the representational limit of that space (`construct_model_hypergrid` in `constructions.py`, Section 5.3 in the paper). With total bits `B`, capacity approaches `2^B` as long as `d_k >= 2`.

Trained models can reach remarkable retrieval capacities (Section 5.3.4), but it's difficult to constrain or measure their true "bit usage". On the other hand, the spherical codes they learn tend to be significantly better separated than the crude lattice codes the paper constructs for similarly-sized models.

### What about Hopfield?

Attention is usually attributed an "exponential in `d_k`" storage capacity, based on the modern Hopfield analysis by Ramsauer et al. This analysis is not wrong, but nor is it universally applicable to Transformers. The analysis assumes all embeddings are random vectors on a sphere, and that their magnitude is fixed. Neither is true in trained models.

Constructed models tested with random-embed and fixed-magnitude constraints (`construct_model_randomsphere` in `constructions.py`) show the expected exponential-in-`d_k` sigmoid error curves very clearly. However, trained models, even when their embeddings are frozen to be random unit vectors, will still elevate their magnitude (ie, the "amplify" mechanism) to heavily compress their error curves and maximize error-free capacity.

### What about real models?

TSAR is a toy task, solved by toy models. It lacks even a thousandth of the complexity of real-world LLMs, and doesn't even have the positional awareness required for language processing. How applicable can it be?

By definition of the math that backs Transformers, this geometric understanding of retrieval is almost certainly applicable to all models with that architecture. However, whether it is the whole story, or to what extent its implications are predictive, is a separate and very important question yet to be answered. This research only documents the process and mechanisms discovered in the simplest possible case, and extrapolates from them to produce predictions about the architecture in general.

It provides specific directions to look in to improve positional encodings or training dynamics, but cannot claim to that those directions will be ultimately productive for real-world models. Clarity and simplicity made the analysis easy, but also limits its immediate application. That said...


## Implications

This is where the interesting predictions and implications of the work lie. These are developed in the paper, based on the data collected by the experiments in this repo.

- Retrieval appears to have self-defeating training dynamics. If our theories are correct, once a retrieval head forms it is on a deathmarch towards losing `W_{QK}` gradients during training, which will almost certainly occur long before the model has finished training.
- Mainstream Positional Encoding approaches appear to directly interfere with retrieval's mechanisms. This suggests that the mere presence of RoPE or Sinusoidal embeddings in a model would significantly reduce its performance on retrieval, which has so far been borne out in preliminary followup experiments.
- Length generalization failures also seem primarily attributable to positional encodings. Never-before-seen encodings warp representational geometry into or out of alignment, breaking separation. The magnitudes are no longer applicable for the geometry they are applied to, compounded by an increased context size which by itself may demand higher magnitudes.
- "Out-of-distribution" can be seen as "never accounted for in the learned spherical code". What hasn't been seen cannot be separated, and what hasn't been separated cannot be distinguished.
- Attention is driven by its inputs. The work it does, the capabilities it has, are wholly defined by the geometry placed in the path of its projections. That geometry is what is shaped to solve a task... or to compensate for a positional encoding applying arbitrary rotations to its dimensions.



## Citation

If this work saved you time or gave you ideas, please cite as:

```bibtex
@misc{maselko2026retrieval,
  author       = {Maselko, Theodore},
  title        = {Separate, Project, and Amplify: Attention's Geometry of Retrieval},
  month        = apr,
  year         = 2026,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.19422845},
  url          = {https://doi.org/10.5281/zenodo.19422845},
}
```