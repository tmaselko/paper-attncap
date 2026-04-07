
# Separate, Project, and Amplify: Attention's Geometry of Retrieval
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19422845.svg)](https://doi.org/10.5281/zenodo.19422845)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

![Plots of embeddings for a `d_k = 3` model, before and after training](_markdown/embed_d003_pca.png)
*A `d_k = 3` model rearranges 512 symbol embeddings from a point cloud to a sphere with 7.5 to 20 degrees of separation between vectors.*


**TL;DR:** By decoupling positional confounds, we demonstrate that attention's retrieval capacity is purely geometric and unconstrained by head dimension. Using our Tuple-Structured Associative Recall (TSAR) framework, **a 1-layer Transformer achieves perfect associative recall on 16K-assignment sequences with a head dimension of only `d_k = 6` and training on sequences of no more than 1K assignments.**

This repository contains the minimal model, the TSAR synthetic task, and the complete reproduction code for the paper: ["Separate, Project, and Amplify: Attention's Geometry of Retrieval"](https://zenodo.org/records/19422845).

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

Tuple-structuring lets you control the exact semantics of your synthetic task, in a way that propagates through the model for easy post-training analysis. Tuple-Structured Associative Recall (TSAR, implemented in `tsar.py`) is MQAR with all positional confounds removed. Instead of a token sequence, its a token-tuple sequence, where each position in the sequence is a BOS, an assignment, or a retrieval.

This can be solved surprisingly easily by Transformers, which only require a single layer with a single head to reach perfect accuracy. No positional encodings, no causal masking, not even an MLP is necessary.

### Retrieval in Three Steps

Transformer models solve retrieval by **separating** its representations into a spherical code, **projecting** the code vectors out of the hidden state, and **amplifying** those vectors to saturate softmax and sharply attend the matching key. All three steps are necessary for robust function, and all three steps are beautifully illustrated in models trained on TSAR (Section 4).

### Spherical Code Representations

The use of a spherical code to represent the symbols demonstrates a third category of internal model representation: Dense spherical codes.

- **Orthogonality** can represent a linear number of features, all totally independent within a single vector.
- **Superposition** can represent an exponential number of features, but interference prevents them all from being present simultaneously.
- **Dense Spherical Codes** can represent an unbounded number of features, but they interfere so heavily that only a tiny fraction of them can be simultaneously present in a vector.

In reality, all three of these are types of spherical codes, with different constraints on their geometry. Dot products treat every single one of them the same: They are different geometric interpretations of the same object, each understandable in terms of the other, and which is most illuminating changes from one to another throughout a model.

To attention, all `Q` and `K` are dense spherical codes (or superpositions, in the case of fuzzy semantic matching), but the `V` they produce become superpositions. An MLP treats a vector as a superposition with `up_proj`, reinterprets it with orthogonality for activation, only to convert it right back into a superposition with `down_proj`. The hidden state is all of these at once, complete chaos where every subspace can behave differently. Tuple-Structuring aids in taming this, by controlling the separation of these geometrically-distinct subspaces.

### Capacity

The "unbounded" descriptor for dense spherical codes is not hyperbolic. For a single-key, single-value associative recall task like TSAR, we see this density exploited to encode each possible symbol as a unique feature, which permits their direct comparison in `Q` and `K`.
With real numbers, `d_k = 2` is sufficient for any problem size (`construct_model_unitcircle` in `constructions.py`, Section 5.1 in the paper). With finite bits spread across any `d_k`, it approaches or reaches the representational limit of that space (`construct_model_hypergrid` in `constructions.py`, Section 5.3 in the paper). With total bits `B`, capacity approaches `2^B` as long as `d_k >= 2`.
Trained models can reach remarkable retrieval capacities (Section 5.3.4), but it is difficult to constrain or measure their true "bit usage". On the other hand, the spherical codes they learn tend to be significantly better optimized than the crude lattice codes the paper constructs for similarly-sized models.

### What about Hopfield?

Attention is usually attributed an "exponential in `d_k`" storage capacity, based on the modern Hopfield analysis by Ramsauer et al. This analysis is not wrong, but nor is it universally applicable to Transformers. The analysis assumes all embeddings are random vectors on a sphere, and that their magnitude is fixed. Neither is true in trained models.

Constructed models tested with random-embed and fixed-magnitude constraints (`construct_model_randomsphere` in `constructions.py`) show the expected exponential-in-`d_k` sigmoid error curves very clearly. However, trained models, even when their embeddings are frozen to be random unit vectors, will still elevate their magnitude (ie, the "amplify" mechanism) to heavily compress their error curves and maximize error-free capacity.

## Implications

This is where the interesting predictions and implications of the work lie. These are developed in the paper, based on the data collected by the experiments in this repo.

- Retrieval appears to have self-defeating training dynamics that cripple its own gradients upon formation. If our theories are correct, this places retrieval heads on a deathmarch towards `W_{QK}` gradient loss during training, one that will almost certainly arrive long before the model has finished training.
- Positional Encodings should be designed with retrieval's mechanisms in mind, but mainstream approaches seem to directly interfere with them. This suggests that the mere presence of RoPE or Sinusoidal in a model would significantly reduce its performance on retrieval.
- Length generalization failures are due to positional encodings not accounting for retrieval's mechanisms. Never-before-seen encodings warp representational geometry into or out of alignment, breaking separation. The magnitudes are no longer applicable for the geometry they are applied to, compounded by an increased context size which by itself may demand higher magnitudes.
- "Out-of-distribution" can be seen as "never accounted for in the learned spherical code". What hasn't been seen cannot be separated, and what cannot be separated cannot be distinguished.
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