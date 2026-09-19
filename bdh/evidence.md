## BDH Evidence Map

**Claim:** BDH working memory involves synaptic plasticity.
**Source:** BDH 2025 paper (arXiv:2509.26507).
**Evidence Level:** Primary source.
**Where Used:** BDH module in `app.py`.
**Limitation:** Our toy model does not reproduce BDH's full architecture.

**Claim:** BDH utilizes a scale-free network structure of locally interacting neurons.
**Source:** BDH 2025 paper.
**Evidence Level:** Primary source.
**Where Used:** BDH module & Blog post.
**Limitation:** Our fast-weight matrix is a dense homogeneous tensor, unlike a scale-free graph.

**Claim:** BDH processes sequences using a persistent internal state instead of a growing cache.
**Source:** BDH 2025 paper / Pathway Research.
**Evidence Level:** Primary source.
**Where Used:** The core scientific claim of the project.
**Limitation:** BDH's GPU-friendly formulation is more complex than a standard outer-product update.
