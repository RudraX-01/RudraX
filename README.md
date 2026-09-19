---
title: RudraX
sdk: streamlit
app_file: app.py

---

# Post-Transformer Architectures: Fixed Memory vs. Growing Context

**🌟 Live Demo:** [https://rudrax.streamlit.app/](https://rudrax.streamlit.app/)

## One-Sentence Claim
A fixed-size recurrent state can process arbitrarily long sequences without allocating a new memory slot for each token, but storing more associations in that fixed state can cause retrieval interference.

## Why this matters
The KV cache constraint physically limits how much history models can process. Bounded recurrent architectures offer an O(1) memory alternative but face profound interference challenges.

## Intended Learner
ML practitioners, students, and engineers familiar with basic Transformers but new to Post-Transformer continuous-state architectures.

## Prerequisites
Understanding that language models generate tokens iteratively, and standard attention requires looking back at previous tokens in a context window.

## Learning Objectives
1. Understand why KV caching scales linearly and limits context lengths.
2. Understand how fixed-size recurrent states keep memory boundaries flat.
3. Observe and explicitly test the interference/forgetting trade-off inherent in fixed capacities.
4. Understand how BDH and BDH-CQ evolve these concepts into richer neural state representations.

## Learning Journey
1. Prediction
2. Write information into fixed state
3. Retrieve it (Truth vs Prediction)
4. Increase facts / correlation
5. Observe interference
6. Why does it happen? (Trade-off visual)
7. Connect to BDH + BDH-CQ
8. Compare KV cache
9. 60-second test
10. Free sandbox

## Mathematical Model
$M_{t+1} = M_t + v_t \cdot k_t^T$ (Outer product associative memory update)
$\hat{y} = M_t \cdot k_{query}$ (Linear retrieval)

## What the Toy Simulates
We simulate a linear recurrent memory (Fast-Weights). We observe retrieval interference as multiple associations are superposed into a fixed-dimensional state.

## What it Does NOT Simulate
It DOES NOT implement BDH's full network topology, sparse activations, or gating mechanisms from other recurrent architectures. It is not an implementation of BDH, BDH-CQ, Mamba, or RWKV.

## Experimental Protocol
See `research/experiment_protocol.md`.

## Results
Live experiments are evaluated with a deterministic seed for instant UI feedback. The Systemic Trade-offs graphs compute the Mean and Standard Deviation across 10 random seeds.

## Evidence Levels
**Formal:** The fixed-state matrix has constant size with respect to sequence length.
**Live empirical:** Our toy model demonstrates retrieval interference.
**Analytical:** KV-cache memory estimates.
**Primary-source research:** BDH architecture and synaptic-plasticity claims, and BDH-CQ reasoning.

## BDH Connection
Explains how Dragon Hatchling implements a richer version of this continuous state concept using scale-free networks and synaptic plasticity as working memory.

## BDH-CQ Connection
Explains how BDH-CQ modifies a recurrent contextual memory via demonstrations before querying a latent workspace for iterative reasoning.

## Limitations
Our toy model lacks gating (like LSTM/Mamba) which real models use to conditionally forget, nor does it model real-world language-model training. Target correlations of 0% do not guarantee perfectly orthogonal keys, meaning base interference always exists when capacity is exceeded.

## Live vs Precomputed
The associative memory tasks are executed LIVE in PyTorch. The Systemic Trade-offs graphs are executed live but cached in Streamlit for performance. The KV Cache is analytically computed.

## Synthetic Data
Vocabulary vectors are random Gaussian vectors generated at runtime.

## Architecture
`app.py`: Streamlit User Interface.
`simulation/`: PyTorch mathematical associative memory, task generation, evaluation, and experiment aggregation.
`tests/`: Unit tests for memory integrity.
`research/`: Evidence mapping and protocol.

## Setup & Reproduction
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Tests
Run tests using pytest: 
```bash
PYTHONPATH=. pytest tests/
```

## Primary Research Sources

1. **The Dragon Hatchling: The Missing Link between the Transformer and Models of the Brain**
   *Authors:* Adrian Kosowski, Przemysław Uznański, Jan Chorowski, Zuzanna Stamirowska, Michał Bartoszkiewicz
   *Year:* 2025
   *Identifier/URL:* arXiv:2509.26507
   *Technical Claim Supported:* Formalizes the BDH architecture, maintaining evolving synaptic state as working memory within a recurrent network (localizing state changes to synaptic interactions).

2. **BDH-CQ: In-Context Learning with Recurrent Latent Reasoning**
   *Authors:* Björn Engdahl, Adrian Kosowski, Jan Chorowski, Zuzanna Stamirowska, Przemysław Uznański, Junlin Jiang, Rohan Phadke, Remigiusz Kinas, Richard Zhong
   *Year:* 2026
   *Identifier/URL:* arXiv:2608.09888
   *Technical Claim Supported:* Uses recurrent contextual memory that updates with demonstrations and inference-time inputs, subsequently utilizing the compressed state for latent iterative computation (with noted compositional binding limitations).

3. **Mamba: Linear-Time Sequence Modeling with Selective State Spaces**
   *Authors:* Albert Gu, Tri Dao
   *Year:* 2023
   *Identifier/URL:* arXiv:2312.00752
   *Technical Claim Supported:* Making key state-space parameters input-dependent allows the model to selectively propagate or forget information along the sequence.

4. **RWKV: Reinventing RNNs for the Transformer Era**
   *Authors:* Bo Peng et al.
   *Year:* 2023
   *Identifier/URL:* arXiv:2305.13048
   *Technical Claim Supported:* Explores persistent recurrent state mechanisms to manage information accumulation as an alternative to the KV cache.

## Source & License Record

| Asset Category | Details & License |
| :--- | :--- |
| **Code** | This repository's original code is released under the MIT License. |
| **Data** | Runtime-generated synthetic Gaussian vectors. No external dataset. |
| **Weights** | No pretrained/model weights used. |
| **Graphics** | No external graphics used. Visualizations generated natively via Streamlit and Altair. |
| **Fonts** | System fonts used via standard Streamlit rendering. N/A for custom fonts. |
| **Third-party/Reused Components** | Libraries specified in `requirements.txt`: `streamlit` (Apache 2.0), `torch` (BSD-3-Clause), `pandas` (BSD-3-Clause), `altair` (BSD-3-Clause), `numpy` (BSD-3-Clause), `pytest` (MIT). |

## AI Assistance Disclosure

AI assistance was used to aid in developing this submission, specifically for:
- UI scaffolding
- boilerplate
- initial associative-memory implementation
- tests
- documentation/refactoring
- blog drafting/editing

**Author Responsibility & Verification:**
The author explicitly directed the scientific framing, selected the experimental protocol, executed and inspected the experiments, verified all outputs, and reviewed the final claims. The author is fully responsible for the final submission. AI was not used to independently validate scientific claims.
