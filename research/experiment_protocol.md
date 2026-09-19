# Experiment Protocol

This document outlines the evaluation protocol for the live and precomputed experiments in the DataForge explainer.

**State dimension:** Varies (default 8, sweep: 4, 8, 16, 32)
**Number of facts:** Varies (default 20, sweep: 5, 20, 50)
**Seed policy:** `42` for live single-shot demos. `0..9` for multi-seed sweep aggregations.
**Number of seeds:** 10 (for Systemic Trade-offs graphs).
**Key generation:** Generated synthetically using a shared base component $\rho$.
**Value generation:** Random Gaussian vectors.
**Correlation parameter:** Shared-Key Component ($0\% - 100\%$). Controls overlap.
**Evaluation metric:** Mean Recall (%). Percentage of queries whose nearest cosine neighbor in the candidate set equals the ground-truth value.
**Retrieval rule:** $y = M k_{query}$, then $\max(cosine\_sim(y, v_{candidate}))$.
**Aggregation method:** Mean and Standard Deviation across 10 random seeds.
