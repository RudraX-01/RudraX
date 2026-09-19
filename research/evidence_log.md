# Evidence Log

| Claim | Evidence Type | Source | Experiment | Result | Limitation | Last Verified |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Fixed matrix capacity is independent of sequence length | Formal | Toy Model Math | N/A | $O(1)$ memory | N/A | Today |
| Bounded states suffer retrieval interference | Live empirical | experiments.py | 10-seed sweep | Mean recall degrades | Not a full LLM | Today |
| KV-cache grows linearly | Analytical | kv_cache.py | N/A | Linear O(N) | Excludes activations | Today |
| BDH uses synaptic plasticity | Primary-source | BDH 2025 | N/A | Network state | Toy doesn't simulate this | Today |
