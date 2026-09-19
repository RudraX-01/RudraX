## How does BDH approach memory differently?

**Shared Idea:** Information persists in an internal state rather than a growing explicit cache.

### OUR TOY MODEL
**Input** $\rightarrow$ **State Update ($s_{t+1} = s_t + v_t k_t^T$)** $\rightarrow$ **State**
*Important Note:* Our simulator is a teaching model demonstrating the bounds of fixed-size continuous memory. It is **NOT** an implementation of BDH.

### DRAGON HATCHLING (BDH)
**Input** $\rightarrow$ **Neuron activity** $\rightarrow$ **Local interactions** $\rightarrow$ **Synaptic state / plasticity** $\rightarrow$ **Updated computation**
The official BDH architecture is a scale-free network structure with locally interacting neuron particles. Working memory during inference does not rely on a growing attention cache, but rather on **synaptic plasticity with Hebbian learning**. It utilizes sparse positive activations and a GPU-friendly state-space formulation to process sequences while updating the connectivity (the "wiring") as its working memory.
