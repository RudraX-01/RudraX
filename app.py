import streamlit as st
import pandas as pd
import numpy as np
import torch
import altair as alt
from simulation.kv_cache import estimate_kv_cache_memory
from simulation.recurrent_memory import AssociativeRecurrentMemory
from simulation.tasks import create_task
from simulation.evaluation import evaluate_memory
from simulation.experiments import run_interference_sweep, run_capacity_sweep, run_state_size_sweep

st.set_page_config(page_title="Post-Transformer Architectures", layout="wide")

st.title("Post-Transformer Architectures: Fixed Memory vs. Growing Context")
st.markdown("""
**Can a tiny memory remember an infinite stream?**

> *"A fixed-size recurrent state can process arbitrarily long sequences without allocating a new memory slot for each token, but storing more associations in that fixed state can cause retrieval interference."*
""")

with st.expander("Model Assumptions & Limitations (Click to view)"):
    st.info("""
    **Misconception:** "Fixed-size memory means the model can remember an unlimited amount of information."
    **Reality:** No. Fixed-size means the state representation doesn't grow with sequence length. Information still has to fit into that representation, so different associations can interfere.

    **This is a toy model:**
    It DOES: fixed-dimensional recurrent state, outer-product associative writes, vector retrieval, controlled key overlap.
    It DOES NOT: implement BDH, implement a trained language model, reproduce natural-language representations, model sophisticated gating, or prove universal memory limits of recurrent models.
    """)

# --- 1. PREDICTION ---
st.caption("PREDICT")
st.header("1. Make a Prediction")

st.markdown("""
**LIVE EXPERIMENT PROTOCOL**
- **State:** 8 × 8
- **Facts:** 20
- **Shared-key component:** 80%
- **Seed:** 42
- **Evaluation:** cosine-similarity nearest-answer retrieval
""")

st.warning("""
**Limitation:** This experiment demonstrates retrieval interference in this specific dense outer-product associative memory. It does not establish that all recurrent architectures exhibit the same interference curve. The toy model is not an implementation of BDH or BDH-CQ.
""")

prediction = st.radio(
    f"What do you expect will happen to recall?",
    ["Select an option...", "Recall will increase", "Recall will decrease", "Recall will stay similar", "The system runs out of GPU memory"]
)

if prediction != "Select an option...":
    st.divider()
    
    # --- 2. WRITE INFORMATION ---
    st.caption("LIVE")
    st.header("2. Write Information into Fixed State")
    st.markdown("**LIVE EXPERIMENT** | Generated and evaluated now in PyTorch.")
    
    # --- EXPERIMENT EXECUTION (DETERMINISTIC LIVE DEMO) ---
    pairs, k_dict, val_dict, measured_sim = create_task(20, 8, 80, seed=42)
    mem = AssociativeRecurrentMemory(8)
    
    fact1_k_str, fact1_v_str, fact1_k_vec, fact1_v_vec = pairs[0]
    mem.update(fact1_k_vec, fact1_v_vec)
    state_after_1 = mem.get_state().numpy()
    
    st.markdown("### What is stored?")
    st.markdown("**No new memory slot is created for this fact. The existing state matrix is updated.**")
    st.latex(r"M_{t+1} = M_t + v_t k_t^T")
    st.latex(r"\hat{y} = M_t k_{query}")
    
    c1, c2, c3, c4 = st.columns([1,1,2,2])
    with c1:
        st.markdown(f"**Key $k$**<br>'{fact1_k_str}'", unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(fact1_k_vec.numpy().round(3), columns=["value"]))
    with c2:
        st.markdown(f"**Value $v$**<br>'{fact1_v_str}'", unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(fact1_v_vec.numpy().round(3), columns=["value"]))
    with c3:
        st.markdown(f"**Outer Product $v k^T$**")
        st.dataframe(pd.DataFrame(torch.outer(fact1_v_vec, fact1_k_vec).numpy().round(3)))
    with c4:
        st.markdown("**State Matrix $M$** (After Addition)")
        st.dataframe(pd.DataFrame(state_after_1.round(3)))
        
    st.caption("PRIMARY SOURCE")
    st.info("""
    **Frontier Connection: Dragon Hatchling (BDH)**  
    Our toy model uses a dense outer-product state. BDH uses a fundamentally richer recurrent network with local interactions, sparse activity, and evolving synaptic/edge state. The BDH paper reports that this synaptic state carries interpretable information—for instance, concept-related inputs dynamically lead to specific synaptic responses (e.g., tracking currencies).
    """)
        
    for i in range(1, len(pairs)):
        mem.update(pairs[i][2], pairs[i][3])
        
    acc, results = evaluate_memory(mem, pairs, val_dict)
    
    st.divider()
    
    # --- 3. RETRIEVE IT ---
    st.caption("MEASURED")
    st.header("3. Retrieve Information (Truth vs Prediction)")
    
    st.markdown("### Prediction → Actual Result")
    st.markdown(f"**Prediction:** {prediction}")
    st.markdown(f"**Measured recall:** {acc:.1f}%")
    
    st.markdown("Retrieval Log (First 5):")
    for res in results[:5]:
        status_icon = "Correct" if res["Correct"] else "Interference"
        st.markdown(f"**Query:** {res['Query']} | **Truth:** {res['Ground Truth']} | **Output:** {res['Model Prediction']} {status_icon}")
        
    st.divider()

    # --- 4 & 5. FACTS & INTERFERENCE ---
    st.header("4. Increase Facts / Observe Interference")
    st.markdown("**FORMAL** | State shape doesn't depend on sequence length.")
    
    @st.cache_data
    def cached_capacity_sweep():
        return pd.DataFrame(run_capacity_sweep([5, 20, 50], 8, 80, num_seeds=10))
    
    df_cap = cached_capacity_sweep()
    st.table(df_cap)
    
    st.markdown("### Why does it happen?")
    st.markdown("> **Every new association is added to the same state. When keys overlap, their writes overlap too. A query therefore receives contributions from multiple stored associations.**")
    st.markdown("""
    **Fact A:** $key_A \\times value_A \\rightarrow \\text{SAME STATE}$
    **Fact B:** $key_B \\times value_B \\rightarrow \\text{SAME STATE}$
    
    If $key_A \\approx key_B$, then $write_A \\approx write_B$. Retrieval becomes mixed.
    """)
    
    st.caption("PRIMARY SOURCE")
    st.warning("""
    **Frontier Connection: BDH-CQ & Capacity Limits**  
    **BDH-CQ** uses a recurrent contextual memory to store history: $S_t=U_\\epsilon(S_{t-1},D_t)$, where demonstrations $D_t$ iteratively update the memory. It then performs latent reasoning on this compressed state: $H_{r+1}=F_\\epsilon(H_r,S_K)$.
    While much richer than our dense matrix, even advanced recurrent states face capacity and composition limits. The BDH-CQ paper's controlled ARC-like interventions found that composing rotation with relocation succeeded on 100% of held-out cases (72/72), but composing a color swap with relocation failed completely — 0 of 72 — revealing a specific object-property binding weakness rather than a general capacity ceiling.
    """)
    
    st.divider()
    
    # --- 6. WHY? THE TRADE-OFF ---
    st.header("6. The Systemic Trade-Off")
    
    colG1, colG2 = st.columns(2)
    with colG1:
        st.subheader("Recall vs Shared-Key Component")
        @st.cache_data
        def cached_interference_sweep():
            return pd.DataFrame(run_interference_sweep(20, 8, [0, 20, 40, 60, 80, 100], num_seeds=10))
        df_sim = cached_interference_sweep()
        chart1 = alt.Chart(df_sim).mark_errorband(extent='stdev').encode(
            x='Shared Component (%):Q',
            y=alt.Y('Mean Recall (%):Q', scale=alt.Scale(domain=[0, 100]))
        ) + alt.Chart(df_sim).mark_line(point=True).encode(
            x='Shared Component (%):Q',
            y='Mean Recall (%):Q'
        )
        st.altair_chart(chart1, use_container_width=True)

    with colG2:
        st.subheader("Recall vs State Size")
        @st.cache_data
        def cached_state_sweep():
            return pd.DataFrame(run_state_size_sweep([4, 8, 16, 32], 20, 80, num_seeds=10))
        df_dim = cached_state_sweep()
        chart2 = alt.Chart(df_dim).mark_errorband(extent='stdev').encode(
            x='State Dimension:O',
            y=alt.Y('Mean Recall (%):Q', scale=alt.Scale(domain=[0, 100]))
        ) + alt.Chart(df_dim).mark_line(point=True).encode(
            x='State Dimension:O',
            y='Mean Recall (%):Q'
        )
        st.altair_chart(chart2, use_container_width=True)
        
    st.divider()

    # --- 7. KV CACHE COMPARISON ---
    st.caption("ANALYTICAL")
    st.header("7. Compare Architecture (KV Cache vs Fixed State)")
    
    st.info("**Fixed state size does not mean unlimited memory capacity. It means the representation does not allocate a new state slot for every token.**")
    st.markdown("**ANALYTICAL** | Calculated from the KV-cache formula.")
    
    colA, colB = st.columns(2)
    with colA:
        st.markdown("**KV Cache**")
        st.markdown("Precise history<br>Memory grows with context", unsafe_allow_html=True)
    with colB:
        st.markdown("**Fixed State**")
        st.markdown("Bounded state<br>Information must be compressed<br>Interference can occur", unsafe_allow_html=True)
        
    seqs = [1000, 10000, 100000, 1000000]
    kv_memories = [estimate_kv_cache_memory(s) / (1024 * 1024 * 1024) for s in seqs] # in GB
    
    df_mem = pd.DataFrame({
        "Sequence Length": seqs,
        "Memory (GB)": kv_memories,
    })
    
    st.markdown("Under our assumptions (12 layers, 8 heads, 64-dim, FP16), the K/V cache footprint grows linearly with sequence length:")
    st.table(df_mem.set_index("Sequence Length"))
    st.caption("This calculation covers the K/V tensors only, not model weights, activations, or allocator overhead.")
    
    st.divider()
    
    # --- 8. 60-SECOND CHALLENGE ---
    st.header("8. 60-Second Test")
    
    q1 = st.radio("1. If the number of facts increases while state dimension stays fixed, what happens?", ["Select...", "State grows", "Recall may degrade", "KV cache appears", "Nothing changes"])
    q2 = st.radio("2. Why?", ["Select...", "The state gets physically larger", "Multiple associations are superposed into the same state, creating interference", "The model stops computing"])
    q3 = st.radio("3. What happens to KV-cache memory as sequence length grows?", ["Select...", "Remains fixed", "Grows linearly under stated assumptions"])
    q4 = st.radio("4. What makes BDH different from our toy?", ["Select...", "Same as our matrix", "BDH uses a richer network of neuron-like units with evolving synaptic/edge state"])
    
    if st.button("Score Challenge"):
        score = 0
        if q1 == "Recall may degrade": score += 1
        else: st.error("Q1 Not quite. The important distinction is that fixed dimensional representations must compress data, causing degradation.")
        
        if q2 == "Multiple associations are superposed into the same state, creating interference": score += 1
        else: st.error("Q2 Not quite. The important distinction is that the state doesn't grow physically, so writes overlap.")
        
        if q3 == "Grows linearly under stated assumptions": score += 1
        else: st.error("Q3 Not quite. The KV cache explicitly stores every token representation, so it grows linearly.")
        
        if q4 == "BDH uses a richer network of neuron-like units with evolving synaptic/edge state": score += 1
        else: st.error("Q4 Not quite. BDH doesn't use a dense outer-product matrix; it uses an evolving network.")
        
        st.info(f"**Score:** {score}/4")
        if score == 4:
            st.success("4/4 - You can now explain the claim!")
            
    st.divider()
    
    # --- 9. EXPLAIN IT YOURSELF ---
    st.header("9. Explain it yourself")
    explain = st.text_area("Why can a fixed-size state use constant memory while still suffering retrieval interference?")
    if st.button("Reveal Model Answer"):
        st.info("**Your explanation:**\n" + explain)
        st.success("**Model answer:**\nBecause the state does not grow with the sequence. New associations therefore have to be represented within the same fixed-dimensional state, and overlapping associations can contaminate retrieval.")
        st.markdown("Compare your explanation with the model explanation.")

    st.divider()

    # --- 10. SANDBOX ---
    st.header("10. Free Sandbox: Break the memory yourself")
    c_sb1, c_sb2, c_sb3, c_sb4 = st.columns(4)
    sb_dim = c_sb1.selectbox("State Dimension", [4, 8, 16, 32], key="sb_dim")
    sb_facts = c_sb2.slider("Facts", 5, 100, 20, key="sb_facts")
    sb_rho = c_sb3.slider("Shared-Key Component (%)", 0, 100, 80, key="sb_rho")
    sb_seed = c_sb4.number_input("Seed", 0, 1000, 42, key="sb_seed")
    
    st.markdown("*(The experiment runs automatically when you change the sliders)*")
    p, _, v, sb_msim = create_task(sb_facts, sb_dim, sb_rho, seed=sb_seed)
    m = AssociativeRecurrentMemory(sb_dim)
    for _, _, kv, vv in p: m.update(kv, vv)
    sb_acc, _ = evaluate_memory(m, p, v)
    
    st.markdown("### Immediate Consequence")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("State size", f"{sb_dim}x{sb_dim}")
    m2.metric("Facts stored", sb_facts)
    m3.metric("Measured correlation", f"{sb_msim:.2f}")
    m4.metric("Recall", f"{sb_acc:.1f}%")
