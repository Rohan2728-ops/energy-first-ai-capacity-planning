
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# PAGE SETUP
# ---------------------------------------------------------

st.set_page_config(
    page_title="Applying an Energy-First Lens to AI Capacity Planning",
    page_icon="⚡",
    layout="wide"
)

# ---------------------------------------------------------
# SYNTHETIC MODEL ASSUMPTIONS
# ---------------------------------------------------------

GPU_POWER_KW = 1.5
INFRASTRUCTURE_OVERHEAD_FACTOR = 1.30
MW_PER_GPU = (
    GPU_POWER_KW * INFRASTRUCTURE_OVERHEAD_FACTOR
) / 1000

# ---------------------------------------------------------
# SYNTHETIC SITE DATA
# ---------------------------------------------------------

sites = pd.DataFrame([
    {
        "site": "Site A",
        "region": "West",
        "gpu_type": "B200",
        "installed_gpus": 2560,
        "energized_mw": 120,
        "power_headroom_mw": 18,
        "future_power_mw": 25,
        "future_power_available_month": "2027-07",
        "network_topology": "High-bandwidth fabric available",
        "contiguous_gpu_block": 2560,
        "currently_committed_gpus": 256,
        "strategic_note":
            "Strong current compute; limited near-term power expansion"
    },
    {
        "site": "Site B",
        "region": "Central",
        "gpu_type": "B200",
        "installed_gpus": 1536,
        "energized_mw": 85,
        "power_headroom_mw": 42,
        "future_power_mw": 110,
        "future_power_available_month": "2027-03",
        "network_topology": "Partial expansion required",
        "contiguous_gpu_block": 1024,
        "currently_committed_gpus": 512,
        "strategic_note":
            "Limited current compute; strongest future energy expansion"
    },
    {
        "site": "Site C",
        "region": "East",
        "gpu_type": "B200",
        "installed_gpus": 4096,
        "energized_mw": 210,
        "power_headroom_mw": 55,
        "future_power_mw": 0,
        "future_power_available_month": "—",
        "network_topology": "High-bandwidth fabric available",
        "contiguous_gpu_block": 4096,
        "currently_committed_gpus": 1024,
        "strategic_note":
            "Strongest immediate placement capacity; limited future expansion"
    }
])

# ---------------------------------------------------------
# SYNTHETIC CUSTOMER DATA
# ---------------------------------------------------------

customers = pd.DataFrame([
    {
        "customer": "Alpha",
        "workload": "Distributed training",
        "gpu_count": 2048,
        "start_month": "2027-01",
        "close_probability": 0.70,
        "contract_value_m": 40,
        "required_region": "Any",
        "high_bandwidth_network_required": True,
        "contiguous_cluster_required": True,
        "status": "Pipeline"
    },
    {
        "customer": "Beta",
        "workload": "Inference",
        "gpu_count": 512,
        "start_month": "2026-12",
        "close_probability": 1.00,
        "contract_value_m": 18,
        "required_region": "Any",
        "high_bandwidth_network_required": False,
        "contiguous_cluster_required": False,
        "status": "Committed"
    },
    {
        "customer": "Gamma",
        "workload": "Distributed training",
        "gpu_count": 1024,
        "start_month": "2027-02",
        "close_probability": 0.40,
        "contract_value_m": 15,
        "required_region": "West",
        "high_bandwidth_network_required": True,
        "contiguous_cluster_required": True,
        "status": "Pipeline"
    }
])

# ---------------------------------------------------------
# MODEL CALCULATIONS
# ---------------------------------------------------------

def workload_mw(gpu_count):
    return gpu_count * MW_PER_GPU


sites["uncommitted_installed_gpus"] = (
    sites["installed_gpus"]
    - sites["currently_committed_gpus"]
)

sites["power_supported_gpu_headroom"] = (
    sites["power_headroom_mw"] / MW_PER_GPU
)

sites["usable_gpu_headroom"] = sites[
    [
        "uncommitted_installed_gpus",
        "power_supported_gpu_headroom"
    ]
].min(axis=1)

alpha = customers.loc[
    customers["customer"] == "Alpha"
].iloc[0]

gamma = customers.loc[
    customers["customer"] == "Gamma"
].iloc[0]

capacity = sites.copy()

capacity["available_gpus_after_beta"] = (
    capacity["usable_gpu_headroom"].astype(float)
)

capacity["available_power_after_beta_mw"] = (
    capacity["power_headroom_mw"].astype(float)
)

capacity.loc[
    capacity["site"] == "Site B",
    "available_gpus_after_beta"
] -= 512

capacity.loc[
    capacity["site"] == "Site B",
    "available_power_after_beta_mw"
] -= workload_mw(512)


def simulate_alpha(alpha_site):
    result = capacity.copy()

    result["available_gpus"] = (
        result["available_gpus_after_beta"]
    )

    result["available_power_mw"] = (
        result["available_power_after_beta_mw"]
    )

    result.loc[
        result["site"] == alpha_site,
        "available_gpus"
    ] -= alpha["gpu_count"]

    result.loc[
        result["site"] == alpha_site,
        "available_power_mw"
    ] -= workload_mw(alpha["gpu_count"])

    return result


alpha_at_a = simulate_alpha("Site A")
alpha_at_c = simulate_alpha("Site C")


def gamma_feasible(scenario):
    west = scenario[
        scenario["region"] == gamma["required_region"]
    ]

    feasible = (
        (west["available_gpus"] >= gamma["gpu_count"])
        & (
            west["network_topology"]
            == "High-bandwidth fabric available"
        )
        & (
            west["available_power_mw"]
            >= workload_mw(gamma["gpu_count"])
        )
    )

    return feasible.any()


gamma_if_a = gamma_feasible(alpha_at_a)
gamma_if_c = gamma_feasible(alpha_at_c)

gamma_exposure = (
    gamma["contract_value_m"]
    * gamma["close_probability"]
)

sites["future_energy_score"] = (
    sites["future_power_mw"]
    / sites["future_power_mw"].max()
)

preferred = alpha_at_c.copy()

preferred["scarcity_score"] = (
    1
    - (
        preferred["available_gpus"]
        / preferred["usable_gpu_headroom"]
    )
)

preferred["future_energy_score"] = preferred[
    "site"
].map(
    sites.set_index("site")["future_energy_score"]
)

preferred["expansion_score"] = (
    0.70 * preferred["future_energy_score"]
    + 0.30 * preferred["scarcity_score"]
)

expand_site = preferred.sort_values(
    "expansion_score",
    ascending=False
).iloc[0]["site"]

# ---------------------------------------------------------
# HERO
# ---------------------------------------------------------

st.title("Energy-First AI Capacity Planning")

st.caption("Synthetic scenario | No Crusoe internal data used")

st.markdown(
    """
**Crusoe's infrastructure strategy starts upstream with where scalable
energy can be brought online. This synthetic model explores the next
planning question: once current and future powered-compute capacity is
understood, how should commercial demand be placed today while preserving
constrained optionality and informing future expansion?**
"""
)

st.info(
    """
**Optimize for current feasibility, constrained-capacity optionality,
and energy-backed expansion — not maximum utilization.**
"""
)

# ---------------------------------------------------------
# DIFFERENTIATING FRAMEWORK
# ---------------------------------------------------------

st.subheader("Why the planning problem starts before the GPU")

framework_cols = st.columns(6)

steps = [
    ("1", "Commercial demand"),
    ("2", "Workload requirements"),
    ("3", "**Energy availability**"),
    ("4", "**Powered compute**"),
    ("5", "Future exposure"),
    ("6", "PLACE / PROTECT / EXPAND")
]

for col, (num, label) in zip(framework_cols, steps):
    with col:
        st.markdown(f"### {num}")
        st.markdown(label)

st.markdown(
    """
A technically feasible placement can still be strategically poor if it
consumes scarce powered compute or removes optionality for demand with
fewer alternatives.

**Capacity-planning implication:** future energy availability becomes an input
to where additional powered-compute capacity may be most attractive to expand.
"""
)

# ---------------------------------------------------------
# CAPACITY TENSION
# ---------------------------------------------------------

st.subheader("The decision tension")

left, right = st.columns(2)

with left:
    st.markdown("### Customer Alpha")
    st.write("**2,048 B200 GPUs**")
    st.write("Distributed training")
    st.write("High-bandwidth requirement")
    st.write("**Geography: flexible**")

with right:
    st.markdown("### Customer Gamma")
    st.write("**1,024 B200 GPUs**")
    st.write("Future distributed training opportunity")
    st.write("High-bandwidth requirement")
    st.write("**Geography: West required**")

st.markdown(
    """
Alpha can be served at either Site A or Site C.

Gamma does not have the same flexibility:
its synthetic requirement constrains it to Site A.

The planning question is therefore not simply **where Alpha fits**,
but **what strategically constrained capacity is consumed by placing
Alpha there**.
"""
)

# ---------------------------------------------------------
# COMPARISON
# ---------------------------------------------------------

comparison = pd.DataFrame({
    "Decision factor": [
        "Alpha feasible",
        "West GPUs remaining",
        "Gamma feasible if it closes",
        "Modeled probability-weighted pipeline exposure",
        "West optionality preserved"
    ],
    "Alpha → Site A": [
        "Yes",
        "256",
        "No",
        f"${gamma_exposure:.1f}M",
        "No"
    ],
    "Alpha → Site C": [
        "Yes",
        "2,304",
        "Yes",
        "$0M",
        "Yes"
    ]
})

st.dataframe(
    comparison,
    hide_index=True,
    use_container_width=True
)

# ---------------------------------------------------------
# RECOMMENDATION
# ---------------------------------------------------------

st.subheader("Recommended capacity action")

place_col, protect_col, expand_col = st.columns(3)

with place_col:
    st.markdown("### PLACE")
    st.metric("Customer Alpha", "Site C")
    st.caption(
        "Serve flexible demand without consuming scarce West capacity."
    )

with protect_col:
    st.markdown("### PROTECT")
    st.metric("Strategic capacity", "Site A")
    st.caption(
        "Preserve high-bandwidth West capacity for constrained demand."
    )

with expand_col:
    st.markdown("### EXPAND")
    st.metric("Future priority", expand_site)
    st.caption(
        "Strongest modeled future energy pipeline and expansion optionality."
    )

# ---------------------------------------------------------
# CAPACITY OPTIONALITY VISUAL
# ---------------------------------------------------------

st.subheader("Why placement matters: preserve constrained optionality")

tradeoff = pd.DataFrame({
    "Placement": [
        "Alpha → Site A",
        "Alpha → Site C"
    ],
    "West GPUs Remaining": [
        256,
        2304
    ]
})

fig, ax = plt.subplots(figsize=(8, 4.2))

ax.bar(
    tradeoff["Placement"],
    tradeoff["West GPUs Remaining"]
)

ax.axhline(
    y=1024,
    linestyle="--",
    linewidth=2,
    label="Gamma requirement: 1,024 GPUs"
)

ax.set_ylabel("West-region B200 GPUs remaining")
ax.set_title(
    "West capacity preserved for constrained future demand"
)

ax.legend()
ax.set_ylim(0, 2600)

for i, value in enumerate(
    tradeoff["West GPUs Remaining"]
):
    ax.text(
        i,
        value + 60,
        f"{value:,}",
        ha="center"
    )

plt.tight_layout()
st.pyplot(fig)

st.markdown(
    f"""
The threshold matters more than raw utilization.

- **Alpha → Site A:** 256 West-region GPUs remain, below Gamma's
  1,024-GPU requirement.
- **Alpha → Site C:** 2,304 West-region GPUs remain, preserving Gamma
  feasibility.

Under the synthetic scenario, this preserves approximately
**${gamma_exposure:.1f}M of modeled probability-weighted pipeline exposure**.
"""
)

# ---------------------------------------------------------
# ENERGY-FIRST EXPANSION
# ---------------------------------------------------------

st.subheader("Energy-first expansion signal")

future_energy = sites[
    ["site", "future_power_mw"]
].copy()

fig2, ax2 = plt.subplots(figsize=(8, 3.2))

ax2.bar(
    future_energy["site"],
    future_energy["future_power_mw"]
)

ax2.set_ylabel("Future power pipeline (MW)")
ax2.set_title("Future energy availability by site")

for i, value in enumerate(
    future_energy["future_power_mw"]
):
    ax2.text(
        i,
        value + 3,
        f"{value:.0f} MW",
        ha="center"
    )

plt.tight_layout()
st.pyplot(fig2)

st.markdown(
    """
**Site B becomes the modeled expansion priority.**

It is not the strongest current placement option, but it has the largest
future energy pipeline.

That distinction is central to the framework:

**current placement capacity and future expansion capacity do not have
to point to the same site.**
"""
)

# ---------------------------------------------------------
# CROSS-FUNCTIONAL TRANSLATION
# ---------------------------------------------------------

st.subheader("One decision, two stakeholder views")

business_col, infra_col = st.columns(2)

with business_col:
    st.markdown("### Business view")
    st.write(
        f"""
Alpha can be supported at Site C without compromising the modeled
deployment requirement.

Using Site C preserves the only West-region option for Gamma-like
constrained demand and avoids approximately **${gamma_exposure:.1f}M**
in modeled probability-weighted exposure.

**Commercial implication:** serve current demand while preserving
flexibility for strategically constrained future opportunities.
"""
    )

with infra_col:
    st.markdown("### Infrastructure view")
    st.write(
        """
Site C supports Alpha's 2,048-GPU high-bandwidth requirement.

Site A remains available as a **2,304-GPU West-region high-bandwidth
capacity block**, preserving feasibility for the 1,024-GPU Gamma
requirement.

Site B's **110 MW future energy pipeline** creates the strongest modeled
signal for future powered-compute expansion.
"""
    )

# ---------------------------------------------------------
# MODEL SCOPE
# ---------------------------------------------------------

with st.expander("Model scope and synthetic assumptions"):
    st.write(
        """
This is a synthetic planning-framework exploration designed to demonstrate
decision structure, not reproduce Crusoe's internal planning systems.

All customers, site capacities, energy assumptions, GPU requirements,
network/topology characteristics, contract values, close probabilities,
regional constraints, and scoring weights are hypothetical.

The model intentionally simplifies electrical, network, cooling,
construction, economic, and operational constraints.
"""
    )
