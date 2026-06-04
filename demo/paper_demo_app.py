from __future__ import annotations

import base64
import json
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
DATA = ROOT / "data"

PAPER_TITLE = "Towards Explainable Graph-Verfied Neuro-symbolic Chest X-Ray Report Generation"
SHORT_TITLE = "Nesy-Gen"
AUTHORS = "Faezeh Safari, Hang Dong, Zeyu Fu, Aline Villavicencio"
AFFILIATION = "University of Exeter"

NODE_COLORS = {
    "Indication": "#d8e7ff",
    "Disease": "#ffe2df",
    "Phenotype": "#ddf6e7",
    "Anatomy": "#efe3ff",
}

ABLATION_ROWS = [
    {"dataset": "IU X-ray", "model": "Baseline (Swin+Trans)", "bleu_4": 11.2, "meteor": 15.5, "rouge_l": 31.2, "f1": 30.3},
    {"dataset": "IU X-ray", "model": "+ VECA", "bleu_4": 15.4, "meteor": 18.8, "rouge_l": 34.5, "f1": 34.2},
    {"dataset": "IU X-ray", "model": "+ Temporal ST", "bleu_4": 16.8, "meteor": 19.9, "rouge_l": 36.8, "f1": 36.6},
    {"dataset": "IU X-ray", "model": "+ LTN", "bleu_4": 17.9, "meteor": 20.1, "rouge_l": 37.2, "f1": 37.9},
    {"dataset": "IU X-ray", "model": "Nesy-Gen (Full)", "bleu_4": 18.5, "meteor": 20.8, "rouge_l": 37.9, "f1": 38.7},
    {"dataset": "MIMIC-CXR", "model": "Baseline (Swin+Trans)", "bleu_4": 5.4, "meteor": 11.8, "rouge_l": 22.4, "f1": 26.7},
    {"dataset": "MIMIC-CXR", "model": "+ VECA", "bleu_4": 8.2, "meteor": 14.5, "rouge_l": 27.2, "f1": 32.3},
    {"dataset": "MIMIC-CXR", "model": "+ Temporal ST", "bleu_4": 9.9, "meteor": 22.0, "rouge_l": 28.5, "f1": 36.8},
    {"dataset": "MIMIC-CXR", "model": "+ LTN", "bleu_4": 10.2, "meteor": 26.8, "rouge_l": 29.4, "f1": 39.8},
    {"dataset": "MIMIC-CXR", "model": "Nesy-Gen (Full)", "bleu_4": 9.8, "meteor": 27.4, "rouge_l": 30.2, "f1": 43.1},
]

HERO_METRICS = {
    "iu_bleu4": "18.5",
    "iu_f1": "38.7",
    "mimic_meteor": "27.4",
    "mimic_f1": "43.1",
}


def inject_css() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(16,185,129,0.12), transparent 24%),
                radial-gradient(circle at top right, rgba(59,130,246,0.12), transparent 22%),
                linear-gradient(180deg, #f8fafc 0%, #eef2ff 100%);
        }
        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 2rem;
            max-width: 1380px;
        }
        .hero-card {
            background: linear-gradient(135deg, #0f172a 0%, #153e75 100%);
            border: 1px solid rgba(255,255,255,0.12);
            border-radius: 28px;
            padding: 1.4rem 1.6rem;
            color: #f8fafc;
            box-shadow: 0 20px 60px rgba(15,23,42,0.18);
            margin-bottom: 1rem;
        }
        .hero-title {
            font-size: 2.2rem;
            line-height: 1.05;
            font-weight: 800;
            margin-bottom: 0.5rem;
        }
        .hero-authors {
            font-size: 1rem;
            color: #bfdbfe;
            margin-top: 0.7rem;
            font-weight: 600;
        }
        .hero-subtitle {
            font-size: 1rem;
            color: #dbeafe;
            max-width: 900px;
        }
        .hero-grid {
            display: grid;
            grid-template-columns: minmax(0, 1fr) 220px;
            gap: 1rem;
            align-items: center;
        }
        .hero-logo-wrap {
            display: flex;
            justify-content: center;
            align-items: center;
        }
        .hero-logo {
            width: 100%;
            max-width: 210px;
            background: rgba(255,255,255,0.95);
            border-radius: 18px;
            padding: 0.6rem 0.8rem;
        }
        .badge-row {
            display: flex;
            gap: 0.6rem;
            flex-wrap: wrap;
            margin-top: 1rem;
        }
        .badge {
            background: rgba(255,255,255,0.1);
            border: 1px solid rgba(255,255,255,0.16);
            border-radius: 999px;
            padding: 0.35rem 0.8rem;
            font-size: 0.9rem;
            font-weight: 600;
        }
        .section-card {
            background: rgba(255,255,255,0.82);
            border: 1px solid rgba(148,163,184,0.22);
            border-radius: 24px;
            padding: 1rem 1.1rem;
            box-shadow: 0 10px 30px rgba(15,23,42,0.06);
            margin-bottom: 1rem;
        }
        .mini-card {
            background: linear-gradient(180deg, rgba(255,255,255,0.95), rgba(248,250,252,0.95));
            border: 1px solid rgba(148,163,184,0.22);
            border-radius: 20px;
            padding: 0.95rem 1rem;
            min-height: 135px;
        }
        .mini-title {
            color: #475569;
            font-size: 0.86rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.35rem;
            font-weight: 700;
        }
        .mini-value {
            color: #0f172a;
            font-size: 1.85rem;
            font-weight: 800;
            margin-bottom: 0.35rem;
        }
        .timeline-step {
            border-left: 5px solid #0ea5a4;
            padding-left: 0.9rem;
            margin-bottom: 0.85rem;
        }
        .callout {
            background: linear-gradient(90deg, rgba(14,165,164,0.12), rgba(37,99,235,0.08));
            border: 1px solid rgba(14,165,164,0.25);
            border-radius: 18px;
            padding: 0.85rem 1rem;
            margin: 0.6rem 0 1rem;
        }
        .presenter-box {
            background: rgba(15,23,42,0.92);
            color: #f8fafc;
            border-radius: 18px;
            padding: 0.95rem 1rem;
            border: 1px solid rgba(148,163,184,0.18);
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.4rem;
        }
        .stTabs [data-baseweb="tab"] {
            background: rgba(255,255,255,0.7);
            border-radius: 999px;
            padding: 0.35rem 0.9rem;
        }
        .footer-note {
            color: #475569;
            font-size: 0.92rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def load_cases() -> list[dict]:
    return json.loads((DATA / "sample_cases.json").read_text())


def accepted_claims(case: dict, beta: float, delta: float, epsilon: float) -> tuple[list[dict], list[dict]]:
    accepted, rejected = [], []
    for claim in case["claims"]:
        is_accepted = claim["ltn"] >= beta and claim["grounding"] >= delta and claim["hallucination_risk"] < epsilon
        payload = {**claim, "status": "accepted" if is_accepted else "rejected"}
        (accepted if is_accepted else rejected).append(payload)
    return accepted, rejected


def draw_graph(case: dict, large: bool = False) -> plt.Figure:
    graph = nx.DiGraph()
    for node in case["nodes"]:
        graph.add_node(node["id"], **node)
    for edge in case["edges"]:
        graph.add_edge(edge["source"], edge["target"], **edge)

    pos = nx.spring_layout(graph, seed=17, k=1.55 if large else 1.3)
    fig, ax = plt.subplots(figsize=(11.5, 6.8) if large else (10.2, 5.8))
    ax.set_facecolor("#f8fafc")

    for node_type, color in NODE_COLORS.items():
        nodes = [n for n, d in graph.nodes(data=True) if d["type"] == node_type]
        if nodes:
            nx.draw_networkx_nodes(
                graph,
                pos,
                nodelist=nodes,
                node_color=color,
                edgecolors="#334155",
                node_size=4800 if large else 4000,
                linewidths=2.2,
                ax=ax,
            )

    nx.draw_networkx_edges(
        graph,
        pos,
        width=3.2 if large else 2.8,
        edge_color="#94a3b8",
        arrows=True,
        arrowstyle="-|>",
        arrowsize=20,
        ax=ax,
    )
    nx.draw_networkx_labels(
        graph,
        pos,
        labels={n: d["label"] for n, d in graph.nodes(data=True)},
        font_size=12 if large else 11,
        font_weight="bold",
        ax=ax,
    )
    nx.draw_networkx_edge_labels(
        graph,
        pos,
        edge_labels={(u, v): f'{d["relation"]}\nw={d["weight"]:.2f}' for u, v, d in graph.edges(data=True)},
        font_size=9 if large else 8.5,
        label_pos=0.52,
        ax=ax,
    )
    ax.set_title("Patient-Specific Temporal Steiner Subgraph", fontsize=17 if large else 15, weight="bold", color="#0f172a")
    ax.axis("off")
    fig.tight_layout()
    return fig


def plot_ablation(dataset: str, metric: str) -> plt.Figure:
    frame = pd.DataFrame(ABLATION_ROWS)
    subset = frame[frame["dataset"] == dataset]
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    colors = ["#cbd5e1", "#93c5fd", "#60a5fa", "#34d399", "#0f766e"]
    bars = ax.bar(subset["model"], subset[metric], color=colors, edgecolor="#0f172a", linewidth=0.4)
    ax.set_ylabel(metric.replace("_", " ").upper())
    ax.set_title(f"{dataset}: {metric.replace('_', ' ').title()} by Module", fontsize=14, weight="bold")
    ax.set_ylim(0, max(subset[metric]) * 1.25)
    ax.tick_params(axis="x", rotation=18)
    ax.spines[["top", "right"]].set_visible(False)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.25, f"{height:.1f}", ha="center", va="bottom", fontsize=9, weight="bold")
    fig.tight_layout()
    return fig


def token_table(case: dict, beta: float, delta: float, epsilon: float) -> pd.DataFrame:
    accepted, rejected = accepted_claims(case, beta, delta, epsilon)
    frame = pd.DataFrame(accepted + rejected)
    if frame.empty:
        return frame
    return frame[["text", "status", "entailment", "grounding", "ltn", "hallucination_risk", "reason"]]


def threshold_margin(claim: dict, beta: float, delta: float, epsilon: float) -> float:
    ltn_margin = claim["ltn"] - beta
    grounding_margin = claim["grounding"] - delta
    risk_margin = epsilon - claim["hallucination_risk"]
    return min(ltn_margin, grounding_margin, risk_margin)


def threshold_sensitivity(case: dict, beta: float, delta: float, epsilon: float) -> pd.DataFrame:
    rows = []
    for claim in case["claims"]:
        margin = threshold_margin(claim, beta, delta, epsilon)
        rows.append(
            {
                "text": claim["text"],
                "margin": margin,
                "status": "accepted" if margin >= 0 else "rejected",
            }
        )
    frame = pd.DataFrame(rows).sort_values("margin")
    return frame


def render_header(subtitle: str) -> None:
    logo_path = ASSETS / "university_of_exeter_logo.png"
    logo_b64 = base64.b64encode(logo_path.read_bytes()).decode("ascii")
    st.markdown(
        f"""
        <div class="hero-card">
            <div class="hero-grid">
                <div>
                    <div class="hero-title">{PAPER_TITLE}</div>
                    <div class="hero-subtitle">{subtitle}</div>
                    <div class="hero-authors">{AUTHORS}<br>{AFFILIATION}</div>
                </div>
                <div class="hero-logo-wrap">
                    <img class="hero-logo" src="data:image/png;base64,{logo_b64}" alt="University of Exeter logo">
                </div>
            </div>
            <div class="badge-row">
                <div class="badge">{SHORT_TITLE}</div>
                <div class="badge">Ante-hoc verification</div>
                <div class="badge">PrimeKG-guided reasoning</div>
                <div class="badge">Hallucination suppression</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_overview() -> None:
    render_header("A polished, conference-facing walkthrough of the method, its logic gate, and its reported gains.")

    stats = st.columns(4)
    stats[0].markdown(f'<div class="mini-card"><div class="mini-title">IU X-ray BLEU-4</div><div class="mini-value">{HERO_METRICS["iu_bleu4"]}</div><div>Full model result reported in the manuscript.</div></div>', unsafe_allow_html=True)
    stats[1].markdown(f'<div class="mini-card"><div class="mini-title">IU Entity F1</div><div class="mini-value">{HERO_METRICS["iu_f1"]}</div><div>Clinical concept recovery after verification.</div></div>', unsafe_allow_html=True)
    stats[2].markdown(f'<div class="mini-card"><div class="mini-title">MIMIC METEOR</div><div class="mini-value">{HERO_METRICS["mimic_meteor"]}</div><div>Best reported fluency gain in the paper.</div></div>', unsafe_allow_html=True)
    stats[3].markdown(f'<div class="mini-card"><div class="mini-title">MIMIC Entity F1</div><div class="mini-value">{HERO_METRICS["mimic_f1"]}</div><div>Strongest signal of factual clinical utility.</div></div>', unsafe_allow_html=True)

    left, right = st.columns([1.1, 0.9])
    with left:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Narrative in one sentence")
        st.write(
            "Nesy-Gen generates a report only after visual evidence, graph reachability, and neuro-symbolic logic agree that a claim is medically supportable."
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Method at a glance")
        st.markdown(
            """
            <div class="timeline-step"><strong>1. VECA</strong><br>Align image patches with indication entities.</div>
            <div class="timeline-step"><strong>2. Temporal Steiner Tree</strong><br>Extract a patient-specific causal subgraph.</div>
            <div class="timeline-step"><strong>3. LTN constraints</strong><br>Score biological plausibility and relation validity.</div>
            <div class="timeline-step"><strong>4. Consistency gate</strong><br>Reject unsupported tokens before final decoding.</div>
            """,
            unsafe_allow_html=True,
        )
        st.image(str(ASSETS / "fig4.png"), caption="PrimeKG-style reasoning graph used in the demo")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    tabs = st.tabs(["Problem", "Contributions", "Why it matters"])
    with tabs[0]:
        st.write(
            "Conventional report generators can sound fluent while hallucinating unsupported findings. In chest X-ray reporting, that is not a style issue — it is a clinical reliability issue."
        )
    with tabs[1]:
        st.write(
            "The manuscript contributes a dual-path system: a visual branch for image evidence and a neuro-symbolic branch for graph-grounded verification, tied together by a consistency gate."
        )
    with tabs[2]:
        st.write(
            "For a conference audience, the biggest hook is that the system does not only generate text better — it explains why a claim was allowed or rejected."
        )
    st.markdown('</div>', unsafe_allow_html=True)


def render_interactive(cases: list[dict]) -> None:
    render_header("Live conference walkthrough: show how verified findings survive the gate while hallucinated phrases are suppressed.")

    with st.sidebar:
        st.subheader("Presentation controls")
        case = st.selectbox("Sample case", cases, format_func=lambda row: f'{row["title"]} ({row["dataset"]})')
        beta = st.slider("LTN acceptance β", 0.4, 0.9, 0.65, 0.01)
        delta = st.slider("Grounding minimum δ", 0.1, 0.8, 0.30, 0.01)
        epsilon = st.slider("Hallucination rejection ε", 0.1, 0.95, 0.50, 0.01)
        presenter_mode = st.toggle("Presenter notes", value=True)

    accepted, rejected = accepted_claims(case, beta, delta, epsilon)
    accepted_text = ". ".join(item["text"] for item in accepted) + "." if accepted else "No claims passed the gate."
    rejected_text = ", ".join(item["text"] for item in rejected) if rejected else "None"

    top = st.columns([1.1, 0.9])
    with top[0]:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader(case["title"])
        st.markdown(f"**Dataset**: `{case['dataset']}`")
        st.markdown(f"**Indication**: `{case['indication']}`")
        st.markdown(f"**Visual summary**: {case['visual_summary']}")
        st.pyplot(draw_graph(case, large=True), clear_figure=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with top[1]:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Gate outcome")
        metric_cols = st.columns(2)
        metric_cols[0].metric("Accepted", len(accepted))
        metric_cols[1].metric("Rejected", len(rejected))
        metric_cols = st.columns(2)
        mean_ltn = sum(x["ltn"] for x in accepted) / max(len(accepted), 1)
        rejection_rate = 100 * len(rejected) / max(len(accepted) + len(rejected), 1)
        metric_cols[0].metric("Mean accepted σ_LTN", f"{mean_ltn:.2f}")
        metric_cols[1].metric("Rejection rate", f"{rejection_rate:.1f}%")

        st.markdown("**Verified report**")
        st.success(accepted_text)
        st.markdown("**Unverified baseline**")
        st.error(case["baseline_report"])
        st.markdown("**Suppressed claims**")
        st.write(rejected_text)
        st.markdown('</div>', unsafe_allow_html=True)

    sensitivity = threshold_sensitivity(case, beta, delta, epsilon)
    borderline = sensitivity[sensitivity["margin"].abs() <= 0.08]

    mid = st.columns([0.9, 1.1])
    with mid[0]:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Threshold-sensitive claims")
        if borderline.empty:
            st.info("No claims are close to the current thresholds. Try moving β, δ, or ε further.")
        else:
            st.dataframe(
                borderline.style.applymap(
                    lambda value: "background-color: #dcfce7;" if value == "accepted" else ("background-color: #fee2e2;" if value == "rejected" else ""),
                    subset=["status"],
                ).format({"margin": "{:.2f}"}),
                use_container_width=True,
            )
        st.caption("`margin` shows how close a claim is to flipping status under the current gate.")
        st.markdown('</div>', unsafe_allow_html=True)
    with mid[1]:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Threshold effect view")
        st.bar_chart(
            sensitivity.set_index("text")[["margin"]],
            horizontal=True,
            use_container_width=True,
        )
        st.caption("Bars crossing above or below zero indicate claims that enter or leave the verified report.")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Token-level verification table")
    frame = token_table(case, beta, delta, epsilon)
    st.dataframe(
        frame.style.applymap(
            lambda value: "background-color: #dcfce7;" if value == "accepted" else ("background-color: #fee2e2;" if value == "rejected" else ""),
            subset=["status"],
        ).format(
            {"entailment": "{:.2f}", "grounding": "{:.2f}", "ltn": "{:.2f}", "hallucination_risk": "{:.2f}"}
        ),
        use_container_width=True,
    )
    st.caption(
        "Demo gate rule: accept a candidate phrase only if `σ_LTN >= β`, `grounding >= δ`, and `hallucination risk < ε`."
    )
    st.markdown('</div>', unsafe_allow_html=True)


def render_results() -> None:
    render_header("Conference results view: clean ablations, headline metrics, and a short story about where the gains come from.")

    left, right = st.columns([0.85, 1.15])
    with left:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        dataset = st.selectbox("Dataset", ["IU X-ray", "MIMIC-CXR"])
        metric = st.selectbox("Metric", ["bleu_4", "meteor", "rouge_l", "f1"], format_func=lambda x: x.replace("_", " ").upper())
        st.pyplot(plot_ablation(dataset, metric), clear_figure=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown(
            """
            - `VECA` improves alignment between the indication and the image evidence.
            - `Temporal ST` strengthens clinically coherent relation chains.
            - `LTN` improves trustworthiness by filtering biologically implausible content.
            - The full model performs best overall, especially on entity-level clinical `F1`.
            """
        )
        st.dataframe(pd.DataFrame(ABLATION_ROWS), use_container_width=True, height=350)
        st.markdown('</div>', unsafe_allow_html=True)


def render_figures() -> None:
    render_header("Poster-friendly figure browser: useful for answering questions quickly during a session.")

    tabs = st.tabs(["Token comparison", "Hallucination reduction", "Subgraph figure"])
    with tabs[0]:
        st.image(str(ASSETS / "fig3.png"), caption="Verified vs unverified token-level comparison from the manuscript")
    with tabs[1]:
        st.image(str(ASSETS / "fig2.png"), caption="Illustration of hallucination detection and token rejection")
    with tabs[2]:
        st.image(str(ASSETS / "fig4.png"), caption="Temporal Steiner subgraph used as the reasoning scaffold")


def main() -> None:
    st.set_page_config(page_title=f"{SHORT_TITLE} Conference Demo", page_icon="🩻", layout="wide")
    inject_css()
    st.sidebar.title(f"{SHORT_TITLE} Demo")
    page = st.sidebar.radio(
        "Navigate",
        ["Overview", "Interactive verifier", "Results dashboard", "Paper figures"],
    )
    st.sidebar.caption("Conference-focused demo build")

    cases = load_cases()
    if page == "Overview":
        render_overview()
    elif page == "Interactive verifier":
        render_interactive(cases)
    elif page == "Results dashboard":
        render_results()
    else:
        render_figures()


if __name__ == "__main__":
    main()
