from pathlib import Path

import pandas as pd
import streamlit as st

from src.causal import estimate_randomized_uplift
from src.features import customer_features
from src.models import add_demo_target, train_retention_model
from src.rag import GroundedAssistant
from src.recommender import ItemRecommender

ROOT = Path(__file__).parent
st.set_page_config(page_title="AI Growth Platform", page_icon="📈", layout="wide")
st.title("AI Growth & Personalization Platform")
st.caption("Recommendations, retention intelligence, causal measurement, and grounded business insights")

events_path, experiment_path = ROOT / "data/events.csv", ROOT / "data/experiment.csv"
if not events_path.exists():
    st.error("Generate demo data first: python scripts/generate_data.py")
    st.stop()
events, experiment = pd.read_csv(events_path), pd.read_csv(experiment_path)
features = add_demo_target(customer_features(events))
model, auc = train_retention_model(features)
features["churn_risk"] = model.predict_proba(features[["interactions", "unique_items", "engagement_score", "recency_days", "avg_event_value"]])[:, 1]
uplift = estimate_randomized_uplift(experiment)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Customers", f"{events.customer_id.nunique():,}")
c2.metric("Interactions", f"{len(events):,}")
c3.metric("Retention model AUC", f"{auc:.2f}")
c4.metric("Campaign uplift", f"{uplift.absolute_uplift:.1%}")

tab1, tab2, tab3, tab4 = st.tabs(["Retention", "Recommendations", "Experiment", "AI Assistant"])
with tab1:
    st.subheader("Customers requiring attention")
    st.dataframe(features.sort_values("churn_risk", ascending=False).head(20), use_container_width=True)
with tab2:
    customer = st.selectbox("Customer", sorted(events.customer_id.unique())[:200])
    recommendations = ItemRecommender().fit(events).recommend(customer)
    st.dataframe(pd.DataFrame(recommendations), use_container_width=True, hide_index=True)
with tab3:
    st.write(f"Treatment conversion: **{uplift.treatment_rate:.1%}**")
    st.write(f"Control conversion: **{uplift.control_rate:.1%}**")
    st.write(f"95% CI for absolute uplift: **[{uplift.ci_low:.1%}, {uplift.ci_high:.1%}]**")
    st.write(f"p-value: **{uplift.p_value:.3f}**")
    if uplift.ci_low <= 0 <= uplift.ci_high:
        st.warning("The interval includes zero. Collect more data before claiming a causal effect.")
    else:
        st.success("The estimated effect is statistically distinguishable from zero at the 5% level.")
with tab4:
    question = st.text_input("Ask about KPIs, experiments, or responsible use", "How should campaign uplift be interpreted?")
    if question:
        for result in GroundedAssistant(ROOT / "knowledge").retrieve(question):
            with st.expander(f"Source: {result['source']} · relevance {result['score']}"):
                st.write(result["content"])

