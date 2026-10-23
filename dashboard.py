"""Local dashboard for comparing saved metrics."""

import pandas as pd
import streamlit as st

from litellm_lab.history import default_db_path, encode, load

st.set_page_config(page_title="LiteLLM Lab", page_icon="🧪", layout="wide")
st.title("LiteLLM Lab")
st.caption("Local benchmark history. Prompts and model responses are not stored.")
path = st.sidebar.text_input("History database", str(default_db_path()))
limit = st.sidebar.slider("Recent runs", 10, 5000, 100, 10)
rows = load(path, limit)

if not rows:
    st.info("No history yet. Run litellm-lab bench or litellm-lab compare first.")
    st.stop()

data = pd.DataFrame(rows)
data["started_at"] = pd.to_datetime(data["started_at"], utc=True, errors="coerce")
models = sorted(data["model"].dropna().unique().tolist())
selected = st.sidebar.multiselect("Models", models, default=models)
data = data[data["model"].isin(selected)]
if data.empty:
    st.warning("Select at least one model.")
    st.stop()

summary = data.groupby("model", as_index=False).agg(
    runs=("id", "count"), p50_seconds=("p50_latency_seconds", "mean"),
    p95_seconds=("p95_latency_seconds", "mean"), ttft_seconds=("mean_ttft_seconds", "mean"),
    requests_per_second=("requests_per_second", "mean"),
    succeeded=("succeeded", "sum"), attempted=("requests", "sum"))
summary["success_rate_percent"] = 100 * summary["succeeded"] / summary["attempted"]

st.subheader("Model comparison")
a, b = st.columns(2)
with a:
    st.caption("Average p50 / p95 latency (seconds)")
    st.bar_chart(summary.set_index("model")[["p50_seconds", "p95_seconds"]])
    st.caption("Average requests per second")
    st.bar_chart(summary.set_index("model")[["requests_per_second"]])
with b:
    st.caption("Average time to first content token (seconds)")
    st.bar_chart(summary.set_index("model")[["ttft_seconds"]])
    st.caption("Successful requests (%)")
    st.bar_chart(summary.set_index("model")[["success_rate_percent"]])

st.subheader("Run history")
shown = data.sort_values(["started_at", "id"], ascending=False).copy()
shown["started_at"] = shown["started_at"].dt.strftime("%Y-%m-%d %H:%M:%S UTC")
st.dataframe(shown, use_container_width=True, hide_index=True)
records = shown.to_dict(orient="records")
c1, c2 = st.columns(2)
with c1:
    st.download_button("Download CSV", encode(records, "csv"),
                       file_name="litellm-benchmark-history.csv", mime="text/csv")
with c2:
    st.download_button("Download JSON", encode(records, "json"),
                       file_name="litellm-benchmark-history.json", mime="application/json")
