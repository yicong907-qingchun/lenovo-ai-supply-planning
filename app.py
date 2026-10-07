from pathlib import Path

import pandas as pd
import streamlit as st
from openai import OpenAI

st.set_page_config(
    page_title="AI Supply Planning Copilot",
    page_icon="🤖",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
CONTEXT_FILE = BASE_DIR / "AI_Planning_Context.csv"
SNAPSHOT_FILE = BASE_DIR / "PowerBI_AI_Planning_Snapshot.csv"

PLANNING_DATE = "2026-10-03"
MODEL_NAME = "gpt-6-luna"

@st.cache_data
def load_context():
    if not CONTEXT_FILE.exists():
        raise FileNotFoundError(
            f"Missing file: {CONTEXT_FILE.name}. "
            "Please keep it in the same folder as app.py."
        )

    df = pd.read_csv(CONTEXT_FILE, keep_default_na=False)

    if "Material DOS" in df.columns:
        dos_num = pd.to_numeric(df["Material DOS"], errors="coerce").round(1)
        df["Material DOS"] = dos_num.where(dos_num.notna(), "N/A")

    return df


@st.cache_data
def load_snapshot():
    if not SNAPSHOT_FILE.exists():
        return None

    return pd.read_csv(SNAPSHOT_FILE)


try:
    ai_context = load_context()
    planning = load_snapshot()
except Exception as e:
    st.error(f"Data loading failed: {e}")
    st.stop()

planning_json = ai_context.to_json(
    orient="records",
    force_ascii=False,
    indent=2,
)

if planning is not None:
    total_demand = float(
        pd.to_numeric(planning["Total Demand Qty"], errors="coerce").fillna(0).sum()
    )
    total_gap = float(
        pd.to_numeric(planning["Supply Gap Qty"], errors="coerce").fillna(0).sum()
    )
    shortage_count = int(
        planning.loc[
            pd.to_numeric(planning["Supply Gap Qty"], errors="coerce").fillna(0) > 0,
            "MaterialID",
        ].nunique()
    )
    overall_coverage = (
        (1 - total_gap / total_demand) * 100 if total_demand else 0.0
    )
else:
    total_demand = 16839.0
    total_gap = 1744.0
    shortage_count = 12
    overall_coverage = 89.6

overall_context = f"""
Planning Snapshot Date: {PLANNING_DATE}
Planning Scope: Full future planning horizon

Total Demand Qty: {total_demand:.0f}
Total Supply Gap Qty: {total_gap:.0f}
Shortage Material Count: {shortage_count}
Overall Supply Coverage: {overall_coverage:.1f}%
""".strip()

st.title("🤖 AI Supply Planning Copilot")
st.caption(
    f"Planning Snapshot: {PLANNING_DATE} | "
    "Scope: Full Future Planning Horizon"
)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Demand", f"{total_demand:,.0f}")
k2.metric("Supply Gap", f"{total_gap:,.0f}")
k3.metric("Shortage Materials", f"{shortage_count}")
k4.metric("Supply Coverage", f"{overall_coverage:.1f}%")

with st.expander("⚙️ API Settings", expanded=False):
    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="sk-...",
        key="openai_api_key",
        help="The key is used only to call the OpenAI API from this running app.",
    )

    st.caption("The API is called only when you send a question.")

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

instructions = f"""
You are an AI Supply Planning Copilot for a server supply chain planning team.

The planning snapshot date is {PLANNING_DATE}.
The analysis covers the full future planning horizon based on information
known at that snapshot date.

Business definitions:
- Total Demand Qty: total material demand across the planning horizon.
- Available Inventory Qty: inventory available at the planning snapshot.
- On-Time Committed Qty: supplier committed quantity expected to arrive
  on time according to the Power BI planning model.
- Supply Gap Qty: demand not covered by available inventory and on-time supply.
- Supply Coverage %: percentage of material demand currently covered.
- Material DOS: days of supply. If it is N/A, do not estimate it.
- OpenPOQty: outstanding purchase order quantity already placed with suppliers.
- TotalCommittedQty_RAW: total supplier commitment recorded in SRM.
- UncommittedPOQty: open PO quantity not yet covered by supplier commitment.
- LeadTimeDays: replenishment lead time.
- CommonSKUCount: number of server products that use the material.

Analysis rules:
1. Use only the supplied planning data.
2. Never invent suppliers, quantities, dates, alternative materials, or business facts.
3. Do not rank risk only by Supply Gap Qty.
4. Consider gap, coverage, DOS, lead time, uncommitted PO, and CommonSKUCount together.
5. Treat On-Time Committed Qty from Power BI as authoritative for on-time supply.
6. Clearly distinguish observed facts from recommendations.
7. If evidence is insufficient, explicitly say so.
8. Recommendations may include supplier commitment confirmation, expediting supply,
   allocation prioritization, monitoring, or evaluating qualified alternatives
   only when the supplied data supports that option.
9. Answer in Chinese unless the user explicitly requests English.
10. Be concise and planner-oriented. Prefer tables when comparing multiple materials.
""".strip()

if "messages" not in st.session_state:
    st.session_state.messages = []

st.markdown("#### 💡 Suggested Questions")

q1, q2, q3 = st.columns(3)
suggested_question = None

with q1:
    if st.button("🔴 Top 3 Supply Risks", use_container_width=True):
        suggested_question = (
            "请识别目前最需要优先处理的三个供应风险，"
            "并说明数据依据、风险原因和建议行动。"
        )

with q2:
    if st.button("🧩 Xeon 6767P Risk", use_container_width=True):
        suggested_question = (
            "为什么 Intel Xeon 6767P 值得重点关注？"
            "请引用具体数据解释。"
        )

with q3:
    if st.button("⏳ Long Lead-Time Risks", use_container_width=True):
        suggested_question = (
            "哪些物料虽然 Supply Gap 不大，"
            "但因为 Lead Time 长、Coverage 低或影响多个 SKU，"
            "仍然存在较高供应风险？"
        )

st.divider()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

typed_question = st.chat_input(
    "例如：目前最需要优先处理的供应风险是什么？"
)

question = suggested_question or typed_question

if question:
    if not api_key:
        st.warning("请先展开“API Settings”并输入 OpenAI API Key。")
        st.stop()

    prior_messages = st.session_state.messages[-6:]
    conversation_history = "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in prior_messages
    )

    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    user_input = f"""
Overall planning context:
{overall_context}

Shortage material details:
{planning_json}

Recent conversation history:
{conversation_history if conversation_history else "None"}

Planner question:
{question}
""".strip()

    client = OpenAI(api_key=api_key)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing supply risks..."):
            try:
                response = client.responses.create(
                    model=MODEL_NAME,
                    instructions=instructions,
                    input=user_input,
                )
                answer = response.output_text
                st.markdown(answer)
            except Exception as e:
                answer = (
                    "API call failed. Please check the API key, network connection, "
                    f"and model access.\n\nError: `{type(e).__name__}: {e}`"
                )
                st.error(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )

st.caption(
    "Prototype architecture: Power BI planning results + ERP/SRM context → "
    "Python → OpenAI Responses API → planner diagnosis and recommendations."
)
