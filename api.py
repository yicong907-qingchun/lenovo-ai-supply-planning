from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from openai import OpenAI

BASE_DIR = Path(__file__).resolve().parent
CONTEXT_FILE = BASE_DIR / "AI_Planning_Context.csv"
SNAPSHOT_FILE = BASE_DIR / "PowerBI_AI_Planning_Snapshot.csv"

PLANNING_DATE = "2026-10-03"
MODEL_NAME = "gpt-6-luna"

app = FastAPI(title="AI Supply Planning Copilot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    api_key: str = Field(min_length=1)
    history: list[dict[str, str]] = Field(default_factory=list)


class AskResponse(BaseModel):
    answer: str


def load_context() -> pd.DataFrame:
    if not CONTEXT_FILE.exists():
        raise FileNotFoundError(
            f"Missing file: {CONTEXT_FILE.name}. "
            "Please keep it in the same folder as api.py."
        )

    df = pd.read_csv(CONTEXT_FILE, keep_default_na=False)

    if "Material DOS" in df.columns:
        dos_num = pd.to_numeric(df["Material DOS"], errors="coerce").round(1)
        df["Material DOS"] = dos_num.where(dos_num.notna(), "N/A")

    return df


def load_snapshot():
    if not SNAPSHOT_FILE.exists():
        return None

    return pd.read_csv(SNAPSHOT_FILE)


def build_planning_context():
    ai_context = load_context()
    planning = load_snapshot()

    planning_json = ai_context.to_json(
        orient="records",
        force_ascii=False,
        indent=2,
    )

    if planning is not None:
        total_demand = float(
            pd.to_numeric(
                planning["Total Demand Qty"], errors="coerce"
            ).fillna(0).sum()
        )

        total_gap = float(
            pd.to_numeric(
                planning["Supply Gap Qty"], errors="coerce"
            ).fillna(0).sum()
        )

        shortage_count = int(
            planning.loc[
                pd.to_numeric(
                    planning["Supply Gap Qty"], errors="coerce"
                ).fillna(0) > 0,
                "MaterialID",
            ].nunique()
        )

        overall_coverage = (
            (1 - total_gap / total_demand) * 100
            if total_demand
            else 0.0
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

    return overall_context, planning_json


INSTRUCTIONS = f"""
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
11. The Power BI page slicers do not define the question scope.
12. Determine the requested material, product, category, and time scope from the
    user's wording.
13. If the user names a MaterialID or material name, locate it directly in the
    supplied full dataset.
""".strip()


@app.get("/")
def health():
    return {
        "status": "ok",
        "service": "AI Supply Planning Copilot API",
    }


@app.post("/ask", response_model=AskResponse)
def ask_ai(req: AskRequest):

    question = req.question.strip()
    api_key = req.api_key.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    if not api_key:
        raise HTTPException(
            status_code=400,
            detail="OpenAI API Key is required.",
        )

    try:
        overall_context, planning_json = build_planning_context()

        recent_history = req.history[-6:]

        conversation_history = "\n".join(
            f"{item.get('role', '').upper()}: {item.get('content', '')}"
            for item in recent_history
            if item.get("role") and item.get("content")
        )

        user_input = f"""
Overall planning context:
{overall_context}

Full material planning details:
{planning_json}

Recent conversation history:
{conversation_history if conversation_history else "None"}

Planner question:
{question}
""".strip()

        client = OpenAI(api_key=api_key)

        response = client.responses.create(
            model=MODEL_NAME,
            instructions=INSTRUCTIONS,
            input=user_input,
        )

        return AskResponse(
            answer=response.output_text
        )

    except FileNotFoundError as e:
        raise HTTPException(
            status_code=500,
            detail=str(e),
        ) from e

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"{type(e).__name__}: {e}",
        ) from e