
import json
import pandas as pd
#from openai import OpenAI
from ai_config import get_ai_client, MODEL_NAME

# =====================================
# 1. BUILD ECONOMIC ANALYSIS TOOLS
# =====================================

def build_tools(df: pd.DataFrame):
    """Create analytical functions using the dataset."""

    data = df.copy()
    data["Quarter"] = data["Quarter"].astype(str)

    data["YoY_Growth"] = (
        data["GDP"].pct_change(4) * 100
    )

    data = data.set_index("Quarter")

    def list_quarters() -> dict:
        """List available quarters."""
        return {"quarters": list(data.index)}

    def get_gdp(quarter: str) -> dict:
        """Get GDP for a specified quarter."""

        if quarter not in data.index:
            return {
                "error": f"Quarter {quarter} not found",
                "available": list(data.index)
            }

        return {
            "quarter": quarter,
            "gdp": float(data.loc[quarter, "GDP"])
        }

    def get_yoy_growth(quarter: str) -> dict:
        """Get YoY GDP growth for a quarter."""

        if quarter not in data.index:
            return {
                "error": f"Quarter {quarter} not found",
                "available": list(data.index)
            }

        value = data.loc[quarter, "YoY_Growth"]

        if pd.isna(value):
            return {
                "error": "Insufficient historical data"
            }

        return {
            "quarter": quarter,
            "yoy_growth_pct": round(float(value), 2)
        }

    def compare_quarters(
        quarter_a: str,
        quarter_b: str
    ) -> dict:
        """Compare GDP levels between two quarters."""

        for q in (quarter_a, quarter_b):
            if q not in data.index:
                return {
                    "error": f"Quarter {q} not found",
                    "available": list(data.index)
                }

        a = float(data.loc[quarter_a, "GDP"])
        b = float(data.loc[quarter_b, "GDP"])

        return {
            "from": quarter_a,
            "to": quarter_b,
            "gdp_from": a,
            "gdp_to": b,
            "pct_change": round(
                (b - a) / a * 100, 2
            )
        }

    def get_extreme_growth(kind: str) -> dict:
        """Find highest or lowest YoY GDP growth."""

        if kind not in ("highest", "lowest"):
            return {
                "error": "kind must be highest or lowest"
            }

        series = data["YoY_Growth"].dropna()

        if series.empty:
            return {
                "error": "No YoY growth data available"
            }

        q = (
            series.idxmax()
            if kind == "highest"
            else series.idxmin()
        )

        return {
            "quarter": str(q),
            "yoy_growth_pct": round(float(series[q]), 2)
        }

    # Map function names to actual Python functions
    return {
        "list_quarters": list_quarters,
        "get_gdp": get_gdp,
        "get_yoy_growth": get_yoy_growth,
        "compare_quarters": compare_quarters,
        "get_extreme_growth": get_extreme_growth
    }


# =====================================
# 2. DEFINE OPENAI TOOL SCHEMAS
# =====================================

TOOLS = [
    {
        "type": "function",
        "name": "list_quarters",
        "description": "List available GDP quarters.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False
        },
        "strict": True
    },
    {
        "type": "function",
        "name": "get_gdp",
        "description": "Get GDP level for one quarter.",
        "parameters": {
            "type": "object",
            "properties": {
                "quarter": {"type": "string"}
            },
            "required": ["quarter"],
            "additionalProperties": False
        },
        "strict": True
    },
    {
        "type": "function",
        "name": "get_yoy_growth",
        "description": "Get YoY GDP growth for a quarter.",
        "parameters": {
            "type": "object",
            "properties": {
                "quarter": {"type": "string"}
            },
            "required": ["quarter"],
            "additionalProperties": False
        },
        "strict": True
    },
    {
        "type": "function",
        "name": "compare_quarters",
        "description": (
            "Compare GDP levels between two quarters. "
            "Returns percentage change in GDP levels, "
            "not the difference between YoY growth rates."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "quarter_a": {"type": "string"},
                "quarter_b": {"type": "string"}
            },
            "required": ["quarter_a", "quarter_b"],
            "additionalProperties": False
        },
        "strict": True
    },
    {
        "type": "function",
        "name": "get_extreme_growth",
        "description": (
            "Find the highest or lowest YoY GDP growth."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "kind": {
                    "type": "string",
                    "enum": ["highest", "lowest"]
                }
            },
            "required": ["kind"],
            "additionalProperties": False
        },
        "strict": True
    }
]


# =====================================
# 3. CHATBOT (Chat Completions format)
# =====================================

# Chat Completions wraps each tool in a "function" key
CHAT_TOOLS = [
    {
        "type": "function",
        "function": {
            k: v for k, v in tool.items()
            if k not in ("type", "strict")
        },
    }
    for tool in TOOLS
]


def ask_data_question(
    df: pd.DataFrame,
    question: str
) -> str:
    """Answer economic questions using tool calling."""

    client = get_ai_client().with_options(max_retries=5)

    # Bind functions to current dataset
    functions = build_tools(df)

    instructions = """
    You are an economic data analyst.

    The dataset contains synthetic quarterly GDP data.

    Rules:
    - Always use the provided tools for numerical data.
    - Never invent figures or economic explanations.
    - Clearly state that data is synthetic.
    - Distinguish changes in GDP levels from
      changes in year-on-year growth rates.
    - If a tool returns an error, explain it.
    - If available tools cannot answer a question,
      explain the limitation.
    """

    messages = [
        {"role": "system", "content": instructions},
        {"role": "user", "content": question},
    ]

    # Limit tool-calling rounds
    for _ in range(3):

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            tools=CHAT_TOOLS,
            tool_choice="auto",
            temperature=0,
        )

        message = response.choices[0].message

        # No tool requested: this is the final answer
        if not message.tool_calls:
            if not message.content:
                raise ValueError(
                    "Cerebras returned no text response"
                )
            return message.content

        # Keep the model's tool request in the conversation
        messages.append(message)

        # Run each requested Python function
        for call in message.tool_calls:
            name = call.function.name

            if name not in functions:
                result = {"error": "Unknown tool requested"}
            else:
                try:
                    arguments = json.loads(call.function.arguments)
                    result = functions[name](**arguments)
                except (ValueError, TypeError, KeyError) as error:
                    result = {"error": str(error)}

            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps(result),
            })

    raise RuntimeError("Maximum tool-calling rounds exceeded")