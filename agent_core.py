import json

from ai_config import get_ai_client, MODEL_NAME


def _memory(history, memory_turns):
    """Turn the chat history into model-ready messages."""
    usable = [
        {"role": m["role"], "content": m["content"]}
        for m in history
        if not m.get("is_error")
    ]
    return usable[-memory_turns * 2:]


def run_tool_agent(
    system_prompt,
    tool_schemas,
    functions,
    question,
    history=None,
    max_rounds=6,
    memory_turns=6,
):
    """
    Generic agent loop: the model calls tools until it can answer.
    Returns (answer, trace), where trace lists every tool call.
    """
    client = get_ai_client().with_options(max_retries=5)

    messages = (
        [{"role": "system", "content": system_prompt}]
        + _memory(history or [], memory_turns)
        + [{"role": "user", "content": question}]
    )

    trace = []

    for _ in range(max_rounds):
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            tools=tool_schemas,
            tool_choice="auto",
            temperature=0,
        )
        message = response.choices[0].message

        if not message.tool_calls:
            if not message.content:
                raise ValueError("Cerebras returned no text response")
            return message.content, trace

        messages.append(message)

        for call in message.tool_calls:
            name = call.function.name
            arguments = {}

            if name not in functions:
                result = {"error": "Unknown tool requested"}
            else:
                try:
                    arguments = json.loads(call.function.arguments or "{}")
                    result = functions[name](**arguments)
                except (ValueError, TypeError, KeyError) as error:
                    result = {"error": str(error)}

            trace.append({
                "tool": name,
                "arguments": arguments,
                "result": result,
            })

            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps(result),
            })

    # Out of rounds: force a best-effort answer instead of crashing
    messages.append({
        "role": "user",
        "content": (
            "You have used all available tool calls. Answer now using "
            "only the results above, and say what is missing."
        ),
    })
    final = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        tools=tool_schemas,
        tool_choice="none",
        temperature=0,
    )
    return final.choices[0].message.content or "No answer produced.", trace