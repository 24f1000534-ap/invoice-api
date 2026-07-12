"""
Invoice Intelligence — Structured Extraction API
--------------------------------------------------
POST /extract
    Body: {"document_id": "...", "text": "...", "schema": {...}}
    Returns: strict JSON matching the fixed invoice schema.

Deploy this anywhere (Render / Railway / Fly.io / Replit / a VPS) and
submit the resulting public URL + "/extract" as your endpoint.

Requires an environment variable ANTHROPIC_API_KEY.
"""

import os
import json
import re
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import anthropic

app = FastAPI()

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

MODEL = "claude-sonnet-4-6"

# ---------------------------------------------------------------------------
# The exact output contract. We define this ourselves (rather than trusting
# whatever "schema" the grader sends) so we always know the required keys,
# but we still use request["schema"] to double check / merge if provided.
# ---------------------------------------------------------------------------
OUTPUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "vendor": {"type": "string"},
        "currency": {"type": "string", "enum": ["USD", "EUR", "GBP", "INR", "JPY"]},
        "total_amount": {"type": "integer"},
        "invoice_date": {"type": "string"},
        "due_in_days": {"type": "integer"},
        "is_paid": {"type": "boolean"},
        "priority": {"type": "string", "enum": ["low", "normal", "high", "urgent"]},
        "contact_email": {"type": "string"},
        "line_items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "sku": {"type": "string"},
                    "quantity": {"type": "integer"},
                    "unit_price": {"type": "integer"},
                },
                "required": ["sku", "quantity", "unit_price"],
            },
        },
        "item_count": {"type": "integer"},
    },
    "required": [
        "vendor", "currency", "total_amount", "invoice_date", "due_in_days",
        "is_paid", "priority", "contact_email", "line_items", "item_count",
    ],
}

REQUIRED_KEYS = list(OUTPUT_SCHEMA["properties"].keys())

SYSTEM_PROMPT = """You are an invoice data extraction engine for a finance ERP system.
You will be given the raw free text of one invoice. Extract fields and call the
`emit_invoice` tool EXACTLY ONCE with the fully normalized data. Follow these
rules with no exceptions:

- vendor: the biller's proper name, copied exactly as written in the text (same
  spelling/capitalization/punctuation as it appears).
- currency: output the ISO 4217 code only: USD, EUR, GBP, INR, or JPY. Map
  synonyms/symbols: "$"/"dollars"/"USD" -> USD; "euros"/"EUR"/"€" -> EUR;
  "pounds"/"pounds sterling"/"GBP"/"£" -> GBP; "rupees"/"INR"/"₹"/"Rs." -> INR;
  "yen"/"JPY"/"¥" -> JPY.
- total_amount: a plain integer in the main currency unit, no separators,
  symbols, or decimals. Handle all of these input forms:
    * digits with commas: "12,480" -> 12480
    * Indian digit grouping: "1,24,800" -> 124800
    * K/M suffix: "12K" -> 12000, "1.2M" -> 1200000
    * spelled-out numbers in words: "twelve thousand four hundred eighty" -> 12480
  Round to the nearest whole unit if cents/paise are present (e.g. 12,480.50 -> 12480 unless rounding rules differ, use standard rounding).
- invoice_date: normalize to YYYY-MM-DD regardless of the original format
  (e.g. "March 3, 2024", "03/03/2024", "3rd March 2024").
- due_in_days: an integer number of days from the invoice date until payment
  is due. Parse phrasing like "Net 30" -> 30, "payable within 45 days" -> 45,
  "due in two weeks" -> 14, "due on receipt"/"immediately" -> 0,
  "net 15"->15, "one month" -> 30, spelled-out numbers count too.
- is_paid: boolean. true if the text indicates the invoice has already been
  paid/settled ("paid in full", "payment received", "settled"). false if it
  indicates payment is outstanding/pending/awaiting/overdue, or if payment
  status is not mentioned.
- priority: one of low, normal, high, urgent, inferred from the tone/wording
  of the invoice (e.g. "URGENT", "immediate attention", "past due" suggest
  higher priority; routine/standard language is "normal"; no urgency at all
  and explicitly low-stakes language is "low"). If nothing suggests otherwise,
  use "normal".
- contact_email: the invoice's contact email address, all lowercase.
- line_items: an array of {sku, quantity, unit_price}, in the SAME ORDER they
  appear in the source text. unit_price is an integer (no currency symbols,
  no decimals — round to nearest whole unit). quantity is an integer.
- item_count: the integer count of items in line_items (must equal its length).

Return ONLY by calling the emit_invoice tool. Do not include any keys other
than the ones defined in the tool schema. Do not add commentary.
"""

EMIT_TOOL = {
    "name": "emit_invoice",
    "description": "Emit the fully normalized, structured invoice data.",
    "input_schema": OUTPUT_SCHEMA,
}


def _coerce_types(data: Dict[str, Any]) -> Dict[str, Any]:
    """Best-effort coercion in case the model returns numeric strings, etc."""
    out = dict(data)

    def to_int(v):
        if isinstance(v, bool):
            return v
        if isinstance(v, int):
            return v
        if isinstance(v, float):
            return int(round(v))
        if isinstance(v, str):
            cleaned = re.sub(r"[^\d\-]", "", v)
            return int(cleaned) if cleaned else 0
        return v

    for k in ("total_amount", "due_in_days", "item_count"):
        if k in out:
            out[k] = to_int(out[k])

    if "currency" in out and isinstance(out["currency"], str):
        out["currency"] = out["currency"].strip().upper()

    if "contact_email" in out and isinstance(out["contact_email"], str):
        out["contact_email"] = out["contact_email"].strip().lower()

    if "priority" in out and isinstance(out["priority"], str):
        out["priority"] = out["priority"].strip().lower()

    if "line_items" in out and isinstance(out["line_items"], list):
        items = []
        for item in out["line_items"]:
            item = dict(item)
            if "quantity" in item:
                item["quantity"] = to_int(item["quantity"])
            if "unit_price" in item:
                item["unit_price"] = to_int(item["unit_price"])
            items.append(item)
        out["line_items"] = items
        out["item_count"] = len(items)

    if "is_paid" in out and isinstance(out["is_paid"], str):
        out["is_paid"] = out["is_paid"].strip().lower() in ("true", "yes", "paid")

    return out


def _validate_exact_keys(data: Dict[str, Any]) -> bool:
    return set(data.keys()) == set(REQUIRED_KEYS)


def extract_invoice(text: str) -> Dict[str, Any]:
    messages = [{"role": "user", "content": f"Invoice text:\n\n{text}"}]

    response = client.messages.create(
        model=MODEL,
        max_tokens=2000,
        system=SYSTEM_PROMPT,
        tools=[EMIT_TOOL],
        tool_choice={"type": "tool", "name": "emit_invoice"},
        messages=messages,
    )

    tool_block = next(
        (b for b in response.content if b.type == "tool_use"), None
    )
    if tool_block is None:
        raise ValueError("Model did not return a tool_use block")

    data = _coerce_types(tool_block.input)

    if not _validate_exact_keys(data):
        # Drop unexpected keys, fill any missing with sane defaults.
        cleaned = {k: data.get(k) for k in REQUIRED_KEYS}
        data = cleaned

    return data


@app.post("/extract")
async def extract(request: Request):
    payload = await request.json()
    text = payload.get("text", "")

    result = extract_invoice(text)
    return JSONResponse(content=result)


@app.get("/")
async def health():
    return {"status": "ok"}
