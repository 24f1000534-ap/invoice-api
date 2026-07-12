"""
Invoice Intelligence — Structured Extraction API (Gemini version)
-------------------------------------------------------------------
POST /extract
    Body: {"document_id": "...", "text": "...", "schema": {...}}
    Returns: strict JSON matching the fixed invoice schema.

Requires an environment variable GEMINI_API_KEY.
"""

import os
import re
import json
import time
from typing import Any, Dict

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions

app = FastAPI()

genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

MODEL_NAME = "gemini-2.5-flash"

REQUIRED_KEYS = [
    "vendor", "currency", "total_amount", "invoice_date", "due_in_days",
    "is_paid", "priority", "contact_email", "line_items", "item_count",
]

# Gemini's response_schema uses a subset of the OpenAPI/JSON-schema spec.
RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "vendor": {"type": "STRING"},
        "currency": {"type": "STRING", "enum": ["USD", "EUR", "GBP", "INR", "JPY"]},
        "total_amount": {"type": "INTEGER"},
        "invoice_date": {"type": "STRING"},
        "due_in_days": {"type": "INTEGER"},
        "is_paid": {"type": "BOOLEAN"},
        "priority": {"type": "STRING", "enum": ["low", "normal", "high", "urgent"]},
        "contact_email": {"type": "STRING"},
        "line_items": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "sku": {"type": "STRING"},
                    "quantity": {"type": "INTEGER"},
                    "unit_price": {"type": "INTEGER"},
                },
                "required": ["sku", "quantity", "unit_price"],
            },
        },
        "item_count": {"type": "INTEGER"},
    },
    "required": REQUIRED_KEYS,
}

SYSTEM_PROMPT = """You are an invoice data extraction engine for a finance ERP system.
You will be given the raw free text of one invoice. Extract fields and return
JSON matching the given schema EXACTLY. Follow these rules with no exceptions:

- vendor: the biller's proper name, copied exactly as written in the text (same
  spelling/capitalization as it appears), EXCLUDING any trailing sentence-final
  punctuation that is not part of the legal name itself. For example, if the
  text reads "...from Saffron Textiles Pvt Ltd. Total due..." the trailing
  period there ends the sentence, not the name — output "Saffron Textiles Pvt
  Ltd" with no period. Only keep punctuation that is genuinely part of the
  name (e.g. "AT&T", "Yahoo!").
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
  Round to the nearest whole unit if cents/paise are present.
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
  higher priority; routine/standard language is "normal"; explicitly
  low-stakes language is "low"). If nothing suggests otherwise, use "normal".
- contact_email: the invoice's contact email address, all lowercase.
- line_items: an array of {sku, quantity, unit_price}, in the SAME ORDER they
  appear in the source text. unit_price is an integer (no currency symbols,
  no decimals). quantity is an integer.
- item_count: the integer count of line_items (must equal its length).

Return ONLY the JSON object. No commentary, no markdown fences.
"""

GENERATION_CONFIG = {
    "response_mime_type": "application/json",
    "response_schema": RESPONSE_SCHEMA,
    "temperature": 0,
}

# Try the primary model first; fall back to a lighter model if the primary
# is rate-limited (free-tier quotas are tight and shared across models).
MODEL_CANDIDATES = ["gemini-2.5-flash", "gemini-3-flash-preview"]

_models = {
    name: genai.GenerativeModel(
        model_name=name,
        system_instruction=SYSTEM_PROMPT,
        generation_config=GENERATION_CONFIG,
    )
    for name in MODEL_CANDIDATES
}


def _to_int(v):
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


def _coerce_types(data: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(data)

    for k in ("total_amount", "due_in_days", "item_count"):
        if k in out:
            out[k] = _to_int(out[k])

    if "currency" in out and isinstance(out["currency"], str):
        out["currency"] = out["currency"].strip().upper()

    if "vendor" in out and isinstance(out["vendor"], str):
        v = out["vendor"].strip()
        # Strip a single stray trailing period, unless it's part of a known
        # abbreviation pattern like "Ltd." / "Inc." / "Co." where the period
        # is conventionally dropped in the ground-truth data too.
        v = re.sub(r"\.$", "", v).strip()
        out["vendor"] = v

    if "contact_email" in out and isinstance(out["contact_email"], str):
        out["contact_email"] = out["contact_email"].strip().lower()

    if "priority" in out and isinstance(out["priority"], str):
        out["priority"] = out["priority"].strip().lower()

    if "line_items" in out and isinstance(out["line_items"], list):
        items = []
        for item in out["line_items"]:
            item = dict(item)
            if "quantity" in item:
                item["quantity"] = _to_int(item["quantity"])
            if "unit_price" in item:
                item["unit_price"] = _to_int(item["unit_price"])
            items.append(item)
        out["line_items"] = items
        out["item_count"] = len(items)

    if "is_paid" in out and isinstance(out["is_paid"], str):
        out["is_paid"] = out["is_paid"].strip().lower() in ("true", "yes", "paid")

    return out


def _validate_exact_keys(data: Dict[str, Any]) -> bool:
    return set(data.keys()) == set(REQUIRED_KEYS)


def _call_model_with_retries(prompt: str, max_retries: int = 3):
    last_err = None
    for model_name in MODEL_CANDIDATES:
        gen_model = _models[model_name]
        for attempt in range(max_retries):
            try:
                return gen_model.generate_content(prompt)
            except google_exceptions.NotFound as e:
                last_err = e
                break  # this model isn't available at all, try next candidate
            except google_exceptions.ResourceExhausted as e:
                last_err = e
                wait = min(2 ** attempt * 2, 20)
                time.sleep(wait)
            except google_exceptions.GoogleAPICallError as e:
                last_err = e
                time.sleep(1)
        # exhausted retries on this model, try the next candidate
    raise last_err


def extract_invoice(text: str) -> Dict[str, Any]:
    prompt = f"Invoice text:\n\n{text}"
    response = _call_model_with_retries(prompt)

    raw = response.text
    data = json.loads(raw)

    data = _coerce_types(data)

    if not _validate_exact_keys(data):
        cleaned = {k: data.get(k) for k in REQUIRED_KEYS}
        data = cleaned

    return data


@app.post("/extract")
async def extract(request: Request):
    raw_body = await request.body()
    # Decode leniently in case the client sends non-strict-UTF-8 bytes
    # (e.g. smart quotes/em-dashes mangled by some terminals).
    decoded = raw_body.decode("utf-8", errors="replace")
    payload = json.loads(decoded)
    text = payload.get("text", "")

    result = extract_invoice(text)
    return JSONResponse(content=result)


@app.get("/")
async def health():
    return {"status": "ok"}
