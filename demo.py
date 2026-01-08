import base64
import json
import os
from typing import Annotated, Any, cast

from dotenv import load_dotenv
from httpx import AsyncClient
from ididi import use
from openai import AsyncOpenAI
from opentelemetry import trace
from opentelemetry.trace import SpanKind, set_span_in_context
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from aceai import AgentBase, Graph, LLMService, Tool, spec, tool
from aceai.errors import AceAIValidationError
from aceai.agent.executor import ToolExecutor
from aceai.llm.openai import OpenAI
from term_ui import run_agent_with_terminal_ui

WAREHOUSE_ORDERS = {
    "ORD-200": {
        "destination": "Denver, CO",
        "priority": "express",
        "notes": "Helios Labs needs hardware before the Wednesday stand-up.",
        "items": [
            {"sku": "sensor-kit", "quantity": 45},
            {"sku": "control-module", "quantity": 30},
            {"sku": "battery-pack", "quantity": 60},
        ],
    },
    "ORD-207": {
        "destination": "Austin, TX",
        "priority": "standard",
        "notes": "Stocking field pod replacements.",
        "items": [
            {"sku": "telemetry-node", "quantity": 22},
            {"sku": "cooling-shroud", "quantity": 18},
        ],
    },
}

SKU_WEIGHTS = {
    "sensor-kit": 0.85,
    "control-module": 1.4,
    "battery-pack": 0.65,
    "telemetry-node": 1.1,
    "cooling-shroud": 2.2,
}

SHIPPING_RATES = {
    "standard": {"base": 48.0, "per_kg": 1.35, "eta_days": 5},
    "express": {"base": 92.0, "per_kg": 1.95, "eta_days": 2},
}


@tool
def lookup_order(
    order_id: Annotated[str, spec(description="Order identifier such as ORD-200.")],
) -> str:
    """Return line items, destination, and priority for a warehouse order."""
    order = WAREHOUSE_ORDERS.get(order_id.upper())
    if not order:
        raise AceAIValidationError(f"Unknown order {order_id}")
    return json.dumps(order)


@tool
def get_sku_weight(
    sku: Annotated[
        str, spec(description="Catalog SKU to pull the per-unit weight for.")
    ],
) -> str:
    """Return the per-unit weight for a SKU in kilograms."""
    key = sku.lower()
    if key not in SKU_WEIGHTS:
        raise AceAIValidationError(f"Unknown SKU {sku}")
    return json.dumps({"sku": key, "weight_kg": SKU_WEIGHTS[key]})


@tool
def estimate_shipping_cost(
    weight_kg: Annotated[float, spec(description="Total shipment mass in kilograms.")],
    method: Annotated[
        str, spec(description="Shipping tier to price (standard or express).")
    ],
) -> str:
    """Quote the shipping cost given a total weight and service tier."""
    method_key = method.strip().lower()
    if method_key not in SHIPPING_RATES:
        raise AceAIValidationError(f"Unsupported shipping method {method}")
    rates = SHIPPING_RATES[method_key]
    cost = rates["base"] + weight_kg * rates["per_kg"]
    return json.dumps(
        {
            "method": method_key,
            "weight_kg": round(weight_kg, 2),
            "cost_usd": round(cost, 2),
            "eta_days": rates["eta_days"],
        }
    )


OPEN_METEO_GEOCODE = "https://geocoding-api.open-meteo.com/v1/search"
OPEN_METEO_FORECAST = "https://api.open-meteo.com/v1/forecast"


def build_langfuse_tracer() -> trace.Tracer:
    public_key = os.environ["LANGFUSE_PUBLIC_KEY"]
    secret_key = os.environ["LANGFUSE_SECRET_KEY"]
    base_url = os.environ["LANGFUSE_BASE_URL"]

    auth = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()

    otel_provider = TracerProvider(
        resource=Resource.create(
            {
                "service.name": "aceai-demo",
            }
        )
    )
    exporter = OTLPSpanExporter(
        endpoint=f"{base_url}/api/public/otel/v1/traces",
        headers={"Authorization": f"Basic {auth}"},
    )
    otel_provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(otel_provider)
    return trace.get_tracer("aceai-demo")


async def build_async_http_client() -> AsyncClient:
    return AsyncClient(timeout=20.0)


@tool
async def fetch_weather_window(
    city: Annotated[str, spec(description="Destination city to inspect")],
    client: Annotated[AsyncClient, use(build_async_http_client, reuse=False)],
) -> str:
    """Fetch a quick temperature and precipitation outlook for the next 24 hours."""
    normalized_city = city.strip()
    if not normalized_city:
        raise AceAIValidationError("City must be a non-empty string")

    queries: list[str] = []
    seen: set[str] = set()

    def add_query(value: str) -> None:
        candidate = value.strip()
        if not candidate:
            return
        key = candidate.lower()
        if key in seen:
            return
        queries.append(candidate)
        seen.add(key)

    # Try bare city names first because Open-Meteo rejects comma-delimited inputs like "Denver, CO".
    primary_city = normalized_city.split(",")[0]
    add_query(primary_city)
    add_query(normalized_city)

    try:
        location = None
        for query in queries:
            geo_resp = await client.get(
                OPEN_METEO_GEOCODE,
                params={"name": query, "count": 1, "language": "en", "format": "json"},
                timeout=15.0,
            )
            geo_resp.raise_for_status()
            geo_payload = geo_resp.json()
            results = geo_payload.get("results", [])
            if results:
                location = results[0]
                break

        if not location:
            raise AceAIValidationError(f"No coordinates found for {city}")

        lat = location["latitude"]
        lon = location["longitude"]

        forecast_resp = await client.get(
            OPEN_METEO_FORECAST,
            params={
                "latitude": lat,
                "longitude": lon,
                "hourly": "temperature_2m,precipitation_probability",
                "forecast_days": 1,
                "timezone": "auto",
            },
            timeout=15.0,
        )
        forecast_resp.raise_for_status()
        forecast = forecast_resp.json()["hourly"]
        temps = forecast["temperature_2m"]
        precip = forecast["precipitation_probability"]
        avg_temp = sum(temps) / len(temps)
        max_precip = max(precip)

        summary = {
            "city": city,
            "latitude": lat,
            "longitude": lon,
            "avg_temp_c": round(avg_temp, 1),
            "max_precip_probability": int(max_precip),
            "source": "open-meteo",
        }
        return json.dumps(summary)
    finally:
        await client.aclose()


def build_agent(
    prompt: str,
    max_turns: int,
    tools: list[Tool[Any, Any]],
    *,
    model: str,
    openai_api_key: str,
    tracer: trace.Tracer,
) -> AgentBase:
    graph = Graph()

    llm_service = LLMService(
        providers=[
            OpenAI(
                client=AsyncOpenAI(api_key=openai_api_key),
                default_meta={"model": model},
            )
        ],
        timeout_seconds=120,
    )
    executor = ToolExecutor(graph=graph, tools=tools, tracer=tracer)

    return AgentBase(
        prompt=prompt,
        default_model=model,
        llm_service=llm_service,
        executor=executor,
        max_steps=max_turns,
        tracer=tracer,
    )


async def main():
    load_dotenv(".env")
    api_key = os.environ["OPENAI_API_KEY"]
    tracer = build_langfuse_tracer()
    multi_step_question = """
    You are preparing a logistics brief for Helios Labs covering order ORD-200.
    
    Tasks:
    1. Call `lookup_order` to restate the destination, priority, and every SKU with its quantity.
    2. Use `get_sku_weight` for each SKU individually so you can calculate the total shipment weight in kilograms.
    3. Price BOTH `standard` and `express` service levels by calling `estimate_shipping_cost` twice with the total weight.
    4. Call `fetch_weather_window` for the destination city to understand short-term weather risks that might impact delivery.
    5. Recommend which service level to book, citing cost, ETA, the customer's stated priority, and the weather outlook.
    
    Present the answer as a brief overview paragraph plus bullet points that cover total weight, each quote, the weather takeaway, and the final recommendation.
    """

    agent = build_agent(
        prompt=(
            "You are the logistics coordinator for AceAI. "
            "Always inspect orders, fetch SKU weights, price shipping strictly through the available tools, "
            "and incorporate the weather outlook before deciding on a service level."
        ),
        max_turns=20,
        tools=[
            lookup_order,
            get_sku_weight,
            estimate_shipping_cost,
            fetch_weather_window,
        ],
        openai_api_key=api_key,
        model="gpt-5.1",
        tracer=tracer,
    )

    try:
        with tracer.start_as_current_span(
            "demo.run",
            kind=SpanKind.INTERNAL,
            attributes={
                "demo.question": multi_step_question,
                "langfuse.trace.name": "aceai.demo",
                "langfuse.trace.input": multi_step_question,
            },
        ) as span:
            parent_ctx = set_span_in_context(span)
            answer = await run_agent_with_terminal_ui(
                agent,
                multi_step_question,
                trace_ctx=parent_ctx,
            )
            span.set_attribute("langfuse.trace.output", answer)
    finally:
        provider = cast(TracerProvider, trace.get_tracer_provider())
        provider.force_flush()
        provider.shutdown()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
