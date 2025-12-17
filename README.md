# AceAI

> ⚠️ **Experimental Project** - AceAI is currently in early development. APIs may change frequently. 
> 
> ⭐ **Star this repo** to stay updated on our progress and be notified of major releases!

**Ace AI - Agent framework that delivers**

AceAI is a powerful and intuitive agent framework designed to help you build intelligent agents that deliver results. Whether you're creating conversational bots, task automation agents, or complex AI workflows, AceAI provides the tools you need to succeed.

## Installation

```bash
pip install aceai
```

## Usage

### Quick Start
```python
from aceai import tool, Param, use

class Tracer:
    """
    Minimal tracer interface for demo.
    Your real implementation can map to OpenTelemetry / Langfuse / LangSmith / your own tracing.
    """

    @contextmanager
    def span(self, name: str, **attrs):
        yield

    def event(self, name: str, payload: dict):
        pass


def get_tracer(secrets: Annotated[Secrets, use(get_secrets)]) -> Tracer:
    return Tracer()

@tool(hide_from_history=True, remote={"cpu": 2, "memory": "2gb"}, description="""
    - You have access to a forecasting tool arima_forecast.
    - Use it only when the user asks for time-series forecasting, trend projection, or short-term prediction.
    
    Before calling the tool:
        - Ensure the input values are ordered by time.
        - Choose order=(p,d,q) if the user specifies it; otherwise start with (1,1,1).
        - Decide steps from the user request (default 14).
        - If timestamps are provided, ensure freq is provided or inferable; otherwise pass a freq.
    After the tool returns:
        - Present the forecast clearly and mention the chosen ARIMA order and steps.
        - If the tool returns warnings, surface them briefly.
"""
)
def arima_forecast(
    tracer: Annotated[Tracer, use(get_tracer)],
    values: Annotated[
        list[float],
        Param("Time-series values ordered from oldest to newest (already validated)."),
    ],
    timestamps: Annotated[
        list[str],
        Param(
            "Timestamps aligned 1:1 with `values`, ISO-8601 strings (already validated). "
            "Will be parsed by pandas.to_datetime."
        ),
    ],
    p: Annotated[int, Param("AR order p (>=0).", ge=0, le=10)] = 1,
    d: Annotated[int, Param("Differencing order d (>=0).", ge=0, le=3)] = 1,
    q: Annotated[int, Param("MA order q (>=0).", ge=0, le=10)] = 1,
    steps: Annotated[int, Param("Number of future steps to forecast.", ge=1, le=365)] = 14,
    freq: Annotated[
        str,
        Param(
            "Time frequency used to generate future timestamps. "
            "Examples: 'D' (daily), 'H' (hourly), 'MS' (month start)."
        ),
    ] = "D",
    return_conf_int: Annotated[
        bool,
        Param("If true, return 95% confidence interval (lower/upper) for each forecast point."),
    ] = True,
    method: Annotated[
        Literal["statsmodels"],
        Param("Implementation backend. Currently only 'statsmodels' is supported."),
    ] = "statsmodels",
):
    """
    Fit an ARIMA(p,d,q) model and return a timestamped forward forecast.

    Assumptions (validated by outer layer):
      - values/timestamps lengths match, timestamps are parseable and monotonic,
        values are finite, series length is sufficient.
      - freq is valid for pandas date_range.

    Output:
      - order: {p,d,q}
      - steps: int
      - forecast: list[{t, yhat, yhat_lower, yhat_upper}] (bounds only if return_conf_int=true)
      - diagnostics: {aic, bic}
      - warnings: list[str]
    """
    with tracer.span(
        "tool.arima_forecast",
        order={"p": p, "d": d, "q": q},
        steps=steps,
        freq=freq,
        return_conf_int=return_conf_int,
        n_points=len(values),
    ):
        tracer.event(
            "tool.input",
            {
                "n_points": len(values),
                "start_ts": timestamps[0],
                "end_ts": timestamps[-1],
                "order": {"p": p, "d": d, "q": q},
                "steps": steps,
                "freq": freq,
                "return_conf_int": return_conf_int,
                "method": method,
            },
        )

        s = pd.Series(values, index=pd.to_datetime(timestamps))

        tool_warnings: list[str] = []
        t0 = perf_counter()
        with tracer.span("arima.fit"):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                fitted = ARIMA(
                    s,
                    order=(p, d, q),
                    enforce_stationarity=True,
                    enforce_invertibility=True,
                ).fit()

                for ww in w:
                    msg = str(ww.message).strip()
                    if msg:
                        tool_warnings.append(msg)

        fit_ms = (perf_counter() - t0) * 1000.0

        with tracer.span("arima.forecast", steps=steps):
            if return_conf_int:
                pred = fitted.get_forecast(steps=steps)
                mean = pred.predicted_mean

                ci = pred.conf_int(alpha=0.05)  # 95%
                lower = ci.iloc[:, 0]
                upper = ci.iloc[:, 1]
            else:
                mean = fitted.forecast(steps=steps)
                lower = upper = None

        future_index = pd.date_range(start=s.index[-1], periods=steps + 1, freq=freq)[1:]

        mean = pd.Series(mean.to_numpy(), index=future_index)
        if return_conf_int:
            lower = pd.Series(lower.to_numpy(), index=future_index)
            upper = pd.Series(upper.to_numpy(), index=future_index)

        forecast_points: list[dict] = []
        if return_conf_int:
            for t in future_index:
                forecast_points.append(
                    {
                        "t": t.isoformat(),
                        "yhat": mean.loc[t],
                        "yhat_lower": lower.loc[t],
                        "yhat_upper": upper.loc[t],
                    }
                )
        else:
            for t in future_index:
                forecast_points.append({"t": t.isoformat(), "yhat": mean.loc[t]})

        result = {
            "order": {"p": p, "d": d, "q": q},
            "steps": steps,
            "freq": freq,
            "forecast": forecast_points,
            "diagnostics": {"aic": fitted.aic, "bic": fitted.bic},
            "warnings": tool_warnings,
        }

        tracer.event(
            "tool.output",
            {
                "fit_ms": fit_ms,
                "aic": fitted.aic,
                "bic": fitted.bic,
                "warnings_count": len(tool_warnings),
                "forecast_points": len(forecast_points),
            },
        )

        if tool_warnings:
            tracer.event("tool.warnings", {"warnings": tool_warnings})

        return result
```

### Advanced Agent Example


TBC

## Key Features

TBC

## Why AceAI?


TBC
