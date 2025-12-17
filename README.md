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

@tool(hide_from_history=True, remote={"cpu": 2, "memory": "2gb"})
def arima_forecast(
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
    s = pd.Series(values, index=pd.to_datetime(timestamps))

    tool_warnings: list[str] = []

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

    if return_conf_int:
        pred = fitted.get_forecast(steps=steps)
        mean = pred.predicted_mean

        ci = pred.conf_int(alpha=0.05)  # 95%
        lower = ci.iloc[:, 0]
        upper = ci.iloc[:, 1]
    else:
        mean = fitted.forecast(steps=steps)
        lower = upper = None

    last_t = s.index[-1]
    future_index = pd.date_range(start=last_t, periods=steps + 1, freq=freq)[1:]

    # Reindex to our generated future_index (avoid index mismatch without manual casting)
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

    return {
        "order": {"p": p, "d": d, "q": q},
        "steps": steps,
        "forecast": forecast_points,
        "diagnostics": {"aic": fitted.aic, "bic": fitted.bic},
        "warnings": tool_warnings,
    }
```

### Advanced Agent Example


TBC

## Key Features

TBC

## Why AceAI?


TBC
