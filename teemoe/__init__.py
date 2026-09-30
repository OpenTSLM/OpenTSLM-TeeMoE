"""OpenTSLM TeeMoE: forecasting, contextual prediction and temporal reasoning with one model."""

__version__ = "1.0.0"


def __getattr__(name):
    if name in {"TeeMoE", "Forecast", "Answer"}:
        from . import teemoe
        return getattr(teemoe, name)
    raise AttributeError(name)
