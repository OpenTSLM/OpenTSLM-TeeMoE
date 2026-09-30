import re

import pytest

from teemoe import prompts


def test_forecast_message_layout():
    past, future = prompts.history_timestamps("2024-01-01 00:00", "h", 3, 2)
    assert past == ["2024-01-01 00:00:00", "2024-01-01 01:00:00", "2024-01-01 02:00:00"]
    assert future == ["2024-01-01 03:00:00", "2024-01-01 04:00:00"]
    message = prompts.forecast_message([1.5, 2, 1234567.0], past, future, context="Demand rises.")
    assert message.startswith("I have a time series forecasting task for you.\n\nContext:\n<context>\nDemand rises.\n")
    assert "(2024-01-01 02:00:00, 1234567)" in message and "(2024-01-01 00:00:00, 1.5)" in message
    assert message.endswith("Future timestamps:\n2024-01-01 03:00:00\n2024-01-01 04:00:00")
    assert "Context" not in prompts.forecast_message([1.0], past[-1:], future)


def test_forecast_output_round_trip():
    stamps = ["2024-01-01 03:00:00", "2024-01-01 04:00:00"]
    text = "<forecast>\n(2024-01-01 03:00:00, 1.25)\n(2024-01-01 04:00:00, -3e2)\n</forecast>"
    assert re.fullmatch(prompts.forecast_regex(stamps), text)
    assert prompts.parse_forecast(text, stamps) == [1.25, -300.0]
    with pytest.raises(ValueError):
        prompts.parse_forecast(text, stamps + ["2024-01-01 05:00:00"])
    assert prompts.forecast_max_tokens(2) == 512


def test_analysis_message_and_choice():
    message = prompts.analysis_message([[1, 2, 3, 4, 5, 6, 7, 8]], "Is it rising?", ["Yes", "No"],
                                       concept_definitions="Trend: a direction.")
    assert "Time series values (successive observations separated by spaces):\n1 2 3 4 5 6 7 8" in message
    assert "Options:\nA) Yes\nB) No" in message and "Supplied concept definitions:\nTrend: a direction." in message
    assert prompts.parse_choice("B) No", ["Yes", "No"]) == 1


def test_routing_view_drops_the_series():
    class Tokenizer:
        def apply_chat_template(self, messages, **_):
            return "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages) + \
                "<|im_start|>assistant\n"

    past, future = prompts.history_timestamps("2024-01-01", "D", 3, 1)
    prompt = prompts.chat_prompt(Tokenizer(), prompts.forecast_message([7.0, 8.0, 9.0], past, future, "Hot week."))
    view = prompts.routing_view(prompt)
    assert "Hot week." in view and "8" not in view.split("<|im_start|>user")[1].split("Future")[0]
