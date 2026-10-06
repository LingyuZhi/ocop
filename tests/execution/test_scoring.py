from decimal import Decimal

import pytest

from ocop.execution.scoring import normalize_reference, parse_answer, score_answer


@pytest.mark.parametrize("text", ["#### 5", "reasoning\n#### +5.00\n", "#### 5\r\n", "#### 05.0\n\n"])
def test_decimal_answers(text):
    assert parse_answer(text) == Decimal(5)
    assert score_answer(text, Decimal(5))["reason"] == "success"


@pytest.mark.parametrize("text", ["5", "####5", "####  5", " #### 5", "#### 5 ", "#### 5 balls",
    "#### 5e0", "#### 5/1", "#### 1,000", "#### .5", "#### 5.", "#### NaN", "#### Infinity",
    "#### ５", "#### 5\nmore", "#### 5\n ", "", "#### 5\r"])
def test_strict_format(text):
    assert parse_answer(text) is None
    assert score_answer(text, Decimal(5))["reason"] == "format_error"


def test_exact_comparison_and_signed_zero():
    assert score_answer("#### -0.00", Decimal("0"))["success"]
    assert score_answer("#### -2.5", Decimal("-2.50"))["success"]
    assert score_answer("#### 0.10000000000000000000000000001", Decimal("0.1"))["reason"] == "wrong_answer"


def test_reference_normalization_is_separate():
    assert normalize_reference("Calculation: 500 * 2 = 1000\n#### 1,000") == Decimal("1000")
    assert normalize_reference("#### -1,234.50\n") == Decimal("-1234.50")
    assert normalize_reference("18") == Decimal(18)
    for text in ("#### 12,34", "#### 1e3", "#### NaN", "18 dollars"):
        with pytest.raises(ValueError):
            normalize_reference(text)
