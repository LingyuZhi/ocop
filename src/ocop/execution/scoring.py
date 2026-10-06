import re
from decimal import Decimal


SCORER_VERSION = "ocop.numeric.v1"
NUMBER = r"[+-]?[0-9]+(?:\.[0-9]+)?"
FINAL_LINE = re.compile(r"#### (" + NUMBER + r")")
REFERENCE_LINE = re.compile(r"#### ([+-]?(?:[0-9]+|[0-9]{1,3}(?:,[0-9]{3})+)(?:\.[0-9]+)?)")


def final_line(content: str) -> str:
    return re.sub(r"(?:\r?\n)+\Z", "", content).split("\n")[-1]


def parse_answer(content: str) -> Decimal | None:
    line = final_line(content)
    match = FINAL_LINE.fullmatch(line)
    return Decimal(match[1]) if match else None


def normalize_reference(answer: str) -> Decimal:
    if re.fullmatch(NUMBER, answer):
        return Decimal(answer)
    match = REFERENCE_LINE.fullmatch(final_line(answer))
    if match is None:
        raise ValueError("Invalid reference answer")
    return Decimal(match[1].replace(",", ""))


def score_answer(content: str, reference: Decimal) -> dict:
    predicted = parse_answer(content)
    reason = "format_error" if predicted is None else "success" if predicted == reference else "wrong_answer"
    return {"scorer_version": SCORER_VERSION, "reason": reason, "success": reason == "success",
            "predicted": str(predicted) if predicted is not None else None, "reference": str(reference)}
