from analyze_hypotheses import fmt


def test_fmt_rounds_decimal_half_values_despite_binary_float_artifacts():
    assert fmt(0.5549999999999999, 2) == "0.56"
    assert fmt(0.5549, 2) == "0.55"
