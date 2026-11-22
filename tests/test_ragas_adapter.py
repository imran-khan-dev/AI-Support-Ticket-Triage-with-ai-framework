import pytest

from app.evaluation.ragas_adapter import record_to_sample, records_to_samples


def test_record_to_sample_basic():
    s = record_to_sample({"user_input": "q", "response": "a", "reference": "r", "retrieved_contexts": ["c"]})
    assert s.user_input == "q"
    assert s.response == "a"


def test_missing_user_input():
    with pytest.raises(ValueError):
        record_to_sample({})


def test_malformed_contexts():
    with pytest.raises(ValueError):
        record_to_sample({"user_input": "q", "retrieved_contexts": "bad"})