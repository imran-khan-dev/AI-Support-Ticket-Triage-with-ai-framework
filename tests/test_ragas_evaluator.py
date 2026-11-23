import pytest
from unittest.mock import patch, MagicMock

from app.evaluation.ragas_evaluator import run_ragas_evaluation, RagasEvaluationError


def test_empty_dataset():
    with pytest.raises(RagasEvaluationError):
        run_ragas_evaluation([])


def test_missing_metric_config():
    with pytest.raises(RagasEvaluationError):
        run_ragas_evaluation([{"user_input": "q", "response": "a", "reference": "a"}], metric_names=[])


def test_successful_evaluation_mocked():
    fake_result = MagicMock()
    fake_result.scores = [{"exact_match": 1.0}]
    with patch("ragas.evaluate", return_value=fake_result):
        out = run_ragas_evaluation([{"user_input": "q", "response": "a", "reference": "a"}])
    assert out["status"] == "completed"
    assert out["sample_count"] == 1
    assert out["aggregate"]["exact_match"] == 1.0


def test_execution_failure():
    with patch("ragas.evaluate", side_effect=RuntimeError("boom")):
        with pytest.raises(RagasEvaluationError):
            run_ragas_evaluation([{"user_input": "q", "response": "a", "reference": "a"}])