import pytest
from unittest.mock import patch, MagicMock

from app.evaluation.experiment import run_experiment
from app.evaluation.report import experiment_to_json, experiment_to_markdown


def test_run_id_created():
    fake_result = MagicMock()
    fake_result.scores = [{"exact_match": 1.0}]
    with patch("ragas.evaluate", return_value=fake_result):
        res = run_experiment([{"user_input": "q", "response": "a", "reference": "a"}])
    assert res.run_id
    assert res.status == "completed"


def test_empty_dataset():
    res = run_experiment([])
    assert res.status == "failed"
    assert res.sample_count == 0


def test_json_report():
    fake_result = MagicMock()
    fake_result.scores = [{"exact_match": 1.0}]
    with patch("ragas.evaluate", return_value=fake_result):
        res = run_experiment([{"user_input": "q", "response": "a", "reference": "a"}])
    j = experiment_to_json(res)
    assert "run_id" in j


def test_markdown_report():
    fake_result = MagicMock()
    fake_result.scores = [{"exact_match": 1.0}]
    with patch("ragas.evaluate", return_value=fake_result):
        res = run_experiment([{"user_input": "q", "response": "a", "reference": "a"}])
    md = experiment_to_markdown(res)
    assert "Evaluation Report" in md


def test_dataset_immutability():
    data = [{"user_input": "q", "response": "a", "reference": "a"}]
    snapshot = [dict(d) for d in data]
    fake_result = MagicMock()
    fake_result.scores = [{"exact_match": 1.0}]
    with patch("ragas.evaluate", return_value=fake_result):
        run_experiment(data)
    assert data == snapshot