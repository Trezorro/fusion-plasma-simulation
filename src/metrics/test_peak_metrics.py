"""Copilot generated and never updated. :D"""
import pytest
import torch
import types
from src.metrics.peak_metric import PeakMetric
import pandas as pd

class DummyConfig:
    class data:
        cols = types.SimpleNamespace(x=["DML", "PD"])
        history_length = 2
        seq_length = 3
    run_name = "testrun"

def dummy_get_current_config():
    return DummyConfig

@pytest.fixture(autouse=True)
def patch_config(monkeypatch):
    monkeypatch.setattr("src.metrics.peak_metric.get_current_config", dummy_get_current_config)

def test_peak_metric_init():
    metric = PeakMetric(condition_history="any_Wh")
    assert hasattr(metric, "CHANNEL_NAMES")
    assert metric.condition == "any_Wh"
    assert "DML" in metric.CHANNEL_NAMES
    assert "PD" in metric.CHANNEL_NAMES

def test_test_condition_any_Wh():
    metric = PeakMetric(condition_history="any_Wh")
    history = torch.tensor([[0, 1], [2, 2]])
    mask = metric.test_condition(history)
    assert mask.all()

def test_test_condition_only():
    metric = PeakMetric(condition_history="L_only_Wh")
    history = torch.tensor([[0, 0], [0, 1], [1, 1]])
    mask = metric.test_condition(history)
    assert mask.tolist() == [True, False, False]

def test_test_condition_in():
    metric = PeakMetric(condition_history="L_in_Wh")
    history = torch.tensor([[0, 1], [1, 2], [2, 2]])
    mask = metric.test_condition(history)
    assert mask.tolist() == [True, False, False]

def test_test_condition_not_in():
    metric = PeakMetric(condition_history="L_not_in_Wh")
    history = torch.tensor([[0, 1], [1, 2], [2, 2]])
    mask = metric.test_condition(history)
    assert mask.tolist() == [False, True, True]

def test_test_condition_mixed():
    metric = PeakMetric(condition_history="mixed")
    history = torch.tensor([[0, 0], [0, 1], [2, 2]])
    mask = metric.test_condition(history)
    assert mask.tolist() == [False, True, False]

def test_update_and_compute(monkeypatch):
    metric = PeakMetric(condition_history="any_Wh")
    # Patch batch_get_peakprops to return dummy objects
    class DummyPeak:
        def __init__(self):
            self.Y = torch.tensor([1.0])
            self.prominences = torch.tensor([2.0])
            self.bases = torch.tensor([3.0])
            self.widths = torch.tensor([4.0])
            self.energy_delta = torch.tensor([5.0])
            self.pd_prominence = torch.tensor([6.0])
            self.energy_ratio = torch.tensor([7.0])
        def num_peaks(self): return 1
        def __sub__(self, other): return self
        height = 1.0
        prominence = 2.0
        base = 3.0
        width = 4.0
        energy_delta = 5.0
        pd_prominence = 6.0
        energy_ratio = 7.0

    def dummy_batch_get_peakprops(x, **kwargs):
        # Return shape (B, C)
        B, C = x.shape[0], 2
        return [[DummyPeak() for _ in range(C)] for _ in range(B)]

    monkeypatch.setattr("src.metrics.peak_metric.batch_get_peakprops", dummy_batch_get_peakprops)
    monkeypatch.setattr("src.metrics.peak_metric.prefix_metrics", lambda d, c: d)

    pred = torch.ones(2, 2, 3)
    target = torch.ones(2, 2, 3)
    labels = torch.zeros(2, 2, dtype=torch.long)
    metric.update(pred, target, labels)
    out = metric.compute()
    assert isinstance(out, dict)
    assert "total_hits" in out

def test_extract_df():
    metric = PeakMetric(condition_history="any_Wh")
    # Simulate some state
    metric.add_state("list:DML/counts_pred", default=torch.tensor([1, 2], dtype=torch.int32), dist_reduce_fx="cat")
    metric.add_state("list:DML/counts_target", default=torch.tensor([1, 2], dtype=torch.int32), dist_reduce_fx="cat")
    df = metric.extract_df("DML", "count")
    assert "value" in df.columns
    assert "distribution" in df.columns

def test_extract_df_all():
    metric = PeakMetric(condition_history="any_Wh")
    # Simulate some state
    metric.add_state("list:DML/counts_pred", default=torch.tensor([1, 2], dtype=torch.int32), dist_reduce_fx="cat")
    metric.add_state("list:DML/counts_target", default=torch.tensor([1, 2], dtype=torch.int32), dist_reduce_fx="cat")
    df = metric.extract_df_all()
    assert isinstance(df, type(__import__("pandas").DataFrame()))

def test_save_histogram(monkeypatch, tmp_path):
    metric = PeakMetric(condition_history="any_Wh")
    df = pd.DataFrame({
        "condition": ["any_Wh"]*2,
        "channel_name": ["DML"]*2,
        "measure": ["count"]*2,
        "distribution": ["Generated", "Real"],
        "value": [1, 2]
    })
    # Patch plotly express and fig.write_image
    class DummyFig:
        def update_layout(self, **kwargs): return None
        def for_each_annotation(self, fn): return None
        def update_xaxes(self, **kwargs): return None
        def update_yaxes(self, **kwargs): return None
        def write_image(self, *a, **k): return None
    monkeypatch.setattr("plotly.express.histogram", lambda *a, **k: DummyFig())
    metric.C.run_name = "testrun"
    metric.save_histogram("DML", "count", df=df)