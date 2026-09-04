import torch

from turntiming.models.joint import JointTimingModel


def test_joint_model_shapes() -> None:
    model = JointTimingModel(n_mels=64, hidden_size=32)
    out = model(torch.randn(4, 100, 64))
    assert out["turn_end_logit"].shape == (4,)
    assert out["onset_delay_ms"].shape == (4,)
    assert torch.all(out["onset_delay_ms"] >= 0)
