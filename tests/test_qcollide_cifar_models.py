import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from qgapselect.qcollide.cifar_models import (  # noqa: E402
    cifar_control_targets,
    make_cifar_model,
)


def test_cifar_architectures_share_the_control_interface() -> None:
    images = torch.rand(2, 3, 32, 32)
    parameter_counts = []
    for architecture in ("resnet18", "vit_tiny", "mlp_mixer"):
        model = make_cifar_model(
            architecture,
            hidden_dimension=128,
            control_dimension=48,
            class_count=10,
        )
        logits, control, hidden = model(images)
        assert logits.shape == (2, 10)
        assert control.shape == (2, 48)
        assert hidden.shape == (2, 128)
        parameter_counts.append(sum(parameter.numel() for parameter in model.parameters()))
    assert max(parameter_counts) / min(parameter_counts) < 1.2


def test_cifar_control_target_shape() -> None:
    images = torch.rand(3, 3, 32, 32).numpy()
    descriptor = cifar_control_targets(images, grid_size=4)
    assert descriptor.shape == (3, 48)
