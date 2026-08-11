from __future__ import annotations

import sys
from types import SimpleNamespace

from qgapselect.qcollide.hf_cifar10 import HFCIFAR10


class _FakeSplit:
    def __init__(self) -> None:
        self._rows = [
            {"img": "image-0", "label": 1},
            {"img": "image-1", "label": 4},
        ]

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, key):
        if key == "label":
            return [row["label"] for row in self._rows]
        return self._rows[int(key)]


def test_hf_cifar10_exposes_torchvision_subset(monkeypatch, tmp_path):
    calls: list[dict[str, object]] = []

    def load_dataset(dataset_id, *, split, cache_dir):
        calls.append(
            {"dataset_id": dataset_id, "split": split, "cache_dir": cache_dir}
        )
        return _FakeSplit()

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=load_dataset))
    dataset = HFCIFAR10(tmp_path, train=False, download=True)

    assert len(dataset) == 2
    assert dataset.targets == [1, 4]
    assert dataset[1] == ("image-1", 4)
    assert calls == [
        {
            "dataset_id": "uoft-cs/cifar10",
            "split": "test",
            "cache_dir": str(tmp_path),
        }
    ]
