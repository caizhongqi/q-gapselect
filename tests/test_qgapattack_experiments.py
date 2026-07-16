from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from qgapselect.qgapattack_experiments import (
    CLAIM_STATUS,
    audit_experiment_design,
    load_and_audit_experiment_design,
    parse_experiment_design,
)


def _config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / "qgapattack_experiments.json"


def _document() -> dict[str, object]:
    return json.loads(_config_path().read_text(encoding="utf-8"))


def test_preregistered_matrix_has_required_coverage_without_claiming_results() -> None:
    design, audit = load_and_audit_experiment_design(_config_path())

    assert len(design.baselines) == 38
    assert len(design.benchmarks) == 7
    assert len(design.metrics) == 19
    assert len(design.panels) == 11
    assert audit.claim_status == CLAIM_STATUS
    assert audit.design_valid
    assert not audit.empirical_ready
    assert not audit.ccf_a_claimable
    assert all(audit.coverage_checks.values())
    assert all(audit.fairness_checks.values())
    assert all(audit.statistics_checks.values())
    assert not any(audit.execution_checks.values())
    assert any("Layer-P" in blocker for blocker in audit.blockers)


def test_primary_baseline_cannot_use_a_mismatched_oracle() -> None:
    document = copy.deepcopy(_document())
    baselines = document["baseline_registry"]
    assert isinstance(baselines, list)
    target = next(
        item
        for item in baselines
        if isinstance(item, dict) and item.get("id") == "oracle_durr_hoyer_minimum"
    )
    target["primary_eligible"] = True

    with pytest.raises(ValueError, match="not eligible for a primary win"):
        parse_experiment_design(document)


def test_panel_rejects_an_unknown_baseline_reference() -> None:
    document = copy.deepcopy(_document())
    panels = document["experiment_panels"]
    assert isinstance(panels, list) and isinstance(panels[0], dict)
    panels[0]["baseline_ids"].append("missing-baseline")

    with pytest.raises(ValueError, match="missing references"):
        parse_experiment_design(document)


def test_audit_detects_a_disabled_fairness_gate() -> None:
    document = copy.deepcopy(_document())
    fairness = document["fairness_contract"]
    assert isinstance(fairness, dict)
    fairness["no_victim_tuning"] = False

    audit = audit_experiment_design(parse_experiment_design(document))

    assert not audit.design_valid
    assert not audit.empirical_ready
    assert not audit.ccf_a_claimable
