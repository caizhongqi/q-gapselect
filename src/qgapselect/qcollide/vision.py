"""Public API for the optional real-image Q-COLLIDE experiment."""

from .vision_campaign import run_real_vision_campaign
from .vision_merge import compact_real_vision_summary, merge_real_vision_artifacts

__all__ = [
    "compact_real_vision_summary",
    "merge_real_vision_artifacts",
    "run_real_vision_campaign",
]
