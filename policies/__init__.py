# Policies package
"""
Policy-driven configuration for the Air & Insights Agent.

Business rules are stored in JSON files for:
- Auditability: Changes are tracked and explainable
- Controllability: Business users can adjust thresholds
- Maintainability: No code changes needed for rule updates
"""

from pathlib import Path
import json

POLICIES_DIR = Path(__file__).parent


def load_safety_rules() -> dict:
    """Load safety rules from JSON configuration."""
    rules_path = POLICIES_DIR / "safety_rules.json"
    with open(rules_path, "r", encoding="utf-8") as f:
        return json.load(f)


# Pre-load rules on import for performance
SAFETY_RULES = load_safety_rules()
