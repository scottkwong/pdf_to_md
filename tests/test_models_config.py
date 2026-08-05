"""
Tests keeping the ModelKey and Provider enums in lockstep with models.json.

The enums exist so code and tests get autocomplete instead of retyping exact
model/provider strings; these tests are what make that safe — any drift
between models.json and the enums fails here.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models_config import ModelKey, Provider, load_models_config  # noqa: E402


def test_model_keys_match_models_json() -> None:
    """ModelKey members and models.json keys are exactly the same set."""
    config_keys = set(load_models_config().keys())
    enum_keys = {member.value for member in ModelKey}
    missing_from_enum = config_keys - enum_keys
    missing_from_config = enum_keys - config_keys
    assert not missing_from_enum, (
        f"models.json keys missing a ModelKey member: {sorted(missing_from_enum)}"
    )
    assert not missing_from_config, (
        f"ModelKey members with no models.json entry: {sorted(missing_from_config)}"
    )


def test_providers_in_models_json_are_valid() -> None:
    """Every provider named in models.json is a Provider member."""
    valid = {member.value for member in Provider}
    for name, cfg in load_models_config().items():
        assert cfg.get("provider") in valid, (
            f"models.json entry '{name}' has unknown provider "
            f"'{cfg.get('provider')}'; valid: {sorted(valid)}"
        )


def test_enums_compare_and_format_as_strings() -> None:
    """The str mixin behaves as expected for comparisons and f-strings."""
    assert Provider.FIREWORKS == "fireworks"
    assert ModelKey.QWEN_3_7_PLUS == "qwen3.7-plus"
    assert f"{Provider.FIREWORKS}" == "fireworks"
    assert f"{ModelKey.GPT_5_5}" == "gpt-5.5"
    # Usable as dict keys interchangeably with plain strings.
    assert {"fireworks": True}[Provider.FIREWORKS] is True


def run_all_tests() -> bool:
    """Run models-config enum tests directly (used by run_tests.py)."""
    test_model_keys_match_models_json()
    test_providers_in_models_json_are_valid()
    test_enums_compare_and_format_as_strings()
    print("  ✓ Models config enum tests passed")
    return True


if __name__ == "__main__":
    run_all_tests()
