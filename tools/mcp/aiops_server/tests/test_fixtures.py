"""Fixture library validation."""

import pytest
from aiops.fixtures import FixtureSpec, get_fixture, load_fixtures


class TestFixtureLibrary:
    def test_fixtures_load_and_validate(self):
        fixtures = load_fixtures()
        assert fixtures, "fixture library must not be empty"
        for fixture in fixtures.values():
            assert isinstance(fixture, FixtureSpec)

    def test_ids_are_versioned(self):
        for fixture_id in load_fixtures():
            assert fixture_id.rsplit("-v", 1)[1].isdigit()

    def test_expected_defaults_exist(self):
        assert get_fixture("inference-basic-v1", kind="inference")
        assert get_fixture("reasoning-basic-v1", kind="reasoning")
        assert get_fixture("toolcall-weather-v1", kind="tool_calling")
        assert get_fixture("tps-decode-v1", kind="tps")

    def test_tool_calling_fixture_is_complete(self):
        fixture = get_fixture("toolcall-weather-v1")
        assert fixture.tools
        assert fixture.expected.tool_name == "get_current_weather"
        assert fixture.expected.arguments_schema

    def test_unknown_fixture_raises(self):
        with pytest.raises(ValueError, match="Unknown fixture"):
            get_fixture("does-not-exist-v1")

    def test_kind_mismatch_raises(self):
        with pytest.raises(ValueError, match="kind"):
            get_fixture("inference-basic-v1", kind="tps")
