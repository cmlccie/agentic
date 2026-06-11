"""Tests for agentic.simple_agent.reload."""

import asyncio

from agentic.simple_agent.reload import AgentState, ReloadCoordinator

# -------------------------------------------------------------------------------------------------
# AgentState
# -------------------------------------------------------------------------------------------------


class TestAgentState:
    def test_running_is_ready(self):
        coord = ReloadCoordinator()
        assert coord.state == AgentState.RUNNING
        assert coord.is_ready is True
        assert coord.is_reloading is False

    def test_draining_is_not_ready(self):
        coord = ReloadCoordinator()
        coord.state = AgentState.DRAINING
        assert coord.is_ready is False
        assert coord.is_reloading is True

    def test_reloading_is_not_ready(self):
        coord = ReloadCoordinator()
        coord.state = AgentState.RELOADING
        assert coord.is_ready is False
        assert coord.is_reloading is True


# -------------------------------------------------------------------------------------------------
# trigger_reload
# -------------------------------------------------------------------------------------------------


class TestTriggerReload:
    def test_running_transitions_to_draining(self):
        coord = ReloadCoordinator()
        coord.trigger_reload()
        assert coord.state == AgentState.DRAINING

    def test_sets_reload_requested(self):
        coord = ReloadCoordinator()
        coord.trigger_reload()
        assert coord._reload_requested.is_set()

    def test_idempotent_when_draining(self):
        coord = ReloadCoordinator()
        coord.trigger_reload()
        coord.trigger_reload()
        assert coord.state == AgentState.DRAINING

    def test_no_op_when_reloading(self):
        coord = ReloadCoordinator()
        coord.state = AgentState.RELOADING
        coord.trigger_reload()
        assert coord.state == AgentState.RELOADING


# -------------------------------------------------------------------------------------------------
# request_slot / release_slot
# -------------------------------------------------------------------------------------------------


class TestSlotManagement:
    def test_request_slot_granted_when_running(self):
        async def _run():
            coord = ReloadCoordinator()
            granted = await coord.request_slot()
            assert granted is True
            assert coord.in_flight_count == 1

        asyncio.run(_run())

    def test_request_slot_denied_when_draining(self):
        async def _run():
            coord = ReloadCoordinator()
            coord.state = AgentState.DRAINING
            granted = await coord.request_slot()
            assert granted is False
            assert coord.in_flight_count == 0

        asyncio.run(_run())

    def test_request_slot_denied_when_reloading(self):
        async def _run():
            coord = ReloadCoordinator()
            coord.state = AgentState.RELOADING
            granted = await coord.request_slot()
            assert granted is False

        asyncio.run(_run())

    def test_release_slot_decrements_count(self):
        async def _run():
            coord = ReloadCoordinator()
            await coord.request_slot()
            await coord.request_slot()
            assert coord.in_flight_count == 2
            await coord.release_slot()
            assert coord.in_flight_count == 1

        asyncio.run(_run())

    def test_release_slot_clamps_at_zero(self):
        async def _run():
            coord = ReloadCoordinator()
            await coord.release_slot()
            assert coord.in_flight_count == 0

        asyncio.run(_run())

    def test_release_slot_signals_drained_when_draining(self):
        async def _run():
            coord = ReloadCoordinator()
            await coord.request_slot()
            coord.state = AgentState.DRAINING
            assert not coord._drained.is_set()
            await coord.release_slot()
            assert coord._drained.is_set()

        asyncio.run(_run())

    def test_release_slot_does_not_signal_when_running(self):
        async def _run():
            coord = ReloadCoordinator()
            await coord.request_slot()
            await coord.release_slot()
            assert not coord._drained.is_set()

        asyncio.run(_run())


# -------------------------------------------------------------------------------------------------
# wait_for_drain
# -------------------------------------------------------------------------------------------------


class TestWaitForDrain:
    def test_transitions_to_reloading_after_drain(self):
        async def _run():
            coord = ReloadCoordinator()
            coord.state = AgentState.DRAINING
            coord._drained.set()
            await coord.wait_for_drain()
            assert coord.state == AgentState.RELOADING

        asyncio.run(_run())

    def test_transitions_to_reloading_on_timeout(self):
        async def _run():
            coord = ReloadCoordinator(drain_timeout=0.01)
            coord.state = AgentState.DRAINING
            coord.in_flight_count = 1
            await coord.wait_for_drain()
            assert coord.state == AgentState.RELOADING

        asyncio.run(_run())

    def test_clears_drained_event_after_drain(self):
        async def _run():
            coord = ReloadCoordinator()
            coord.state = AgentState.DRAINING
            coord._drained.set()
            await coord.wait_for_drain()
            assert not coord._drained.is_set()

        asyncio.run(_run())


# -------------------------------------------------------------------------------------------------
# mark_running
# -------------------------------------------------------------------------------------------------


class TestMarkRunning:
    def test_transitions_to_running(self):
        coord = ReloadCoordinator()
        coord.state = AgentState.RELOADING
        coord.mark_running()
        assert coord.state == AgentState.RUNNING

    def test_clears_reload_requested(self):
        coord = ReloadCoordinator()
        coord._reload_requested.set()
        coord.mark_running()
        assert not coord._reload_requested.is_set()
