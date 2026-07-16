import sys
from unittest.mock import MagicMock

# Define MockSessionState class for dictionary & attribute access
class MockSessionState(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)
    def __setattr__(self, name, value):
        self[name] = value
    def __delattr__(self, name):
        try:
            del self[name]
        except KeyError:
            raise AttributeError(name)

# Create mock streamlit module
mock_st = MagicMock()
mock_st.session_state = MockSessionState()
# Pre-populate bot_thread to prevent the background thread from launching during tests
mock_st.session_state["bot_thread"] = True

# Also mock layout elements
mock_st.columns = lambda *args, **kwargs: [MagicMock() for _ in range(args[0] if isinstance(args[0], int) else len(args[0]))]
mock_st.tabs = lambda *args, **kwargs: [MagicMock() for _ in range(len(args[0]))]

# Apply mock to sys.modules
sys.modules['streamlit'] = mock_st

# Import app under test
import app

def test_init_game_state():
    # Reset session state except bot_thread
    app.st.session_state = MockSessionState()
    app.st.session_state["bot_thread"] = True

    # Run init
    app.init_game_state()

    assert app.st.session_state.game_initialized is True
    assert len(app.st.session_state.neuron_grid) == 6
    assert app.st.session_state.chemicals["sanity"] == 100.0
    assert app.st.session_state.chemicals["energy"] == 100.0
    assert app.st.session_state.stats["iq"] == 0.0

    # Check updated attributes are initialized
    assert "doping" in app.st.session_state.cooldowns
    assert "reflex" in app.st.session_state.missions

def test_evolution_stages():
    assert app.get_evolution_stage(10.0) == "Bò sát (Instinct)"
    assert app.get_evolution_stage(200.0) == "Thú cổ (Emotional)"
    assert app.get_evolution_stage(1000.0) == "Người tinh khôn (Logical)"
    assert app.get_evolution_stage(4000.0) == "Siêu trí tuệ lượng tử (Transcendence)"

def test_simulation_tick_metabolism():
    # Reset and initialize
    app.st.session_state = MockSessionState()
    app.st.session_state["bot_thread"] = True
    app.init_game_state()

    # Force full energy and known upgrades
    app.st.session_state.chemicals["energy"] = 100.0
    app.st.session_state.upgrades["brainstem"] = 1 # generates 6.0 energy per tick

    # Total neurons: 3 starting neurons. Cost = 1.0 + 3 * 0.4 = 2.2
    # Net energy change should be +6.0 - 2.2 = +3.8. Capped at 100.0.
    # Set starting energy to 50.0 to see the increase
    app.st.session_state.chemicals["energy"] = 50.0

    app.run_simulation_tick()

    assert app.st.session_state.chemicals["energy"] == 53.8

def test_sensory_neuron_buildup():
    app.st.session_state = MockSessionState()
    app.st.session_state["bot_thread"] = True
    app.init_game_state()

    # Empty all but one Sensory Neuron at [0][0]
    for r in range(6):
        for c in range(6):
            app.st.session_state.neuron_grid[r][c] = {"type": "Empty", "charge": 0.0, "threshold": 0.5, "fire_rate": 0.0, "last_fired": -1, "direction": "All"}

    app.st.session_state.neuron_grid[0][0] = {
        "type": "Sensory",
        "charge": 0.1,
        "threshold": 0.5,
        "fire_rate": 0.2,
        "last_fired": -1,
        "direction": "All"
    }
    app.st.session_state.chemicals["dopamine"] = 50.0 # boost = 1.5. Increase = 0.2 * 1.5 = 0.3

    app.run_simulation_tick()

    # 0.1 + 0.3 = 0.4
    assert abs(app.st.session_state.neuron_grid[0][0]["charge"] - 0.4) < 1e-5

def test_directional_signal_propagation():
    """
    UPGRADE TEST: Directional Axon Growth
    Verify that if a firing neuron's direction is set to 'Right', it transfers charge
    only to the neighbor to its right, and NOT to the neighbor below it.
    """
    app.st.session_state = MockSessionState()
    app.st.session_state["bot_thread"] = True
    app.init_game_state()

    # Clean grid
    for r in range(6):
        for c in range(6):
            app.st.session_state.neuron_grid[r][c] = {"type": "Empty", "charge": 0.0, "threshold": 0.5, "fire_rate": 0.0, "last_fired": -1, "direction": "All"}

    # Place a firing Sensory neuron at [0][0] with direction set to 'Right'
    app.st.session_state.neuron_grid[0][0] = {
        "type": "Sensory",
        "charge": 0.6, # threshold 0.4, fires
        "threshold": 0.4,
        "fire_rate": 0.0,
        "last_fired": -1,
        "direction": "Right" # Only send right!
    }
    # Interneuron to its right
    app.st.session_state.neuron_grid[0][1] = {
        "type": "Interneuron",
        "charge": 0.1,
        "threshold": 0.5,
        "fire_rate": 0.0,
        "last_fired": -1,
        "direction": "All"
    }
    # Interneuron to its bottom
    app.st.session_state.neuron_grid[1][0] = {
        "type": "Interneuron",
        "charge": 0.1,
        "threshold": 0.5,
        "fire_rate": 0.0,
        "last_fired": -1,
        "direction": "All"
    }

    # With myelin = 0, signal efficiency is 0.35.
    # Allowed direction is 'Right', so neighbor is only [0][1].
    # Transferred charge = (0.6 * 0.35) / 1 = 0.21.
    # [0][1] (right) should get 0.1 + 0.21 = 0.31
    # [1][0] (bottom) should remain at 0.1
    app.run_simulation_tick()

    assert abs(app.st.session_state.neuron_grid[0][1]["charge"] - 0.31) < 1e-5
    assert app.st.session_state.neuron_grid[1][0]["charge"] == 0.1
    assert app.st.session_state.neuron_grid[0][0]["charge"] == 0.0

def test_motor_neuron_firing_yields():
    app.st.session_state = MockSessionState()
    app.st.session_state["bot_thread"] = True
    app.init_game_state()

    # Clean grid
    for r in range(6):
        for c in range(6):
            app.st.session_state.neuron_grid[r][c] = {"type": "Empty", "charge": 0.0, "threshold": 0.5, "fire_rate": 0.0, "last_fired": -1, "direction": "All"}

    # Place a Motor neuron that has full charge
    app.st.session_state.neuron_grid[3][3] = {
        "type": "Motor",
        "charge": 0.9, # threshold 0.5, fires
        "threshold": 0.5,
        "fire_rate": 0.0,
        "last_fired": -1,
        "direction": "All"
    }

    app.st.session_state.stats["iq"] = 10.0
    app.st.session_state.stats["memory"] = 5.0

    app.run_simulation_tick()

    # Motor firing should yield IQ and memory
    assert app.st.session_state.stats["iq"] > 10.0
    assert app.st.session_state.stats["memory"] > 5.0

def test_sanity_burnout_recovery():
    app.st.session_state = MockSessionState()
    app.st.session_state["bot_thread"] = True
    app.init_game_state()

    # Place an Interneuron at [1][1]
    app.st.session_state.neuron_grid[1][1] = {
        "type": "Interneuron",
        "charge": 0.0,
        "threshold": 0.5,
        "fire_rate": 0.0,
        "last_fired": -1,
        "direction": "All"
    }

    # Reduce Sanity to 0.1 to trigger burnout on tick
    app.st.session_state.chemicals["sanity"] = 0.1
    app.st.session_state.chemicals["stress"] = 90.0

    app.run_simulation_tick()

    # Sanity is reset to 25.0 and burnout count increments
    assert app.st.session_state.chemicals["sanity"] == 25.0
    assert app.st.session_state.stats["burnout_count"] == 1

def test_apply_event_effects():
    app.st.session_state = MockSessionState()
    app.st.session_state["bot_thread"] = True
    app.init_game_state()

    app.st.session_state.chemicals["dopamine"] = 50.0
    app.st.session_state.chemicals["acetylcholine"] = 50.0

    # Apply custom effects
    app.apply_event_effects(
        da=10.0, ach=-5.0, stress=15.0, energy=-20.0, se=5.0, sanity=-10.0,
        log_msg="Thử nghiệm hiệu ứng biến cố", iq_gain=25.0, mem_gain=5.0
    )

    assert app.st.session_state.chemicals["dopamine"] == 60.0
    assert app.st.session_state.chemicals["acetylcholine"] == 45.0
    assert app.st.session_state.chemicals["stress"] == 25.0 # starting stress was 10.0
    assert app.st.session_state.chemicals["energy"] == 80.0 # starting energy was 100.0
    assert app.st.session_state.chemicals["sanity"] == 90.0 # starting sanity was 100.0
    assert app.st.session_state.stats["iq"] == 25.0
    assert app.st.session_state.stats["memory"] == 15.0 # starting memory was 10.0

def test_active_hormone_abilities():
    """
    UPGRADE TEST: Hormone Special Abilities
    Ensure cooldown decrements and status values update correctly during simulation ticks.
    """
    app.st.session_state = MockSessionState()
    app.st.session_state["bot_thread"] = True
    app.init_game_state()

    # Set a mock cooldown
    app.st.session_state.cooldowns["doping"] = 5

    # Run simulation tick and verify decrement
    app.run_simulation_tick()
    assert app.st.session_state.cooldowns["doping"] == 4

def test_mission_system_evaluation():
    """
    UPGRADE TEST: Cognitive Missions
    Verify that missions can transition from 'In Progress' to 'Completed' when criteria are met.
    """
    app.st.session_state = MockSessionState()
    app.st.session_state["bot_thread"] = True
    app.init_game_state()

    # Empty grid
    for r in range(6):
        for c in range(6):
            app.st.session_state.neuron_grid[r][c] = {"type": "Empty", "charge": 0.0, "threshold": 0.5, "fire_rate": 0.0, "last_fired": -1, "direction": "All"}

    # Condition: Sensory & Motor present (Reflex Arc)
    app.st.session_state.neuron_grid[0][0] = {"type": "Sensory", "charge": 0.0, "threshold": 0.5, "fire_rate": 0.0, "last_fired": -1, "direction": "All"}
    app.st.session_state.neuron_grid[0][1] = {"type": "Motor", "charge": 0.0, "threshold": 0.5, "fire_rate": 0.0, "last_fired": -1, "direction": "All"}

    # Prior state is In Progress
    assert app.st.session_state.missions["reflex"]["status"] == "In Progress"

    # Run evaluation
    app.check_mission_statuses()

    # State should now be Completed!
    assert app.st.session_state.missions["reflex"]["status"] == "Completed"
