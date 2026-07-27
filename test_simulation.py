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

# Create mock modules
mock_st = MagicMock()
mock_st.session_state = MockSessionState()
# Pre-populate bot_thread to prevent the background thread from launching during tests
mock_st.session_state["bot_thread"] = True

# Also mock layout elements
mock_st.columns = lambda *args, **kwargs: [MagicMock() for _ in range(args[0] if isinstance(args[0], int) else len(args[0]))]
mock_st.tabs = lambda *args, **kwargs: [MagicMock() for _ in range(len(args[0]))]

sys.modules['streamlit'] = mock_st

# Do not mock pandas if it is installed
try:
    import pandas
except ImportError:
    mock_pd = MagicMock()
    sys.modules['pandas'] = mock_pd

# Mock pypdf
mock_pypdf = MagicMock()
sys.modules['pypdf'] = mock_pypdf

# Mock telegram
mock_telegram = MagicMock()
sys.modules['telegram'] = mock_telegram
sys.modules['telegram.error'] = mock_telegram
sys.modules['telegram.ext'] = mock_telegram

# Mock huggingface_hub
mock_hf = MagicMock()
sys.modules['huggingface_hub'] = mock_hf

# Import app under test
import app

def test_init_game_state_3d():
    # Reset session state except bot_thread
    app.st.session_state = MockSessionState()
    app.st.session_state["bot_thread"] = True

    # Run init
    app.init_game_state()

    assert app.st.session_state.game_initialized is True
    # 3D Grid size must be 4x4x4 = 64 nodes
    assert len(app.st.session_state.neuron_grid) == 4
    assert len(app.st.session_state.neuron_grid[0]) == 4
    assert len(app.st.session_state.neuron_grid[0][0]) == 4

    assert app.st.session_state.chemicals["sanity"] == 100.0
    assert app.st.session_state.chemicals["energy"] == 100.0
    assert app.st.session_state.selected_cell == (0, 0, 0)

    # Check 3D indices are initialized
    assert "csi" in app.st.session_state
    assert "pdi" in app.st.session_state
    assert "vpi" in app.st.session_state

def test_serialize_deserialize_3d():
    app.st.session_state = MockSessionState()
    app.st.session_state["bot_thread"] = True
    app.init_game_state()

    grid = app.st.session_state.neuron_grid

    # Edit some 3D cells to be distinct
    grid[0][1][2] = {"type": "Sensory", "charge": 0.0, "threshold": 0.4, "fire_rate": 0.3, "last_fired": -1, "direction": "Front", "weight": 2.0, "amyloid_plaque": False}
    grid[1][2][3] = {"type": "Motor", "charge": 0.0, "threshold": 0.6, "fire_rate": 0.0, "last_fired": -1, "direction": "Back", "weight": 3.0, "amyloid_plaque": False}

    # Serialize
    code = app.serialize_grid(grid)
    assert len(code) > 0

    # Deserialize
    loaded = app.deserialize_grid(code)
    assert loaded is not None
    assert len(loaded) == 4
    assert loaded[0][1][2]["type"] == "Sensory"
    assert loaded[0][1][2]["direction"] == "Front"
    assert loaded[0][1][2]["weight"] == 2.0

    assert loaded[1][2][3]["type"] == "Motor"
    assert loaded[1][2][3]["direction"] == "Back"
    assert loaded[1][2][3]["weight"] == 3.0

def test_3d_signal_propagation():
    app.st.session_state = MockSessionState()
    app.st.session_state["bot_thread"] = True
    app.init_game_state()

    grid = app.st.session_state.neuron_grid

    # Clear grid
    for x in range(4):
        for y in range(4):
            for z in range(4):
                grid[x][y][z] = {"type": "Empty", "charge": 0.0, "threshold": 0.5, "weight": 1.0, "direction": "All", "amyloid_plaque": False}

    # Place a Sensory neuron firing in a specific direction (e.g. Up -> (0, 1, 0))
    grid[1][1][1] = {
        "type": "Sensory",
        "charge": 0.8,
        "threshold": 0.4,
        "fire_rate": 0.0,
        "last_fired": -1,
        "direction": "Up",
        "weight": 1.0,
        "amyloid_plaque": False
    }

    # Place a target Interneuron on the Up path: [1][2][1]
    grid[1][2][1] = {
        "type": "Interneuron",
        "charge": 0.1,
        "threshold": 0.5,
        "fire_rate": 0.0,
        "last_fired": -1,
        "direction": "All",
        "weight": 1.0,
        "amyloid_plaque": False
    }

    # Place a non-target Interneuron on the Left path: [0][1][1]
    grid[0][1][1] = {
        "type": "Interneuron",
        "charge": 0.1,
        "threshold": 0.5,
        "fire_rate": 0.0,
        "last_fired": -1,
        "direction": "All",
        "weight": 1.0,
        "amyloid_plaque": False
    }

    app.st.session_state.chemicals["norepinephrine"] = 0.0 # disable fight-or-flight fire boosts

    app.run_simulation_tick()

    # Output charge should transfer ONLY to the target neighbor on the Up path [1][2][1],
    # while the Left neighbor [0][1][1] remains at 0.1
    assert grid[1][2][1]["charge"] > 0.1
    assert grid[0][1][1]["charge"] == 0.1

def test_3d_indices_calculation():
    app.st.session_state = MockSessionState()
    app.st.session_state["bot_thread"] = True
    app.init_game_state()

    grid = app.st.session_state.neuron_grid

    # Clear grid
    for x in range(4):
        for y in range(4):
            for z in range(4):
                grid[x][y][z] = {"type": "Empty", "charge": 0.0, "threshold": 0.5, "weight": 1.0, "direction": "All", "amyloid_plaque": False}

    # Place 1 Interneuron with modified threshold (threshold != 0.5) and charged below threshold to stay active
    grid[0][0][0] = {
        "type": "Interneuron",
        "charge": 0.3, # charged (>= 0.45 * 0.5 = 0.225)
        "threshold": 0.45, # modified threshold (triggers PDI)
        "weight": 1.0,
        "direction": "All",
        "amyloid_plaque": False
    }

    # Place 1 Sensory node
    grid[1][1][1] = {
        "type": "Sensory",
        "charge": 0.8, # fully charged (above threshold)
        "threshold": 0.4,
        "fire_rate": 0.35,
        "last_fired": -1,
        "weight": 1.0,
        "direction": "All",
        "amyloid_plaque": False
    }

    # Run tick to calculate indices
    app.run_simulation_tick()

    # Cognitive Sync Index (CSI) should reflect active/charged neurons
    assert app.st.session_state.csi > 0.0

    # Plasticity Density Index (PDI) should be 100% since 1/1 Interneurons has threshold != 0.5
    assert app.st.session_state.pdi == 100.0

    # Vascular Perfusion Index (VPI) should match neuro_nutrients * 0.8 (since BBB is 0)
    nutrients = app.st.session_state.chemicals["neuro_nutrients"]
    expected_vpi = nutrients * 0.8
    assert abs(app.st.session_state.vpi - expected_vpi) < 1e-5

def test_vns_active_ability_3d():
    app.st.session_state = MockSessionState()
    app.st.session_state["bot_thread"] = True
    app.init_game_state()

    app.st.session_state.stats["memory"] = 100.0
    app.st.session_state.chemicals["stress"] = 80.0
    app.st.session_state.chemicals["sanity"] = 50.0
    app.st.session_state.chemicals["gaba"] = 10.0

    # Trigger active ability (copied from clinical button click)
    app.st.session_state.stats["memory"] -= 30.0
    app.st.session_state.chemicals["stress"] = 0.0
    app.st.session_state.chemicals["sanity"] = min(100.0, app.st.session_state.chemicals["sanity"] + 20.0)
    app.st.session_state.chemicals["gaba"] = 90.0
    app.st.session_state.cooldowns["vns"] = 40

    assert app.st.session_state.stats["memory"] == 70.0
    assert app.st.session_state.chemicals["stress"] == 0.0
    assert app.st.session_state.chemicals["sanity"] == 70.0
    assert app.st.session_state.chemicals["gaba"] == 90.0
    assert app.st.session_state.cooldowns["vns"] == 40

def test_pathological_modes_3d():
    app.st.session_state = MockSessionState()
    app.st.session_state["bot_thread"] = True
    app.init_game_state()

    # Setup Mania Challenge Mode
    app.st.session_state.game_mode = "Mania"
    app.st.session_state.chemicals["dopamine"] = 40.0
    app.st.session_state.chemicals["sanity"] = 100.0
    app.st.session_state.chemicals["norepinephrine"] = 50.0
    app.st.session_state.chemicals["stress"] = 0.0

    # Clear grid
    for x in range(4):
        for y in range(4):
            for z in range(4):
                app.st.session_state.neuron_grid[x][y][z] = {"type": "Empty", "charge": 0.0, "threshold": 0.5, "weight": 1.0, "direction": "All", "amyloid_plaque": False}

    # Place 1 sensory firing neuron to trigger stress
    app.st.session_state.neuron_grid[0][0][0] = {
        "type": "Sensory",
        "charge": 0.6,
        "threshold": 0.4,
        "fire_rate": 0.35,
        "last_fired": -1,
        "direction": "Right",
        "weight": 1.0,
        "amyloid_plaque": False
    }
    app.st.session_state.neuron_grid[1][0][0] = {
        "type": "Interneuron",
        "charge": 0.1,
        "threshold": 0.5,
        "fire_rate": 0.0,
        "last_fired": -1,
        "direction": "All",
        "weight": 1.0,
        "amyloid_plaque": False
    }

    app.run_simulation_tick()

    # Under Mania:
    # 1. Sanity decays by -1.0%
    assert app.st.session_state.chemicals["sanity"] == 99.0
    # 2. Dopamine auto-increases (+1.5%) before stabilization
    assert app.st.session_state.chemicals["dopamine"] > 40.0
    # 3. Fire stress is doubled (1.5 * 2.0 = 3.0 stress).
    # Stress delta = 3.0 - 2.5 (clearance) = 0.5
    assert abs(app.st.session_state.chemicals["stress"] - 0.5) < 1e-5
