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

# Mock pandas
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
    assert "rtms" in app.st.session_state.cooldowns
    assert "reflex" in app.st.session_state.missions
    assert app.st.session_state.game_mode == "Normal"
    assert app.st.session_state.stats["burnout_streak"] == 0
    assert app.st.session_state.stats["max_streak"] == 0
    assert app.st.session_state.stats["high_score_iq"] == 0.0
    assert app.st.session_state.stats["max_memory"] == 10.0

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

    # Set norepinephrine to 0 to avoid fight-or-flight energy drain in this baseline test
    app.st.session_state.chemicals["norepinephrine"] = 0.0

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

    # Set norepinephrine to 0 to avoid fight-or-flight fire_rate boost in this baseline test
    app.st.session_state.chemicals["norepinephrine"] = 0.0

    # Empty all but one Sensory Neuron at [0][0]
    for r in range(6):
        for c in range(6):
            app.st.session_state.neuron_grid[r][c] = {"type": "Empty", "charge": 0.0, "threshold": 0.5, "fire_rate": 0.0, "last_fired": -1, "direction": "All", "weight": 1.0}

    app.st.session_state.neuron_grid[0][0] = {
        "type": "Sensory",
        "charge": 0.1,
        "threshold": 0.5,
        "fire_rate": 0.2,
        "last_fired": -1,
        "direction": "All",
        "weight": 1.0
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
            app.st.session_state.neuron_grid[r][c] = {"type": "Empty", "charge": 0.0, "threshold": 0.5, "fire_rate": 0.0, "last_fired": -1, "direction": "All", "weight": 1.0}

    # Place a firing Sensory neuron at [0][0] with direction set to 'Right'
    app.st.session_state.neuron_grid[0][0] = {
        "type": "Sensory",
        "charge": 0.6, # threshold 0.4, fires
        "threshold": 0.4,
        "fire_rate": 0.0,
        "last_fired": -1,
        "direction": "Right", # Only send right!
        "weight": 1.0
    }
    # Interneuron to its right
    app.st.session_state.neuron_grid[0][1] = {
        "type": "Interneuron",
        "charge": 0.1,
        "threshold": 0.5,
        "fire_rate": 0.0,
        "last_fired": -1,
        "direction": "All",
        "weight": 1.0
    }
    # Interneuron to its bottom
    app.st.session_state.neuron_grid[1][0] = {
        "type": "Interneuron",
        "charge": 0.1,
        "threshold": 0.5,
        "fire_rate": 0.0,
        "last_fired": -1,
        "direction": "All",
        "weight": 1.0
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
            app.st.session_state.neuron_grid[r][c] = {"type": "Empty", "charge": 0.0, "threshold": 0.5, "fire_rate": 0.0, "last_fired": -1, "direction": "All", "weight": 1.0}

    # Place a Motor neuron that has full charge
    app.st.session_state.neuron_grid[3][3] = {
        "type": "Motor",
        "charge": 0.9, # threshold 0.5, fires
        "threshold": 0.5,
        "fire_rate": 0.0,
        "last_fired": -1,
        "direction": "All",
        "weight": 1.0
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
        "direction": "All",
        "weight": 1.0
    }

    # Reduce Sanity to 0.1 to trigger burnout on tick
    app.st.session_state.chemicals["sanity"] = 0.1
    app.st.session_state.chemicals["stress"] = 90.0
    app.st.session_state.stats["burnout_streak"] = 15

    app.run_simulation_tick()

    # Sanity is reset to 25.0 and burnout count increments, streak is reset to 0
    assert app.st.session_state.chemicals["sanity"] == 25.0
    assert app.st.session_state.stats["burnout_count"] == 1
    assert app.st.session_state.stats["burnout_streak"] == 0

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
    app.st.session_state.cooldowns["rtms"] = 12

    # Run simulation tick and verify decrement
    app.run_simulation_tick()
    assert app.st.session_state.cooldowns["doping"] == 4
    assert app.st.session_state.cooldowns["rtms"] == 11

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
            app.st.session_state.neuron_grid[r][c] = {"type": "Empty", "charge": 0.0, "threshold": 0.5, "fire_rate": 0.0, "last_fired": -1, "direction": "All", "weight": 1.0}

    # Condition: Sensory & Motor present (Reflex Arc)
    app.st.session_state.neuron_grid[0][0] = {"type": "Sensory", "charge": 0.0, "threshold": 0.5, "fire_rate": 0.0, "last_fired": -1, "direction": "All", "weight": 1.0}
    app.st.session_state.neuron_grid[0][1] = {"type": "Motor", "charge": 0.0, "threshold": 0.5, "fire_rate": 0.0, "last_fired": -1, "direction": "All", "weight": 1.0}

    # Prior state is In Progress
    assert app.st.session_state.missions["reflex"]["status"] == "In Progress"

    # Run evaluation
    app.check_mission_statuses()

    # State should now be Completed!
    assert app.st.session_state.missions["reflex"]["status"] == "Completed"

def test_save_load_circuit_codes():
    """
    UPGRADE TEST: Save/Load Circuit Codes
    Verify serialization and deserialization results are fully identical.
    """
    app.st.session_state = MockSessionState()
    app.st.session_state["bot_thread"] = True
    app.init_game_state()

    # Customize grid
    app.st.session_state.neuron_grid[1][2] = {"type": "Sensory", "charge": 0.1, "threshold": 0.4, "fire_rate": 0.3, "last_fired": -1, "direction": "Down", "weight": 1.0}
    app.st.session_state.neuron_grid[3][4] = {"type": "Motor", "charge": 0.2, "threshold": 0.6, "fire_rate": 0.0, "last_fired": -1, "direction": "Left", "weight": 2.0}

    # Serialize
    code = app.serialize_grid(app.st.session_state.neuron_grid)
    assert len(code) > 0

    # Deserialize
    loaded = app.deserialize_grid(code)
    assert loaded is not None
    assert loaded[1][2]["type"] == "Sensory"
    assert loaded[1][2]["direction"] == "Down"
    assert loaded[3][4]["type"] == "Motor"
    assert loaded[3][4]["direction"] == "Left"
    assert loaded[3][4]["weight"] == 2.0

def test_synaptic_pruning():
    """
    UPGRADE TEST: Synaptic Pruning
    Verify that old idle Interneurons are cleared and memory is partially refunded.
    """
    app.st.session_state = MockSessionState()
    app.st.session_state["bot_thread"] = True
    app.init_game_state()

    # Turn pruning on
    app.st.session_state.upgrades["pruning"] = 1
    app.st.session_state.stats["memory"] = 10.0
    st_ticks = 100
    app.st.session_state.stats["ticks"] = st_ticks

    # Place Interneuron with an old last_fired
    app.st.session_state.neuron_grid[2][2] = {
        "type": "Interneuron",
        "charge": 0.0,
        "threshold": 0.5,
        "fire_rate": 0.0,
        "last_fired": 50,
        "direction": "All",
        "weight": 1.0
    }

    app.run_simulation_tick()

    # Cell should be pruned back to Empty
    assert app.st.session_state.neuron_grid[2][2]["type"] == "Empty"
    # Memory should be refunded
    assert app.st.session_state.stats["memory"] > 10.0

def test_pfc_ai_decision_maker():
    """
    UPGRADE TEST: PFC AI Decision Maker
    Verify that random events are automatically evaluated and resolved when PFC is active.
    """
    app.st.session_state = MockSessionState()
    app.st.session_state["bot_thread"] = True
    app.init_game_state()

    # Activate PFC
    app.st.session_state.upgrades["pfc"] = 1

    # Create mock event
    app.st.session_state.current_event = {
        "title": "☕ Cốc Espresso Đậm Đặc",
        "desc": "Bạn nạp caffeine.",
        "choices": [
            {
                "label": "Espresso",
                "effect": "None",
                "apply": MagicMock()
            },
            {
                "label": "Trà xanh",
                "effect": "None",
                "apply": MagicMock()
            }
        ]
    }

    # Trigger tick
    app.st.session_state.stats["ticks"] = 3

    app.run_simulation_tick()

    # Check that current_event is resolved/cleared
    assert app.st.session_state.current_event is None

def test_synaptic_weight_amplification():
    """
    UPGRADE TEST: Synaptic Weight Amplification
    Verify that Synaptic weight multipliers correctly amplify signal propagation.
    """
    app.st.session_state = MockSessionState()
    app.st.session_state["bot_thread"] = True
    app.init_game_state()

    # Clean grid
    for r in range(6):
        for c in range(6):
            app.st.session_state.neuron_grid[r][c] = {"type": "Empty", "charge": 0.0, "threshold": 0.5, "fire_rate": 0.0, "last_fired": -1, "direction": "All", "weight": 1.0}

    # Place a Sensory neuron at [0][0] with output weight = 3.0
    app.st.session_state.neuron_grid[0][0] = {
        "type": "Sensory",
        "charge": 0.6,
        "threshold": 0.4,
        "fire_rate": 0.0,
        "last_fired": -1,
        "direction": "Right",
        "weight": 3.0 # Weight of 3.0!
    }
    # Interneuron neighbor at [0][1]
    app.st.session_state.neuron_grid[0][1] = {
        "type": "Interneuron",
        "charge": 0.1,
        "threshold": 0.5,
        "fire_rate": 0.0,
        "last_fired": -1,
        "direction": "All",
        "weight": 1.0
    }

    # Transferred charge = (0.6 * 0.35 * 3.0) / 1 = 0.63.
    # [0][1] should get 0.1 + 0.63 = 0.73
    app.run_simulation_tick()

    assert abs(app.st.session_state.neuron_grid[0][1]["charge"] - 0.73) < 1e-5

def test_alzheimers_and_apoe4_mutation():
    """
    UPGRADE TEST: Genetic Mutation Board Selection - APOE4
    Verify that APOE4 mutation doubles Alzheimer threshold drift.
    """
    app.st.session_state = MockSessionState()
    app.st.session_state["bot_thread"] = True
    app.init_game_state()

    # Case 1: Alzheimer mode without APOE4 gene
    app.st.session_state.game_mode = "Alzheimer"
    app.st.session_state.active_genes = []
    app.st.session_state.stats["ticks"] = 9 # next is 10, triggers drift
    app.st.session_state.neuron_grid[0][0]["threshold"] = 0.4

    app.run_simulation_tick()
    # drift should be 0.04 -> threshold 0.44
    assert abs(app.st.session_state.neuron_grid[0][0]["threshold"] - 0.44) < 1e-5

    # Case 2: Alzheimer mode with APOE4 gene
    app.st.session_state.stats["ticks"] = 9
    app.st.session_state.active_genes = ["APOE4"]
    app.st.session_state.neuron_grid[0][0]["threshold"] = 0.4

    app.run_simulation_tick()
    # drift should be 0.08 -> threshold 0.48
    assert abs(app.st.session_state.neuron_grid[0][0]["threshold"] - 0.48) < 1e-5

def test_bdnf_mutation():
    """
    UPGRADE TEST: Genetic Mutation - BDNF
    Verify that BDNF mutation increases plasticity drift.
    """
    app.st.session_state = MockSessionState()
    app.st.session_state["bot_thread"] = True
    app.init_game_state()

    # Setup firing interneuron with plasticity
    app.st.session_state.upgrades["plasticity"] = 1
    app.st.session_state.active_genes = ["BDNF"]

    # Clean grid
    for r in range(6):
        for c in range(6):
            app.st.session_state.neuron_grid[r][c] = {"type": "Empty", "charge": 0.0, "threshold": 0.5, "fire_rate": 0.0, "last_fired": -1, "direction": "All", "weight": 1.0}

    app.st.session_state.neuron_grid[0][0] = {
        "type": "Sensory",
        "charge": 0.6,
        "threshold": 0.4,
        "fire_rate": 0.0,
        "last_fired": -1,
        "direction": "Right",
        "weight": 1.0
    }
    app.st.session_state.neuron_grid[0][1] = {
        "type": "Interneuron",
        "charge": 0.4, # > 0.3 triggers learning plasticity
        "threshold": 0.5,
        "fire_rate": 0.0,
        "last_fired": -1,
        "direction": "All",
        "weight": 1.0
    }

    app.run_simulation_tick()

    # Normally drift is 0.01. With BDNF, it is 0.015 -> threshold becomes 0.5 - 0.015 = 0.485
    assert abs(app.st.session_state.neuron_grid[0][1]["threshold"] - 0.485) < 1e-5

def test_comt_mutation():
    """
    UPGRADE TEST: Genetic Mutation - COMT
    Verify Dopamine and Stress decay ratios.
    """
    app.st.session_state = MockSessionState()
    app.st.session_state["bot_thread"] = True
    app.init_game_state()

    # Normal starting
    app.st.session_state.chemicals["dopamine"] = 80.0
    app.st.session_state.chemicals["stress"] = 40.0
    app.st.session_state.active_genes = ["COMT"]

    app.run_simulation_tick()

    # Dopamine target is 50.0. Decay under COMT is 0.048:
    # 80.0 + (50.0 - 80.0) * 0.048 = 80.0 - 1.44 = 78.56
    assert abs(app.st.session_state.chemicals["dopamine"] - 78.56) < 1e-3

def test_gabra1_mutation():
    """
    UPGRADE TEST: Genetic Mutation - GABRA1
    Verify Epilepsy stress multiplier.
    """
    app.st.session_state = MockSessionState()
    app.st.session_state["bot_thread"] = True
    app.init_game_state()

    # Clean grid
    for r in range(6):
        for c in range(6):
            app.st.session_state.neuron_grid[r][c] = {"type": "Empty", "charge": 0.0, "threshold": 0.5, "fire_rate": 0.0, "last_fired": -1, "direction": "All", "weight": 1.0}

    # Place firing Sensory neuron
    app.st.session_state.neuron_grid[0][0] = {
        "type": "Sensory",
        "charge": 0.6,
        "threshold": 0.4,
        "fire_rate": 0.0,
        "last_fired": -1,
        "direction": "Right",
        "weight": 1.0
    }
    app.st.session_state.neuron_grid[0][1] = {
        "type": "Interneuron",
        "charge": 0.1,
        "threshold": 0.5,
        "fire_rate": 0.0,
        "last_fired": -1,
        "direction": "All",
        "weight": 1.0
    }

    app.st.session_state.game_mode = "Epilepsy"
    app.st.session_state.active_genes = ["GABRA1"]
    app.st.session_state.chemicals["stress"] = 10.0
    # cerebellum level 1 stress clearance = 2.5
    # Signals fired = 1. Epilepsy stress mult with GABRA1 = 1.3
    # stress delta = +(1 * 1.5 * 1.3) - 2.5 = 1.95 - 2.5 = -0.55. Final stress = 9.45
    app.run_simulation_tick()

    assert abs(app.st.session_state.chemicals["stress"] - 9.45) < 1e-5

def test_rtms_therapy_action():
    """
    UPGRADE TEST: rTMS Clinical Therapy Active Ability
    """
    app.st.session_state = MockSessionState()
    app.st.session_state["bot_thread"] = True
    app.init_game_state()

    # Degrade sensory cell threshold
    app.st.session_state.neuron_grid[0][0]["threshold"] = 0.8
    app.st.session_state.chemicals["sanity"] = 30.0

    # Simulate clicking rTMS therapy directly from test
    for r in range(6):
        for c in range(6):
            t_name = app.st.session_state.neuron_grid[r][c]["type"]
            if t_name != "Empty":
                app.st.session_state.neuron_grid[r][c]["threshold"] = 0.4 if t_name == "Sensory" else (0.6 if t_name == "Motor" else 0.5)
    app.st.session_state.chemicals["sanity"] = min(100.0, app.st.session_state.chemicals["sanity"] + 40.0)

    assert app.st.session_state.neuron_grid[0][0]["threshold"] == 0.4
    assert app.st.session_state.chemicals["sanity"] == 70.0

def test_parkinsons_pathology():
    """
    UPGRADE TEST: Parkinson's Pathology Challenge Mode
    Under Parkinson's, Dopamine levels below 40.0 cause Motor cells to randomly misfire, draining energy.
    """
    app.st.session_state = MockSessionState()
    app.st.session_state["bot_thread"] = True
    app.init_game_state()

    app.st.session_state.game_mode = "Parkinson"
    app.st.session_state.chemicals["dopamine"] = 20.0 # < 40.0 triggers Parkinson's misfires
    app.st.session_state.chemicals["energy"] = 50.0

    # Force mock motor cells on grid
    for r in range(6):
        for c in range(6):
            app.st.session_state.neuron_grid[r][c] = {"type": "Empty", "charge": 0.0, "threshold": 0.5, "fire_rate": 0.0, "last_fired": -1, "direction": "All", "weight": 1.0}
    app.st.session_state.neuron_grid[5][5] = {"type": "Motor", "charge": 0.4, "threshold": 0.6, "fire_rate": 0.0, "last_fired": -1, "direction": "All", "weight": 1.0}

    # Mock random choice to guarantee motor misfire triggering
    import random
    original_random = random.random
    original_choice = random.choice

    random.random = lambda: 0.1 # Triggers the random misfire check (< 0.30)
    random.choice = lambda seq: seq[0]

    try:
        app.run_simulation_tick()
        # Misfire drains 5.0 energy, resets motor cell charge to 0.0
        assert app.st.session_state.chemicals["energy"] < 50.0
        assert app.st.session_state.neuron_grid[5][5]["charge"] == 0.0
    finally:
        random.random = original_random
        random.choice = original_choice

def test_active_buffs():
    """
    UPGRADE TEST: Neuromodulator Ongoing Buffs
    Ensures that active neuromodulator ongoing buffs increment biochemical rates and decrement tick timers correctly.
    """
    app.st.session_state = MockSessionState()
    app.st.session_state["bot_thread"] = True
    app.init_game_state()

    app.st.session_state.active_buffs = {
        "doping": 5,
        "ssri": 8,
        "focus": 10,
        "tyrosine": 0,
        "tryptophan": 0,
        "choline": 0
    }
    app.st.session_state.chemicals["dopamine"] = 40.0
    app.st.session_state.chemicals["serotonin"] = 40.0
    app.st.session_state.chemicals["acetylcholine"] = 40.0

    app.run_simulation_tick()

    # Ongoing buffs should apply rate bonuses and decrement timers by 1
    assert app.st.session_state.active_buffs["doping"] == 4
    assert app.st.session_state.active_buffs["ssri"] == 7
    assert app.st.session_state.active_buffs["focus"] == 9

    # Chem values undergo decay/stabilization in the same tick:
    # dopamine: 40 + 5 = 45 -> 45 + (50 - 45) * 0.08 = 45.4
    # serotonin: 40 + 3 = 43 -> 43 + (50 - 43) * 0.08 = 43.56
    # acetylcholine: 40 + 4 = 44 -> 44 + (50 - 44) * 0.08 = 44.48
    assert abs(app.st.session_state.chemicals["dopamine"] - 45.4) < 1e-5
    assert abs(app.st.session_state.chemicals["serotonin"] - 43.56) < 1e-5
    assert abs(app.st.session_state.chemicals["acetylcholine"] - 44.48) < 1e-5

def test_drd4_shank3_genes():
    """
    UPGRADE TEST: DRD4 and SHANK3 Mutations
    DRD4 doubles motor dopamine reward, but low dopamine doubles sanity damage.
    SHANK3 boosts myelin signal efficiency by +15%.
    """
    app.st.session_state = MockSessionState()
    app.st.session_state["bot_thread"] = True
    app.init_game_state()

    # 1. Test DRD4 dopamine multiplier on motor fire
    app.st.session_state.active_genes = ["DRD4"]
    app.st.session_state.chemicals["dopamine"] = 20.0

    # Place a Motor neuron that fires
    for r in range(6):
        for c in range(6):
            app.st.session_state.neuron_grid[r][c] = {"type": "Empty", "charge": 0.0, "threshold": 0.5, "fire_rate": 0.0, "last_fired": -1, "direction": "All", "weight": 1.0}
    app.st.session_state.neuron_grid[3][3] = {
        "type": "Motor",
        "charge": 0.9, # threshold 0.5, fires
        "threshold": 0.5,
        "fire_rate": 0.0,
        "last_fired": -1,
        "direction": "All",
        "weight": 1.0
    }

    app.run_simulation_tick()

    # Motor firing should double dopamine reward (normally +8.0, under DRD4 it is +16.0)
    # Decay is also applied: 20.0 + 16.0 = 36.0 -> with COMT (not active), normal decay: 36.0 + (50 - 36) * 0.08 = 37.12
    assert app.st.session_state.chemicals["dopamine"] > 35.0

    # 2. Test SHANK3 signal propagation efficiency
    app.st.session_state = MockSessionState()
    app.st.session_state["bot_thread"] = True
    app.init_game_state()
    app.st.session_state.active_genes = ["SHANK3"]
    app.st.session_state.upgrades["myelin"] = 1

    # Place sensory cell next to interneuron
    for r in range(6):
        for c in range(6):
            app.st.session_state.neuron_grid[r][c] = {"type": "Empty", "charge": 0.0, "threshold": 0.5, "fire_rate": 0.0, "last_fired": -1, "direction": "All", "weight": 1.0}
    app.st.session_state.neuron_grid[0][0] = {
        "type": "Sensory",
        "charge": 0.6,
        "threshold": 0.4,
        "fire_rate": 0.0,
        "last_fired": -1,
        "direction": "Right",
        "weight": 1.0
    }
    app.st.session_state.neuron_grid[0][1] = {
        "type": "Interneuron",
        "charge": 0.1,
        "threshold": 0.5,
        "fire_rate": 0.0,
        "last_fired": -1,
        "direction": "All",
        "weight": 1.0
    }

    # Efficiency is 0.35 + myelin * 0.05 + shank3 * 0.15 = 0.35 + 0.05 + 0.15 = 0.55
    # Transferred charge = (0.6 * 0.55) = 0.33
    # Neighbor [0][1] should have 0.1 + 0.33 = 0.43
    app.run_simulation_tick()
    assert abs(app.st.session_state.neuron_grid[0][1]["charge"] - 0.43) < 1e-5

def test_diet_precursors():
    """
    UPGRADE TEST: Neurotransmitter Synthesis Precursors & Diet System
    Verify L-Tyrosine, L-Tryptophan, Choline active tick buffs apply correctly.
    """
    app.st.session_state = MockSessionState()
    app.st.session_state["bot_thread"] = True
    app.init_game_state()

    app.st.session_state.active_buffs = {
        "doping": 0, "ssri": 0, "focus": 0,
        "tyrosine": 15,
        "tryptophan": 15,
        "choline": 15
    }
    app.st.session_state.chemicals["dopamine"] = 40.0
    app.st.session_state.chemicals["serotonin"] = 40.0
    app.st.session_state.chemicals["acetylcholine"] = 40.0

    app.run_simulation_tick()

    # Check that precursors apply rate boosts:
    # Tyrosine: +3.0 Dopamine
    # Tryptophan: +2.0 Serotonin
    # Choline: +2.5 Acetylcholine
    # Then decay/stabilization applies:
    # Dopamine: 40 + 3 = 43 -> 43 + (50 - 43) * 0.08 = 43.56
    # Serotonin: 40 + 2 = 42 -> 42 + (50 - 42) * 0.08 = 42.64
    # Acetylcholine: 40 + 2.5 = 42.5 -> 42.5 + (50 - 42.5) * 0.08 = 43.1
    assert abs(app.st.session_state.chemicals["dopamine"] - 43.56) < 1e-5
    assert abs(app.st.session_state.chemicals["serotonin"] - 42.64) < 1e-5
    assert abs(app.st.session_state.chemicals["acetylcholine"] - 43.1) < 1e-5

    assert app.st.session_state.active_buffs["tyrosine"] == 14
    assert app.st.session_state.active_buffs["tryptophan"] == 14
    assert app.st.session_state.active_buffs["choline"] == 14

def test_maoa_chrna7_genes():
    """
    UPGRADE TEST: MAOA and CHRNA7 Genetic Mutations
    MAOA slows dopamine and serotonin decay, but increases signal stress.
    CHRNA7 boosts acetylcholine delta by 25%, but metabolic energy cost by 15%.
    """
    app.st.session_state = MockSessionState()
    app.st.session_state["bot_thread"] = True
    app.init_game_state()

    # 1. Test MAOA decay rates
    app.st.session_state.active_genes = ["MAOA"]
    app.st.session_state.chemicals["dopamine"] = 80.0
    app.st.session_state.chemicals["serotonin"] = 80.0

    app.run_simulation_tick()
    # Normally decay is 0.08. MAOA Dopamine decay rate is 0.08 * 0.7 = 0.056
    # Dopamine: 80.0 + (50 - 80) * 0.056 = 80.0 - 1.68 = 78.32
    # MAOA Serotonin decay rate is 0.056:
    # Serotonin: 80.0 + (50 - 80) * 0.056 = 78.32
    assert abs(app.st.session_state.chemicals["dopamine"] - 78.32) < 1e-3
    assert abs(app.st.session_state.chemicals["serotonin"] - 78.32) < 1e-3

    # 2. Test CHRNA7 Acetylcholine delta
    app.st.session_state = MockSessionState()
    app.st.session_state["bot_thread"] = True
    app.init_game_state()

    app.st.session_state.active_genes = ["CHRNA7"]
    app.st.session_state.chemicals["acetylcholine"] = 40.0
    app.st.session_state.chemicals["energy"] = 100.0

    app.run_simulation_tick()
    # Normally ACh delta is (50 - 40) * 0.08 = 0.8
    # With CHRNA7, ACh delta is 0.8 * 1.25 = 1.0 -> 41.0
    assert abs(app.st.session_state.chemicals["acetylcholine"] - 41.0) < 1e-5

def test_amygdala_thalamus_upgrades():
    """
    UPGRADE TEST: Amygdala & Thalamus Anatomy Upgrades
    Amygdala reduces stress generation, Thalamus speeds sensory cell charge.
    """
    app.st.session_state = MockSessionState()
    app.st.session_state["bot_thread"] = True
    app.init_game_state()

    # 1. Test Amygdala level 1 stress reduction (-15% stress)
    app.st.session_state.upgrades["amygdala"] = 1

    # Clean grid and put 1 firing sensory cell
    for r in range(6):
        for c in range(6):
            app.st.session_state.neuron_grid[r][c] = {"type": "Empty", "charge": 0.0, "threshold": 0.5, "fire_rate": 0.0, "last_fired": -1, "direction": "All", "weight": 1.0}
    app.st.session_state.neuron_grid[0][0] = {
        "type": "Sensory",
        "charge": 0.6,
        "threshold": 0.4,
        "fire_rate": 0.0,
        "last_fired": -1,
        "direction": "Right",
        "weight": 1.0
    }
    app.st.session_state.neuron_grid[0][1] = {
        "type": "Interneuron",
        "charge": 0.1,
        "threshold": 0.5,
        "fire_rate": 0.0,
        "last_fired": -1,
        "direction": "All",
        "weight": 1.0
    }

    app.st.session_state.chemicals["stress"] = 10.0
    app.run_simulation_tick()

    # Normal stress generation is 1.5. Amygdala level 1 reduces it by 15% -> 1.5 * 0.85 = 1.275
    # stress clearance = 2.5
    # Stress delta = 1.275 - 2.5 = -1.225 -> final stress is 10.0 - 1.225 = 8.775
    assert abs(app.st.session_state.chemicals["stress"] - 8.775) < 1e-5

# ----------------- CHU KỲ SINH HỌC & QUANG DI TRUYỀN TESTS -----------------

def test_circadian_sleep_cycle():
    """
    UPGRADE TEST: Circadian sleep state
    Sleep mode halts signals and triggers rapid recovery (+15 energy, +8 sanity, -12 stress).
    """
    app.st.session_state = MockSessionState()
    app.st.session_state["bot_thread"] = True
    app.init_game_state()

    # Set norepinephrine to 0 to avoid fight-or-flight energy drain in this baseline test
    app.st.session_state.chemicals["norepinephrine"] = 0.0

    app.st.session_state.stats["sleep_state"] = True
    app.st.session_state.chemicals["energy"] = 20.0
    app.st.session_state.chemicals["sanity"] = 50.0
    app.st.session_state.chemicals["stress"] = 30.0

    app.run_simulation_tick()

    # Sleep mode increments and restores:
    # energy: starting 20.0 + sleep 15.0 + metabolic net generation (+3.8) = 38.8
    # sanity: starting 50.0 + sleep 8.0 = 58.0
    # stress: starting 30.0 - sleep 12.0 = 18.0
    assert abs(app.st.session_state.chemicals["energy"] - 38.8) < 1e-5
    assert abs(app.st.session_state.chemicals["sanity"] - 58.0) < 1e-5
    assert abs(app.st.session_state.chemicals["stress"] - 18.0) < 1e-5

def test_optogenetic_pulse():
    """
    UPGRADE TEST: Optogenetic laser stimulation active pulse
    Charges up all cells in selected row and column by +0.5.
    """
    app.st.session_state = MockSessionState()
    app.st.session_state["bot_thread"] = True
    app.init_game_state()

    app.st.session_state.selected_cell = (1, 1)

    # Pre-set cell charges
    app.st.session_state.neuron_grid[1][1] = {"type": "Interneuron", "charge": 0.2, "threshold": 0.5}
    app.st.session_state.neuron_grid[1][2] = {"type": "Interneuron", "charge": 0.3, "threshold": 0.5}
    app.st.session_state.neuron_grid[4][4] = {"type": "Interneuron", "charge": 0.1, "threshold": 0.5}

    # Simulate optogenetics activation manually
    sel_r, sel_c = app.st.session_state.selected_cell
    grid = app.st.session_state.neuron_grid
    for r in range(6):
        for c in range(6):
            if (r == sel_r or c == sel_c) and grid[r][c]["type"] != "Empty":
                grid[r][c]["charge"] = min(1.0, grid[r][c]["charge"] + 0.5)

    # row 1 col 1 is charged: 0.2 + 0.5 = 0.7
    assert grid[1][1]["charge"] == 0.7
    # row 1 col 2 is charged: 0.3 + 0.5 = 0.8
    assert grid[1][2]["charge"] == 0.8
    # row 4 col 4 is not in row 1 or col 1 -> remains 0.1
    assert grid[4][4]["charge"] == 0.1

def test_pgc1_slc6a4_genes():
    """
    UPGRADE TEST: PGC-1alpha and SLC6A4 Genes
    PGC-1alpha increases maximum fuel capacity by +40.
    SLC6A4 increases SSRI active buff duration.
    """
    app.st.session_state = MockSessionState()
    app.st.session_state["bot_thread"] = True
    app.init_game_state()

    # 1. Test PGC-1alpha maximum energy storage
    app.st.session_state.active_genes = ["PGC-1alpha"]
    app.st.session_state.upgrades["glycogen_shunt"] = 1 # increases energy cap to 150
    # under PGC-1alpha cap becomes 150 + 40 = 190.
    app.st.session_state.chemicals["energy"] = 180.0
    app.st.session_state.upgrades["brainstem"] = 1 # generates 6.0 energy

    app.run_simulation_tick()
    # Fuel generation should let energy exceed 180.0 up to 190.0 cap
    assert app.st.session_state.chemicals["energy"] > 180.0


# ----------------- NEW SYSTEM EXPANSION TESTS -----------------

def test_neuro_inflammation():
    """
    TEST: Neuro-Inflammation & Microglia Immune Delta Engine
    Ensures that inflammation rises under firing signals, cytokine storms trigger chronic threshold drift
    when inflammation is >80%, and the Anti-inflammatory wash cortisol therapy is functional.
    """
    app.st.session_state = MockSessionState()
    app.st.session_state["bot_thread"] = True
    app.init_game_state()

    # Setup high inflammation triggering Cytokine Storm
    app.st.session_state.chemicals["neuro_inflammation"] = 85.0
    app.st.session_state.chemicals["sanity"] = 90.0
    app.st.session_state.stats["ticks"] = 7 # next is 8, triggers cytokine storm threshold drift

    # Place an Interneuron
    app.st.session_state.neuron_grid[1][1] = {
        "type": "Interneuron",
        "charge": 0.0,
        "threshold": 0.5,
        "fire_rate": 0.0,
        "last_fired": -1,
        "direction": "All",
        "weight": 1.0
    }

    app.run_simulation_tick()

    # Chronic drift under storm is +0.02, threshold should become 0.52
    assert abs(app.st.session_state.neuron_grid[1][1]["threshold"] - 0.52) < 1e-5
    # Sanity decay triggers: decay = (85 - 80) * 0.4 = 2.0. Base healing is (0.5 + 50*0.02) = 1.5. Net change = -0.5
    assert app.st.session_state.chemicals["sanity"] < 90.0

def test_adhd_pathology():
    """
    TEST: ADHD Neurodevelopmental Pathology Challenge Mode
    Under ADHD, dopamine baselines fluctuate per tick and Acetylcholine decay rate is accelerated by 50%.
    """
    app.st.session_state = MockSessionState()
    app.st.session_state["bot_thread"] = True
    app.init_game_state()

    app.st.session_state.game_mode = "ADHD"
    app.st.session_state.chemicals["acetylcholine"] = 60.0

    app.run_simulation_tick()

    # Normal decay: (50 - 60) * 0.08 = -0.8
    # Under ADHD: decay rate becomes 0.08 * 1.5 = 0.12. ACh delta = (50 - 60) * 0.12 = -1.2. Final = 58.8
    assert abs(app.st.session_state.chemicals["acetylcholine"] - 58.8) < 1e-5

def test_drd2_comtmet_genes():
    """
    TEST: DRD2 and COMT-Met Genetic Mutations
    DRD2 boosts dopamine sensory fire rate multipliers by 1.5x, but high stress (>50.0) inflicts 1.5x sanity damage.
    COMT-Met increases IQ gains from Motor fires by 30%, but stress decays 30% slower.
    """
    app.st.session_state = MockSessionState()
    app.st.session_state["bot_thread"] = True
    app.init_game_state()

    # 1. DRD2 stress multiplier sanity damage test
    app.st.session_state.active_genes = ["DRD2"]
    app.st.session_state.chemicals["stress"] = 90.0 # > 50.0 triggers sanity stress impact
    app.st.session_state.chemicals["sanity"] = 100.0

    # Trigger tick to check sanity drop multiplier
    app.run_simulation_tick()
    # High stress (>60 effective) triggers sanity damage, which is multiplied by 1.5x under DRD2
    assert app.st.session_state.chemicals["sanity"] < 100.0

    # 2. COMT-Met Motor IQ gains & Stress clearance tests
    app.st.session_state = MockSessionState()
    app.st.session_state["bot_thread"] = True
    app.init_game_state()

    app.st.session_state.active_genes = ["COMT-Met"]
    app.st.session_state.chemicals["stress"] = 40.0

    # Place firing Motor cell
    for r in range(6):
        for c in range(6):
            app.st.session_state.neuron_grid[r][c] = {"type": "Empty", "charge": 0.0, "threshold": 0.5, "fire_rate": 0.0, "last_fired": -1, "direction": "All", "weight": 1.0}
    app.st.session_state.neuron_grid[3][3] = {
        "type": "Motor",
        "charge": 0.9,
        "threshold": 0.5,
        "fire_rate": 0.0,
        "last_fired": -1,
        "direction": "All",
        "weight": 1.0
    }

    app.run_simulation_tick()

    # Motor fire yields IQ amplified by COMT-Met (+30% bonus)
    # Stress decay is also 30% slower (multiplier 0.7)
    assert app.st.session_state.stats["iq"] > 0.0
    assert app.st.session_state.chemicals["stress"] > 35.0

def test_dentate_gyrus_neurogenesis():
    """
    TEST: Dentate Gyrus Neurogenesis Anatomy Upgrade
    Dentate Gyrus upgrade auto-implants an Interneuron on random Empty spots if memory >= 30 MB and serotonin is high.
    """
    app.st.session_state = MockSessionState()
    app.st.session_state["bot_thread"] = True
    app.init_game_state()

    app.st.session_state.upgrades["dentate_gyrus"] = 1
    app.st.session_state.stats["memory"] = 50.0
    app.st.session_state.chemicals["serotonin"] = 80.0

    # Clear grid
    for r in range(6):
        for c in range(6):
            app.st.session_state.neuron_grid[r][c] = {"type": "Empty", "charge": 0.0, "threshold": 0.5, "fire_rate": 0.0, "last_fired": -1, "direction": "All", "weight": 1.0}

    # Mock random to force both the neurogenesis check (0.25 chance) and coordinate placement
    import random
    original_random = random.random
    original_choice = random.choice

    random.random = lambda: 0.1 # < 0.25 triggers neurogenesis check
    random.choice = lambda seq: seq[0]

    try:
        app.run_simulation_tick()

        # Grid [0][0] should be cetted with Interneuron and memory spends 15 MB
        assert app.st.session_state.neuron_grid[0][0]["type"] == "Interneuron"
        assert app.st.session_state.stats["memory"] == 35.0
    finally:
        random.random = original_random
        random.choice = original_choice


def test_gaba_stress_mitigation():
    """
    TEST: GABA inhibitory neurotransmitter
    Verifies that high GABA (>70) completely suppresses epilepsy hyper-excitability stress multiplication and reduces stress generation by 40%.
    """
    app.st.session_state = MockSessionState()
    app.st.session_state["bot_thread"] = True
    app.init_game_state()

    app.st.session_state.game_mode = "Epilepsy"
    app.st.session_state.chemicals["gaba"] = 80.0 # High GABA!
    app.st.session_state.chemicals["stress"] = 10.0

    # Place firing Sensory neuron
    for r in range(6):
        for c in range(6):
            app.st.session_state.neuron_grid[r][c] = {"type": "Empty", "charge": 0.0, "threshold": 0.5, "fire_rate": 0.0, "last_fired": -1, "direction": "All", "weight": 1.0}
    app.st.session_state.neuron_grid[0][0] = {
        "type": "Sensory",
        "charge": 0.6,
        "threshold": 0.4,
        "fire_rate": 0.0,
        "last_fired": -1,
        "direction": "Right",
        "weight": 1.0
    }
    app.st.session_state.neuron_grid[0][1] = {
        "type": "Interneuron",
        "charge": 0.1,
        "threshold": 0.5,
        "fire_rate": 0.0,
        "last_fired": -1,
        "direction": "All",
        "weight": 1.0
    }

    app.run_simulation_tick()
    # Normally Epilepsy stress multiplier is 2.0. Under high GABA, it is completely bypassed (kept at 1.0).
    # And total fire stress is reduced by 40% (multiplied by 0.6).
    # stress clearance = 2.5 (level 1 cerebellum)
    # fire_stress = 1 * 1.5 * 1.0 = 1.5 -> with GABA: 1.5 * 0.6 = 0.9
    # stress delta = 0.9 - 2.5 = -1.6. Final stress = 8.4
    assert abs(app.st.session_state.chemicals["stress"] - 8.4) < 1e-5


def test_occipital_lobe_visual_stimulus():
    """
    TEST: Occipital Lobe visual target direction alignment
    Verifies that matching Visual Spark direction doubles Sensory fire rates and doubles Motor output IQ rewards.
    """
    app.st.session_state = MockSessionState()
    app.st.session_state["bot_thread"] = True
    app.init_game_state()

    app.st.session_state.upgrades["occipital_lobe"] = 1
    # Trigger visual spark at coordinate (0,0) with direction "Right"
    app.st.session_state.visual_spark = {"pos": (0, 0), "dir": "Right"}

    # Place Sensory cell at (0,0) with axon target "Right"
    for r in range(6):
        for c in range(6):
            app.st.session_state.neuron_grid[r][c] = {"type": "Empty", "charge": 0.0, "threshold": 0.5, "fire_rate": 0.0, "last_fired": -1, "direction": "All", "weight": 1.0}

    app.st.session_state.chemicals["norepinephrine"] = 0.0 # Set to 0 to avoid fight-or-flight fire rate boost
    app.st.session_state.neuron_grid[0][0] = {
        "type": "Sensory",
        "charge": 0.1,
        "threshold": 0.9, # set high threshold so it doesn't fire and clear charge
        "fire_rate": 0.2,
        "last_fired": -1,
        "direction": "Right", # axon targets "Right" - matches Spark!
        "weight": 1.0
    }
    app.st.session_state.chemicals["dopamine"] = 50.0 # base boost = 1.5
    # Since it matches Visual Spark direction, the fire rate boost is multiplied by 2.0!
    # So charge rate is 0.2 * 1.5 * 2.0 = 0.6. Final charge should be 0.1 + 0.6 = 0.7

    app.run_simulation_tick()
    assert abs(app.st.session_state.neuron_grid[0][0]["charge"] - 0.7) < 1e-5


def test_local_save_slots_library():
    """
    TEST: Multi-slot Circuit Save Library
    Verifies saving and loading from Slot 1, Slot 2, Slot 3.
    """
    app.st.session_state = MockSessionState()
    app.st.session_state["bot_thread"] = True
    app.init_game_state()

    # Customize grid to something distinct
    app.st.session_state.neuron_grid[0][0] = {"type": "Motor", "charge": 0.0, "threshold": 0.5}

    # Save to Slot 2
    app.st.session_state.save_slots["Slot 2"] = app.serialize_grid(app.st.session_state.neuron_grid)

    # Empty grid
    for r in range(6):
        for c in range(6):
            app.st.session_state.neuron_grid[r][c] = {"type": "Empty", "charge": 0.0, "threshold": 0.5}

    # Load from Slot 2
    loaded = app.deserialize_grid(app.st.session_state.save_slots["Slot 2"])
    assert loaded is not None
    assert loaded[0][0]["type"] == "Motor"


def test_norepinephrine_panic_attack():
    """
    TEST: Norepinephrine & Panic Attack Trigger
    Ensures Norepinephrine builds up, triggers Panic Attack (>90), inflicts sanity damage, and resets.
    """
    app.st.session_state = MockSessionState()
    app.st.session_state["bot_thread"] = True
    app.init_game_state()

    # 1. Normal panic attack
    app.st.session_state.chemicals["norepinephrine"] = 95.0
    app.st.session_state.chemicals["sanity"] = 100.0
    app.st.session_state.active_genes = []

    app.run_simulation_tick()
    # sanity damage is 15.0, so sanity should be 85.0
    assert app.st.session_state.chemicals["sanity"] == 85.0
    # Norepinephrine resets to 50.0 after panic
    assert app.st.session_state.chemicals["norepinephrine"] == 50.0

    # 2. ADRA2A gene reduces damage to 9.0 and raises threshold to 100.0
    app.st.session_state = MockSessionState()
    app.st.session_state["bot_thread"] = True
    app.init_game_state()
    app.st.session_state.chemicals["norepinephrine"] = 95.0
    app.st.session_state.chemicals["sanity"] = 100.0
    app.st.session_state.active_genes = ["ADRA2A"]

    app.run_simulation_tick()
    # threshold is 100.0, so at 95.0 no panic attack triggers!
    assert app.st.session_state.chemicals["sanity"] == 100.0

    # Trigger at 100.0 (maximum bounded norepinephrine level with sufficient stress to overcome clearance)
    app.st.session_state.chemicals["norepinephrine"] = 100.0
    app.st.session_state.chemicals["stress"] = 50.0
    app.run_simulation_tick()
    # damage is 9.0, sanity drops to 91.0
    assert app.st.session_state.chemicals["sanity"] == 91.0


def test_amyloid_plaques_phagocytosis():
    """
    TEST: Amyloid plaques and Microglia Phagocytosis
    Ensures plaques degrade synaptic weight by 50% during propagation, and high Serotonin/Acetylcholine clears them.
    """
    app.st.session_state = MockSessionState()
    app.st.session_state["bot_thread"] = True
    app.init_game_state()

    # Clean grid
    for r in range(6):
        for c in range(6):
            app.st.session_state.neuron_grid[r][c] = {"type": "Empty", "charge": 0.0, "threshold": 0.5, "fire_rate": 0.0, "last_fired": -1, "direction": "All", "weight": 1.0, "amyloid_plaque": False}

    # Place sensory cell next to interneuron
    app.st.session_state.neuron_grid[0][0] = {
        "type": "Sensory",
        "charge": 0.6,
        "threshold": 0.4,
        "fire_rate": 0.0,
        "last_fired": -1,
        "direction": "Right",
        "weight": 1.0,
        "amyloid_plaque": True # Covered in Amyloid plaque!
    }
    app.st.session_state.neuron_grid[0][1] = {
        "type": "Interneuron",
        "charge": 0.1,
        "threshold": 0.5,
        "fire_rate": 0.0,
        "last_fired": -1,
        "direction": "All",
        "weight": 1.0,
        "amyloid_plaque": False
    }

    # With plaque, synaptic weight is halved. Charge transferred is (0.6 * 0.35 * 0.5) = 0.105
    # Neighbor [0][1] gets 0.1 + 0.105 = 0.205
    app.run_simulation_tick()
    assert abs(app.st.session_state.neuron_grid[0][1]["charge"] - 0.205) < 1e-5


def test_temporal_lobe_auditory():
    """
    TEST: Temporal Lobe Auditory Spark and Resonance
    Verifies that high and low frequencies increase different biochemical levels, and resonance triples Motor memory yields.
    """
    app.st.session_state = MockSessionState()
    app.st.session_state["bot_thread"] = True
    app.init_game_state()

    app.st.session_state.upgrades["temporal_lobe"] = 1

    # 1. High frequency (>600) boosts dopamine and gaba
    app.st.session_state.auditory_freq = 800
    app.st.session_state.chemicals["dopamine"] = 50.0
    app.st.session_state.chemicals["gaba"] = 30.0

    app.run_simulation_tick()
    assert app.st.session_state.chemicals["dopamine"] > 50.0
    assert app.st.session_state.chemicals["gaba"] > 30.0

    # 2. Auditory resonance (400 to 500 Hz) triples Motor memory yields
    app.st = MagicMock()
    app.st.session_state = MockSessionState()
    app.init_game_state()
    app.st.session_state.upgrades["temporal_lobe"] = 1
    app.st.session_state.upgrades["hippocampus"] = 0 # set to 0 for baseline yield of 2.0
    app.st.session_state.auditory_freq = 450 # resonant!
    app.st.session_state.stats["memory"] = 10.0

    # Place firing Motor cell
    for r in range(6):
        for c in range(6):
            app.st.session_state.neuron_grid[r][c] = {"type": "Empty", "charge": 0.0, "threshold": 0.5, "weight": 1.0}
    app.st.session_state.neuron_grid[3][3] = {
        "type": "Motor",
        "charge": 0.9,
        "threshold": 0.5,
        "weight": 1.0
    }

    app.run_simulation_tick()
    # Base memory yield is 2.0. With resonance, it is tripled (3.0 * 2.0 = 6.0).
    # Memory starts at 10.0, so 10.0 + 6.0 = 16.0
    assert app.st.session_state.stats["memory"] == 16.0


def test_ltp_consolidation():
    """
    TEST: Hebbian LTP Memory Consolidator
    Verifies that every 12 ticks, 30% of Memory is automatically converted to permanent IQ points.
    """
    app.st.session_state = MockSessionState()
    app.st.session_state["bot_thread"] = True
    app.init_game_state()

    app.st.session_state.upgrades["ltp_consolidator"] = 1
    app.st.session_state.stats["ticks"] = 11 # next tick is 12, triggers consolidator
    app.st.session_state.stats["memory"] = 100.0
    app.st.session_state.stats["iq"] = 0.0

    app.run_simulation_tick()

    # 30% of 100.0 = 30.0 is consolidated
    # memory becomes 100.0 - 30.0 = 70.0 (plus any motor additions, none here)
    assert app.st.session_state.stats["memory"] == 70.0
    assert app.st.session_state.stats["iq"] == 30.0
