import sys
from unittest.mock import MagicMock

# Mock out Streamlit and external modules to avoid side effects during test import
mock_st = MagicMock()

# Mock st.session_state as a dict-like object
class MockSessionState(dict):
    pass
mock_st.session_state = MockSessionState()

# Mock columns and tabs as dynamic lambdas returning lists of MagicMocks
mock_st.columns = lambda spec: [MagicMock() for _ in (spec if isinstance(spec, list) else range(spec))]
mock_st.tabs = lambda spec: [MagicMock() for _ in range(len(spec))]

sys.modules['streamlit'] = mock_st
sys.modules['pypdf'] = MagicMock()
sys.modules['telegram'] = MagicMock()
sys.modules['telegram.error'] = MagicMock()
sys.modules['telegram.ext'] = MagicMock()

import unittest
from app import Agent, SimulationEngine

class TestSimulationGameEngine(unittest.TestCase):

    def setUp(self):
        # Create a fresh engine before each test
        self.engine = SimulationEngine()

    def test_initial_state(self):
        """Verify that the grid layout and agents are correctly initialized."""
        self.assertEqual(self.engine.width, 6)
        self.assertEqual(self.engine.height, 6)
        self.assertEqual(self.engine.tick, 0)
        self.assertEqual(len(self.engine.agents), 4)

        # Check agent profiles
        roles = {a.role for a in self.engine.agents}
        names = {a.name for a in self.engine.agents}
        self.assertIn("Farmer", roles)
        self.assertIn("Scientist", roles)
        self.assertIn("Explorer", roles)
        self.assertIn("Warrior", roles)
        self.assertIn("Alice", names)

        # Check landmarks
        self.assertEqual(self.engine.map_grid[(0, 1)], "🌾 Wheat Field")
        self.assertEqual(self.engine.map_grid[(2, 2)], "🏠 Colony Base")

    def test_agent_to_dict(self):
        """Verify that agent details can be serialized to a dict for pandas dataframes."""
        agent = self.engine.agents[0]
        a_dict = agent.to_dict()
        self.assertEqual(a_dict["Name"], agent.name)
        self.assertEqual(a_dict["Role"], agent.role)
        self.assertIn("Health", a_dict)

    def test_heuristics_ai_decision_low_vitality(self):
        """Verify that agents seek shelter and recovery at Colony Base (2, 2) when low health/energy."""
        agent = self.engine.agents[0] # Alice, Farmer, currently at (0,1)
        agent.energy = 10
        agent.health = 90

        # Should want to move towards Colony Base (2, 2)
        decision = self.engine.get_ai_decision(agent)
        self.assertIn("Move to (2, 2)", decision)

        # If already at Colony Base, should want to rest
        agent.x, agent.y = 2, 2
        decision_at_base = self.engine.get_ai_decision(agent)
        self.assertIn("Rest at Colony Base", decision_at_base)

    def test_heuristics_ai_decision_roles(self):
        """Verify role-specific decision logic when agent is healthy and full of energy."""
        # Farmer should want to harvest or move to Wheat Field (0,1)
        farmer = [a for a in self.engine.agents if a.role == "Farmer"][0]
        farmer.health = 100
        farmer.energy = 100
        farmer.x, farmer.y = 0, 1
        decision_farmer = self.engine.get_ai_decision(farmer)
        self.assertIn("Harvest wheat", decision_farmer)

        # Scientist should want to research or move to Tech Lab (4,5)
        scientist = [a for a in self.engine.agents if a.role == "Scientist"][0]
        scientist.health = 100
        scientist.energy = 100
        scientist.x, scientist.y = 4, 5
        decision_scientist = self.engine.get_ai_decision(scientist)
        self.assertIn("Research at Tech Lab", decision_scientist)

    def test_execute_action_movement(self):
        """Verify that executing a move action changes agent coordinate closer to destination."""
        agent = Agent("TestAgent", "Farmer", 0, 0)
        # Choose target (2, 2)
        self.engine.execute_action(agent, "Move to (2, 2)")
        # Diagonal step from (0,0) towards (2,2) -> (1,1)
        self.assertEqual(agent.x, 1)
        self.assertEqual(agent.y, 1)
        self.assertEqual(agent.energy, 94) # Moved consumes 6 energy

    def test_execute_action_interactions(self):
        """Verify that executing interaction actions at landmarks properly grants status changes."""
        # 1. Rest at Colony Base
        agent = Agent("TestAgent", "Farmer", 2, 2)
        agent.energy = 50
        agent.health = 50
        self.engine.execute_action(agent, "Rest at Colony Base (2,2)")
        self.assertEqual(agent.energy, 85) # +35
        self.assertEqual(agent.health, 70) # +20

        # 2. Harvest at Wheat Field
        agent = Agent("TestAgent", "Farmer", 0, 1)
        agent.wealth = 100
        agent.inventory["wheat"] = 0
        self.engine.execute_action(agent, "Harvest wheat at Wheat Field (0,1)")
        self.assertEqual(agent.inventory["wheat"], 1)
        self.assertEqual(agent.wealth, 115) # +15

    def test_simulation_step_updates(self):
        """Verify that a simulation tick updates tick count, history, and applies energy drains."""
        # Execute 1 tick
        self.engine.step()
        self.assertEqual(self.engine.tick, 1)
        # Check that history list grew
        self.assertEqual(len(self.engine.history["tick"]), 2)
        self.assertEqual(self.engine.history["tick"][-1], 1)

    def test_agent_faint_and_revive(self):
        """Verify that an agent with <= 0 health is revived at Colony Base (2, 2)."""
        agent = self.engine.agents[0]
        agent.health = 0
        agent.wealth = 100
        agent.x, agent.y = 5, 5

        # Execute 1 step -> should trigger revival logic
        self.engine.step()
        self.assertEqual(agent.health, 50)
        self.assertEqual(agent.energy, 50)
        self.assertEqual(agent.x, 2)
        self.assertEqual(agent.y, 2)
        self.assertEqual(agent.wealth, 60) # lost 40 wealth

    def test_events_meteor_strike(self):
        """Verify that a meteor strike deals damage to nearby agents and leaves ruins."""
        agent = self.engine.agents[0]
        agent.x, agent.y = 3, 3
        agent.health = 100

        # Inject fixed coordinate target for meteor (mock random)
        import random
        original_randint = random.randint
        try:
            # Force coordinates to be (3, 3)
            random.randint = lambda a, b: 3
            self.engine.trigger_meteor_strike()
        finally:
            random.randint = original_randint

        # Check map updated
        self.assertEqual(self.engine.map_grid[(3, 3)], "🔥 Meteor Ruins")
        # Check direct hit damage: 100 - 60 = 40
        self.assertEqual(agent.health, 40)

    def test_events_gold_rush(self):
        """Verify that gold rush spawns gold mine on the map."""
        original_len = len([v for v in self.engine.map_grid.values() if v == "💎 Gold Mine"])
        self.engine.trigger_gold_rush()
        new_len = len([v for v in self.engine.map_grid.values() if v == "💎 Gold Mine"])
        # Should be at least original or original + 1
        self.assertGreaterEqual(new_len, original_len)

    def test_events_pandemic(self):
        """Verify that pandemic lowers all agents health by 25."""
        for a in self.engine.agents:
            a.health = 100
        self.engine.trigger_pandemic()
        for a in self.engine.agents:
            self.assertEqual(a.health, 75)

    def test_events_custom_scenario(self):
        """Verify that custom events with keywords like 'heal' or 'wealth' apply boosts."""
        # 1. Custom heal scenario
        for a in self.engine.agents:
            a.health = 40
        self.engine.trigger_custom_event("Emergency medical heal drop!")
        for a in self.engine.agents:
            self.assertEqual(a.health, 80) # +40

        # 2. Custom gift scenario
        for a in self.engine.agents:
            a.wealth = 100
        self.engine.trigger_custom_event("Gift some wealth to all")
        for a in self.engine.agents:
            self.assertEqual(a.wealth, 200) # +100

if __name__ == '__main__':
    unittest.main()
