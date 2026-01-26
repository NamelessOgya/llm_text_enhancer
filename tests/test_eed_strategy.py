import unittest
import os
import shutil
import json
from unittest.mock import MagicMock, patch
from llm.interface import LLMInterface
from evolution_strategies import EEDStrategy, TextGradV2Strategy

class MockLLM(LLMInterface):
    def __init__(self, responses: list = None):
        self.responses = responses or []
        self.call_count = 0
        self.calls = []

    def generate(self, prompt: str) -> str:
        self.calls.append(prompt)
        response = "Mock Response"
        if self.call_count < len(self.responses):
            response = self.responses[self.call_count]
        self.call_count += 1
        return response

    def get_token_usage(self):
        return {}

class TestEEDStrategy(unittest.TestCase):
    def setUp(self):
        self.test_dir = "tests/test_results_eed"
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        os.makedirs(self.test_dir, exist_ok=True)
        
        self.task_def = "Test Task Definition"
        self.k = 10 
        # Config: 3 Exploit, 2 Diversity, 5 Explore
        # Elitism 1
        self.context = {
            "result_dir": self.test_dir,
            "iteration": 1,
            "task_name": "test_task"
        }
        
        self.population = []
        for i in range(10):
            self.population.append({
                "text": f"Parent Text {i} score {1.0 - i*0.1}",
                "score": 1.0 - i*0.1,
                "logic": "init"
            })
            
    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_evolve_flow(self):
        strategy = EEDStrategy()
        
        # We need to mock a SEQUENCE of LLM responses corresponding to the flow:
        # 1. Exploit (3 samples) -> 3 calls to (Critique + Rewrite) = 6 calls?
        #    Wait, loop runs 3 times? 
        #    Exploit loop: for parent in top 3. for i in range(2).
        #    We need 3 exploit samples total.
        #    Each sample = 2 calls (Critique, Update).
        #    So ~4-6 calls for Exploit.
        
        # 2. Diversity (2 samples) -> Novelty Selection (1 call) + Refinement (2 calls per sample)
        #    Refinement = Critique + Update = 2 calls per candidate.
        #    Total ~ 1 + 2*2 = 5 calls.
        
        # 3. Explore (5 samples) -> Gap Analysis (1 call for Norms, 1 call for Missing, 1 call for Selection?)
        #    Norms = 1 call.
        #    Missing Domains = 1 call.
        #    Generation = Explore(5) * PassRate(2) = 10 calls.
        #    Selection is internal logic? Yes, unless it uses LLM.
        #    Wait, `_calculate_novelty` uses embeddings, not LLM.
        
        # Total calls ~ 20+ calls.
        
        # Instead of exact matching, let's provide a generic generator or large list.
        responses = []
        
        # Exploit Responses
        for _ in range(10):
            responses.append("Exploit Gradient")
            responses.append("Exploit Text")
            
        # Diversity Responses
        # Novelty Selection
        # Expecting indices "0, 1"
        responses.append("1, 2") 
        # Diversity Refinement
        for _ in range(5):
            responses.append("Diversity Gradient")
            responses.append("Diversity Text")
            
        # Explore Responses
        # Norms
        responses.append("Common Stance: Support\nCore Arguments:\n- Arg 1")
        # Missing Domains
        # Explore Responses
        # Norms
        responses.append("Common Stance: Support\nCore Arguments:\n- Arg 1")
        # Persona Identification
        responses.append('["Economist", "Lawyer", "Doctor"]')
        # Generation
        for i in range(20):
            responses.append(f"Explore Text {i}")
            
        llm = MockLLM(responses)
        
        # Set config manually to match test assumptions if needed, 
        # but EEDStrategy loads from yaml. 
        # We might need to patch self.config if we want stability.
        # But let's assume yaml is default (3, 2, 5).
        
        new_texts = strategy.evolve(llm, self.population, self.k, self.task_def, self.context)
        
        self.assertEqual(len(new_texts), self.k)
        
        # Verify call content implies we ran phases
        
        # Check for Diversity Phase specifics
        diversity_refinement_call = False
        novelty_selection_call = False
        explore_persona_call = False
        
        for call in llm.calls:
            if "Candidate Perspectives:" in call and "Identify the top" in call:
                novelty_selection_call = True
            if "Current Text (Selected for Novelty):" in call:
                diversity_refinement_call = True
            if "Identify 5 distinct STAKEHOLDERS" in call:
                explore_persona_call = True
            if "Role: You are a Economist" in call: 
                # We expect at least one call to have picked Economist from the list
                pass

        self.assertTrue(novelty_selection_call, "Novelty Selection prompt not found")
        self.assertTrue(diversity_refinement_call, "Novelty Refinement prompt not found")
        self.assertTrue(explore_persona_call, "Persona Identification prompt not found")

if __name__ == "__main__":
    unittest.main()
