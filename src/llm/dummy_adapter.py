import random
from typing import Dict, Optional
from .interface import LLMInterface

class DummyAdapter(LLMInterface):
    """
    A dummy adapter for testing purposes.
    Returns random Pokemon names for generation and random scores for evaluation.
    """
    def __init__(self):
        self.pokemon_names = [
            "Pikachu", "Charizard", "Bulbasaur", "Squirtle", "Eevee", 
            "Snorlax", "Gengar", "Jigglypuff", "Meowth", "Psyduck",
            "Magikarp", "Gyarados", "Lapras", "Ditto", "Mewtwo"
        ]
        self.token_usage = {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15
        }

    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        # Check if it looks like an evaluation prompt (asking for a score)
        if "Provide a score from 0 to 10" in prompt:
            return str(random.randint(0, 10))
        
        # Check if it looks like a mutation prompt
        if "Rewrite it to be even more effective" in prompt:
             # Just return another random pokemon with a prefix to indicate mutation
             return f"Mutated {random.choice(self.pokemon_names)}"

        # Default generation (Task Definition prompt)
        # Check if it asks for multiple prompts (initialization)
        if "Generate" in prompt and "prompts" in prompt:
            # Need to return K lines
            # Rough heuristic to find K
            import re
            match = re.search(r'Generate (\d+) distinct prompts', prompt)
            k = int(match.group(1)) if match else 3
            return "\n".join([f"Generate text for {random.choice(self.pokemon_names)}" for _ in range(k)])

        # Single generation
        return f"This is a {random.choice(self.pokemon_names)}."

    def get_token_usage(self) -> Dict[str, int]:
        return self.token_usage
