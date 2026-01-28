from typing import Tuple, Optional

def parse_stance_reason(text: str) -> Tuple[Optional[str], str]:
    """
    Parses "STANCE: Reason" string.
    Returns upper-case STANCE and trimmed Reason.
    """
    # Supported stances: SUPPORT, OPPOSE
    text = text.strip()
    parts = text.split(":", 1)
    if len(parts) < 2:
        return None, text # Cannot parse
        
    potential_stance = parts[0].strip().upper()
    content = parts[1].strip()
    
    if potential_stance in ["SUPPORT", "OPPOSE"]:
        return potential_stance, content
        
    return None, text
