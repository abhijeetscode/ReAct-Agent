import re
import json

def load_json_string(json_string: str)->dict:
    cleaned = re.sub(r'```json\s*', '', json_string)
    cleaned = re.sub(r'```\s*$', '', cleaned)
    
    # Remove extra whitespace and newlines at start/end
    cleaned = cleaned.strip()
    
    # Remove any trailing commas before closing braces/brackets
    cleaned = re.sub(r',(\s*[}\]])', r'\1', cleaned)
    
    try:
        # Parse the cleaned JSON
        parsed_json = json.loads(cleaned)
        return parsed_json
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON after cleaning: {e}")