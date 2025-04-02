import re
import json

from modules import config

class Middleware:
    """Generic middleware for parsing LLM IO data"""
    
    def handle_incoming(self, data):
        """Parse incoming data"""
        return data
    
    def handle_outgoing(self, data):
        """Parse outgoing data"""
        return data
   
class CommentRemover(Middleware):
    """Middleware for removing comments from output"""
    
    def handle_outgoing(self, data):
        """Remove comments from outgoing data"""
        if 'choices' in data and len(data['choices']) > 0 and 'text' in data['choices'][0]:
            text = data['choices'][0]['text']
            
            pattern = r'(\/\/.*?$|\/\*.*?\*\/)'
            text = re.sub(pattern, '', text, flags=re.MULTILINE|re.DOTALL)
            
            data['choices'][0]['text'] = text
    
        return data
    
class ValidJSONExtractor(Middleware):
    """Middleware for extracting valid JSON data"""
    
    def handle_outgoing(self, data):
        """Extract the first valid JSON object"""
        if 'choices' in data and len(data['choices']) > 0 and 'text' in data['choices'][0]:
            text = data['choices'][0]['text']
            
            # Look for patterns between curly braces, including nested structures
            pattern = r'(\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*\}))*\})'
            json_candidates = re.findall(pattern, text)
            
            # Try each
            for candidate in json_candidates:
                try:
                    json.loads(candidate)
                    data['choices'][0]['text'] = candidate
                    break
                except json.JSONDecodeError:
                    continue
        
        return data


ALL_MIDDLEWARE = {
    'comment_remover': CommentRemover(),
    'valid_json_extractor': ValidJSONExtractor()
}