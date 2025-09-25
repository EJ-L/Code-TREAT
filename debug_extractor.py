#!/usr/bin/env python3
"""
Debug the Code Reasoning Extractor to see why parsed_response field is missing
"""

import sys
import os

# Add the project root to Python path
project_root = "/Users/ericjohnli/Downloads/Code-TREAT"
sys.path.insert(0, project_root)

from extractors.llm_extraction_utils.code_reasoning_extractors import CodeReasoningExtractor
from models.model_list import MODELS
import json


def debug_extractor():
    """Debug a single line extraction"""
    
    # Get GLM-4-Flash model for parsing
    glm4_flash_model = MODELS.get('glm-4-flash')
    if glm4_flash_model is None:
        print("❌ GLM-4-Flash model not found in MODELS!")
        return
    
    print(f"✅ Using GLM-4-Flash model for parsing: {glm4_flash_model}")
    
    # Initialize extractor
    extractor = CodeReasoningExtractor(
        parsing_model=glm4_flash_model,
        max_responses_to_parse=1,  # Just 1 response for debugging
        max_workers=1,
        max_retries=1
    )
    
    # Test with a single line
    input_file = f"{project_root}/results/input_prediction/hackerrank/predictions/GLM-4-Flash.jsonl"
    
    # Read first line
    with open(input_file, 'r') as f:
        first_line = f.readline().strip()
        json_line = json.loads(first_line)
    
    print(f"📄 Original line keys: {list(json_line.keys())}")
    print(f"📤 Original response: {json_line.get('response', [])}")
    
    # Process single line manually
    try:
        ref_key = extractor._generate_ref_key(json_line)
        print(f"🔑 Generated ref_key: {ref_key}")
        
        progress_data = {}
        processed_line = extractor._process_single_line(json_line, ref_key, progress_data)
        
        print(f"✅ Processed line keys: {list(processed_line.keys())}")
        print(f"📥 Parsed response: {processed_line.get('parsed_response', 'MISSING!')}")
        print(f"📊 Progress data: {progress_data}")
        
    except Exception as e:
        print(f"❌ Error processing line: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    debug_extractor()