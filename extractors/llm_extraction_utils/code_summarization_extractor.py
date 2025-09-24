# extractors/llm_extraction_utils/code_summarization_extractor.py
from extractors.llm_extraction_utils.base_llm_extraction import BaseLLMExtractor
from evaluators.utils.utils import parse_response_outputs, remove_thinking_tag
import json
import os
from typing import Any, Dict, List


class CodeSummarizationExtractor(BaseLLMExtractor):
    """
    Code summarization extractor for code summarization tasks.
    
    Extracts code comments/summaries from raw model responses and creates
    parsed_response field with cleaned extracted comments.
    
    Key behaviors:
      - Removes thinking tags from responses
      - Parses response outputs using language-specific patterns
      - Extracts comments/summaries from various formats
      - Creates parsed_response field for evaluation
    """

    def __init__(self, model_name: str = None, max_workers: int = 8, save_every: int = 50, max_attempts: int = 3):
        super().__init__(model_name, max_workers, save_every, max_attempts)

    def make_unique_key(self, original_entry: Dict[str, Any]) -> str:
        """
        Build a unique key for the entry:
        dataset_modelName_codeSnippet_promptCat_promptId
        """
        dataset = original_entry.get('dataset', 'general')
        model_name = original_entry.get('model_name', 'unknown')
        
        # Use code snippet or function name for uniqueness
        code_snippet = original_entry.get('code/function', original_entry.get('url', 'unknown'))
        if isinstance(code_snippet, str):
            code_snippet = code_snippet[:50].replace('/', '_').replace(' ', '_').replace('\n', '_')
        else:
            code_snippet = 'unknown'
        
        prompt_cat = original_entry.get('prompting_category', original_entry.get('prompt_category', 'direct'))
        if isinstance(prompt_cat, list):
            prompt_cat = prompt_cat[0]
        
        prompt_id = original_entry.get('prompt_id', 1)
        
        return f"{dataset}_{model_name}_{code_snippet}_{prompt_cat}_{prompt_id}"

    def extract_from_response(self, response_text: str, lang: str = "python") -> List[str]:
        """Extract code summarization comments from response text."""
        if not response_text or response_text == "Exceeds Context Length":
            return [response_text] if response_text else [""]
        
        # Remove thinking tags
        cleaned_response = remove_thinking_tag(response_text)
        
        # Parse response outputs using existing utility
        parsed_outputs = parse_response_outputs([cleaned_response], lang)
        
        return parsed_outputs

    def fallback_extract(self, response_text: str, lang: str = "python") -> List[str]:
        """Fallback extraction for when primary extraction fails."""
        if not response_text:
            return [""]
        
        # Just return cleaned response as fallback
        cleaned_response = remove_thinking_tag(response_text)
        return [cleaned_response]

    def extract(self, input_file: str) -> str:
        """
        Extract code summarization responses from predictions file.
        
        Args:
            input_file: Path to predictions JSONL file
            
        Returns:
            Path to the created parsed JSONL file
        """
        print(f"Extracting code summarization responses from: {input_file}")
        
        # Determine output path (predictions -> parsed)
        if "/predictions/" in input_file:
            output_file = input_file.replace("/predictions/", "/parsed/")
        else:
            output_file = input_file.replace('.jsonl', '_parsed.jsonl')
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Load and process data
        processed_entries = []
        
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                    
                try:
                    entry = json.loads(line)
                    
                    # Extract responses
                    responses = entry.get('response', entry.get('outputs', []))
                    lang = entry.get('lang', 'python')
                    parsed_responses = []
                    
                    for response in responses:
                        extracted = self.extract_from_response(response, lang)
                        parsed_responses.extend(extracted)
                    
                    # Add parsed_response field
                    entry['parsed_response'] = parsed_responses
                    processed_entries.append(entry)
                    
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse JSON line: {e}")
                    continue
        
        # Write processed entries
        with open(output_file, 'w', encoding='utf-8') as f:
            for entry in processed_entries:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        
        print(f"Extraction complete: {len(processed_entries)} entries processed")
        print(f"Parsed file created: {output_file}")
        
        return output_file

    def extract_batch(self, input_files: List[str]) -> List[str]:
        """Extract multiple files in batch."""
        output_files = []
        for input_file in input_files:
            try:
                output_file = self.extract(input_file)
                output_files.append(output_file)
            except Exception as e:
                print(f"Error processing {input_file}: {e}")
                continue
        return output_files


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python code_summarization_extractor.py <predictions_file>")
        print("Example: python code_summarization_extractor.py results/code_summarization/github/predictions/GPT-5.jsonl")
        sys.exit(1)
    
    input_file = sys.argv[1]
    extractor = CodeSummarizationExtractor()
    output_file = extractor.extract(input_file)
    print(f"Extraction complete: {output_file}")