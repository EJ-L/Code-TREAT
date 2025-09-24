# extractors/code_generation_extractor.py
from models.model_list import MODELS
from extractors.llm_extraction_utils.base_llm_extraction import BaseLLMExtractor
import json
import os
from typing import Any, Dict, List
from collections import defaultdict

class CodeGenerationExtractor(BaseLLMExtractor):
    """
    Code extractor for code generation tasks.

    Key behaviors:
      - max_attempts (default: 3): per-response_index retry cap when verification fails.
      - Verified-aware resume: when resuming from state, skip only responses that are already verified=True.
      - If a response_index exists in state but verified=False and attempts < max_attempts, it will be retried.
      - New responses appended to the input JSONL (e.g., response indices 1, 2 after 0) will be parsed fresh.
    """

    def __init__(self, model_name: str, max_workers: int = 8, save_every: int = 10, max_attempts: int = 3):
        super().__init__(model_name, max_workers, save_every, max_attempts)
        self.max_attempts = max_attempts

    def make_unique_key(self, original_entry: Dict[str, Any]) -> str:
        """
        Build a unique key for the entry:
        id_targetLang_promptCat_promptId

        Note:
        - Supports either 'prompt_category' or 'prompting_category'
        - prompt_category can be list or str; list joined by ','
        """
        prompt_cat = original_entry.get('prompt_category', original_entry.get('prompting_category', []))
        prompt_id = original_entry.get('prompt_id', 0)

        if isinstance(prompt_cat, list):
            prompt_cat_str = ','.join(prompt_cat)
        else:
            prompt_cat_str = str(prompt_cat)

        return (
            f"{original_entry['id']}_"
            f"{original_entry.get('lang', 'unknown')}_"
            f"{prompt_cat_str}_"
            f"{prompt_id}"
        )

    def select_best_code_block(self, code_blocks: List[str], target_language: str) -> str:
        """
        Use LLM to select the most complete code block when multiple blocks are found.
        """
        if len(code_blocks) == 1:
            return code_blocks[0]
        
        blocks_text = ""
        for i, block in enumerate(code_blocks, 1):
            blocks_text += f"OPTION {i}:\n```{target_language}\n{block}\n```\n\n"
        
        prompt = f"""You are given multiple {target_language} code blocks. Select the MOST COMPLETE one.

Choose the option that:
- Contains the most comprehensive implementation
- Has the most complete functionality
- Includes necessary imports/dependencies
- Is most likely to be a working solution

{blocks_text}

Respond with only the number (1, 2, 3, etc.) of the best option."""

        messages = [{"role": "user", "content": prompt}]
        try:
            model_responses = self.MODEL.chat(messages)
            if not model_responses:
                return code_blocks[0]  # fallback to first
                
            response = model_responses[0]
            if isinstance(response, dict) and "content" in response:
                response = response["content"]
            
            # Extract number from response
            import re
            match = re.search(r'\b(\d+)\b', str(response))
            if match:
                choice = int(match.group(1)) - 1  # Convert to 0-based index
                if 0 <= choice < len(code_blocks):
                    return code_blocks[choice]
            
            return code_blocks[0]  # fallback to first
        except Exception as e:
            print(f"Block selection failed: {e}")
            return code_blocks[0]  # fallback to first

    def fallback_extract(self, response_text: str, target_language: str) -> str:
        """
        LLM-based extraction used when no fenced code is detected or verified.
        Returns extracted code as string.
        """
        prompt = f"""Extract the complete {target_language} code from the following text. Look for:
- Code blocks (with or without markdown fencing)
- Complete {target_language} functions, classes, or scripts
- Code that may be embedded within explanations

Rules:
1. If you find code, respond with:
EXTRACTED_CODE:
```{target_language.lower()}
<the complete executable code>
```

2. Extract the most complete and executable version
3. Include all necessary imports and dependencies
4. If absolutely no programming code exists, respond with: NO_CODE_FOUND

Text to analyze:
{response_text}"""

        messages = [{"role": "user", "content": prompt}]
        try:
            model_responses = self.MODEL.chat(messages)
            if not model_responses:
                return ""

            response = model_responses[0]
            if isinstance(response, dict) and "content" in response:
                response = response["content"]

            if not isinstance(response, str):
                response = str(response)

            if "NO_CODE_FOUND" in response.upper():
                return ""

            # Try fenced extraction from the model's reply
            from extractors.regex_extraction_utils.markdown_utils import extract_fenced_code
            has_code, code_blocks = extract_fenced_code(response)
            if has_code and code_blocks:
                return self.select_best_code_block(code_blocks, target_language)
            
            return ""
        except Exception as e:
            print(f"LLM code extraction failed: {e}")
            return ""

    # ---- Internal helpers -------------------------------------------------

    def _summarize_existing(self, existing_items: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
        """
        Summarize prior state by response_index:
        returns: idx -> { 'verified': bool, 'attempts': int }

        - attempts falls back to 1 if item exists but 'attempts' missing (older states).
        - verified=True if ANY existing item for that index was verified.
        """
        summary: Dict[int, Dict[str, Any]] = defaultdict(lambda: {"verified": False, "attempts": 0})
        for it in existing_items or []:
            idx = it.get("response_index")
            if idx is None:
                continue
            attempts = it.get("attempts")
            if attempts is None:
                attempts = 1  # conservative default for legacy records
            prev = summary[idx]
            prev["attempts"] = max(prev["attempts"], attempts)
            prev["verified"] = prev["verified"] or bool(it.get("verified", False))
        return summary

    # ---- Per-entry processing (called by BaseLLMExtractor.parse_file) -----

    def _process_entry(self, raw_line: str, state: Dict[str, Any], responses_per_entry: int):
        """Process a single JSONL line (one task/entry)."""
        raw = (raw_line or "").strip()
        if not raw:
            return None

        try:
            data = json.loads(raw)
        except Exception:
            return None

        # Build key & basic fields
        try:
            key = self.make_unique_key(data)
            target_language = data.get('lang', 'unknown')
        except Exception:
            return None

        # Existing state for this key
        existing: List[Dict[str, Any]] = state.get("by_key", {}).get(key, [])
        existing_summary = self._summarize_existing(existing)

        # Skip fully-verified entries
        verified_count = sum(1 for v in existing_summary.values() if v.get("verified", False))
        if verified_count >= responses_per_entry:
            return None

        # Get outputs list
        outputs = data.get("output", data.get("response", []))
        if not isinstance(outputs, list):
            return None

        upper = min(responses_per_entry, len(outputs))
        extracted: List[Dict[str, Any]] = []

        from extractors.regex_extraction_utils.markdown_utils import extract_fenced_code

        for i in range(0, upper):
            if i >= len(outputs):
                break

            response = outputs[i]
            if not isinstance(response, str):
                continue

            prior = existing_summary.get(i, {"verified": False, "attempts": 0})
            prior_verified = bool(prior.get("verified", False))
            prior_attempts = int(prior.get("attempts", 0))

            # Already verified -> skip
            if prior_verified:
                continue
            # Attempts exhausted -> skip
            if prior_attempts >= self.max_attempts:
                continue

            attempt_num = prior_attempts + 1

            # Single entry per response with verification-driven approach
            code_text = ""
            method = ""
            extracted_successfully = False
            verified = False
            error = None
            
            try:
                # 1) Try direct fenced code extraction
                has_fenced, code_blocks = extract_fenced_code(response)
                if has_fenced and code_blocks:
                    # Select best code block if multiple
                    code_text = self.select_best_code_block(code_blocks, target_language)
                    method = "fenced_code"
                    extracted_successfully = True
                    # Verify fenced code
                    verified = self.verify(code_text, response)
                
                # 2) If no fenced code or fenced code failed verification, try LLM extraction
                if not verified:
                    fallback_code = self.fallback_extract(response, target_language)
                    if fallback_code:
                        code_text = fallback_code
                        method = "llm_fallback"
                        extracted_successfully = True
                        # Verify LLM extracted code
                        verified = self.verify(code_text, response)
                    else:
                        code_text = ""
                        method = "extraction_failed"
                        extracted_successfully = False
                        error = "No code found in response"
                        verified = False

            except Exception as e:
                code_text = ""
                method = "extraction_failed"
                extracted_successfully = False
                error = str(e)
                verified = False

            # Create single entry for this response
            entry = {
                "response_index": i,
                "code": code_text,
                "language": target_language,
                "method": method,
                "extracted_successfully": extracted_successfully,
                "verified": verified,
                "attempts": attempt_num,
                "regenerated": prior_attempts > 0
            }
            
            if error:
                entry["error"] = error
                
            extracted.append(entry)

        if extracted:
            return {"key": key, "extracted": extracted}
        return None

    # ---- Public API -------------------------------------------------------

    def extract(self, filename: str, max_responses: int = 1) -> str:
        """
        Extract codes from a JSONL file. Resumes automatically if a state file exists.

        Args:
            filename: path to the .jsonl file containing entries with "output"/"response" arrays.
            max_responses: maximum responses to process per entry (by index order).

        Returns:
            The path to the parsed JSONL file produced by BaseLLMExtractor.parse_file.
        """
        print(f"🚀 Starting code extraction for: {os.path.basename(filename)}")

        parsed_dir, state_path = self._paths_for(filename)
        resume = os.path.exists(state_path)

        if resume:
            print(f"📂 Found existing state file - resuming from: {state_path}")
        else:
            print("📄 No state file found - starting fresh extraction")

        print(f"📊 Settings: max_responses={max_responses}, resume={resume}, max_attempts={self.max_attempts}")

        final_state_path = self.parse_file(filename, max_responses, resume)

        parsed_file = os.path.join(parsed_dir, os.path.basename(filename))

        print("✅ Extraction complete!")
        print(f"📁 State file: {final_state_path}")
        print(f"📄 Parsed file: {parsed_file}")

        return parsed_file
