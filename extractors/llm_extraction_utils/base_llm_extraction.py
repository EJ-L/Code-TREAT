# treat/extractors/llm_extractor.py
from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

from tqdm import tqdm

from models.model_list import MODELS
from extractors.regex_extraction_utils.markdown_utils import extract_fenced_code

RESULT_DIR_NAMES = {"predictions", "evaluations", "metrics", "parsed"}

class BaseLLMExtractor(ABC):
    def __init__(
        self,
        model_name: Optional[str] = None,
        max_workers: int = 8,
        save_every: int = 50,
        max_attempts: int = 3,
    ):
        if model_name and model_name not in MODELS:
            raise ValueError(f"Model {model_name} not found in MODELS")
        self.model_name = model_name
        self.MODEL = MODELS.get(model_name) if model_name else None
        self.max_workers = max_workers
        self.save_every = max(1, save_every)
        self.max_attempts = max(1, int(max_attempts))

    @abstractmethod
    def make_unique_key(self, original_entry: Dict[str, Any]) -> str:
        """Return a stable unique key for the entry."""

    def extract_from_response(self, response_text: str) -> List[Dict[str, Any]]:
        """Extract responses. Override for task-specific logic."""
        extracted, code_blocks = extract_fenced_code(response_text)
        if extracted:
            # Convert strings to expected dictionary format
            if isinstance(code_blocks, list):
                return [{"code": block, "language": "python", "method": "fenced"} for block in code_blocks]
            else:
                return [{"code": code_blocks, "language": "python", "method": "fenced"}]
        return []

    def fallback_extract(self, response_text: str) -> List[Dict[str, Any]]:
        """Optional LLM fallback extraction. Override to implement."""
        return []

    def verify(self, code_snippet: str, source_text: str) -> bool:
        """Verify extracted code. Override for stronger checks."""
        return code_snippet in source_text

    def _paths_for(self, filename: str) -> tuple[str, str]:
        """Return (parsed_dir, state_path) using same logic as evaluation_dir."""
        # Use same pattern as evaluator: file_dir -> parent_dir -> parsed_dir
        file_dir = os.path.dirname(filename)
        parent_dir = os.path.dirname(file_dir)
        parsed_dir = os.path.join(parent_dir, 'parsed')

        # Remove extension and add .json for state file
        base_name_no_ext = os.path.splitext(os.path.basename(filename))[0]
        meta_name = f"{base_name_no_ext}.json"

        return parsed_dir, os.path.join(parsed_dir, meta_name)

    def _load_state(self, state_path: str, filename: str, responses_per_entry: int) -> Dict[str, Any]:
        """Load existing state or create new one."""
        if os.path.exists(state_path):
            try:
                with open(state_path, "r", encoding="utf-8") as f:
                    state = json.load(f)
                meta = state.get("meta", {})
                if (meta.get("responses_per_entry") == responses_per_entry and 
                    meta.get("source_file") == os.path.abspath(filename)):
                    return state
            except Exception:
                pass

        # Create fresh state
        return {
            "meta": {
                "source_file": os.path.abspath(filename),
                "responses_per_entry": responses_per_entry,
                "updated_at": datetime.now().isoformat(timespec="seconds"),
                "task_name": self.__class__.__name__,
                "model_name": self.model_name,
            },
            "by_key": {}
        }

    def _save_state(self, state: Dict[str, Any], state_path: str) -> None:
        """Save state to file."""
        state["meta"]["updated_at"] = datetime.now().isoformat(timespec="seconds")
        tmp = f"{state_path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        os.replace(tmp, state_path)

    # helper: summarize prior attempts & verification per response index
    def _summarize_existing(self, existing_items: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
        """
        Returns a map: idx -> { 'verified': bool, 'attempts': int }
        - attempts defaults to 1 for legacy items missing the field.
        - verified is True if any existing item at that index has verified=True.
        """
        summary: Dict[int, Dict[str, Any]] = defaultdict(lambda: {"verified": False, "attempts": 0})
        for it in existing_items or []:
            idx = it.get("response_index")
            if idx is None:
                continue
            att = it.get("attempts")
            if att is None:
                att = 1  # legacy record without attempts
            prev = summary[idx]
            prev["attempts"] = max(prev["attempts"], int(att))
            prev["verified"] = prev["verified"] or bool(it.get("verified", False))
        return summary

    def _process_entry(self, raw_line: str, state: Dict[str, Any], responses_per_entry: int) -> Optional[Dict[str, Any]]:
        """Process a single entry."""
        raw = (raw_line or "").strip()
        if not raw:
            return None

        try:
            data = json.loads(raw)
        except Exception:
            return None

        # Key
        try:
            key = self.make_unique_key(data)
        except Exception:
            return None

        # Existing items & summary
        existing = state.get("by_key", {}).get(key, [])
        existing_summary = self._summarize_existing(existing)

        # If we already have enough VERIFIED responses, skip
        verified_count = sum(1 for v in existing_summary.values() if v.get("verified", False))
        if verified_count >= responses_per_entry:
            return None

        # Read outputs from either 'output' or 'response' (be permissive) ★
        outputs = data.get("output", data.get("response"))
        if not isinstance(outputs, list):
            return None

        # Consider indices up to responses_per_entry (or available outputs), allowing retries of old unverified ones ★
        upper = min(responses_per_entry, len(outputs))
        extracted: List[Dict[str, Any]] = []

        for i in range(0, upper):
            if i >= len(outputs):
                break

            response = outputs[i]
            if not isinstance(response, str):
                continue

            prior = existing_summary.get(i, {"verified": False, "attempts": 0})
            prior_verified = bool(prior.get("verified", False))
            prior_attempts = int(prior.get("attempts", 0))

            # Skip if verified already ★
            if prior_verified:
                continue
            # Skip if attempts exhausted ★
            if prior_attempts >= self.max_attempts:
                continue

            attempt_num = prior_attempts + 1

            # Primary extraction
            codes = self.extract_from_response(response)
            if codes:
                for c in codes:
                    code_text = c.get("code", "")
                    extracted.append({
                        "response_index": i,
                        **c,
                        "verified": self.verify(code_text, response),
                        "attempts": attempt_num,
                        "regenerated": prior_attempts > 0,
                        "extracted_successfully": True
                    })
            else:
                # Fallback extraction
                try:
                    fallback_codes = self.fallback_extract(response)
                    if fallback_codes:
                        for c in fallback_codes:
                            code_text = c.get("code", "")
                            extracted.append({
                                "response_index": i,
                                **c,
                                "verified": self.verify(code_text, response),
                                "attempts": attempt_num,
                                "regenerated": prior_attempts > 0,
                                "extracted_successfully": True
                            })
                    else:
                        # Record a failed attempt (helps capping attempts) ★
                        extracted.append({
                            "response_index": i,
                            "code": "",
                            "language": "unknown",
                            "method": "extraction_failed",
                            "verified": False,
                            "attempts": attempt_num,
                            "regenerated": prior_attempts > 0,
                            "extracted_successfully": False,
                            "error": "No code found in response"
                        })
                except Exception as e:
                    extracted.append({
                        "response_index": i,
                        "code": "",
                        "language": "unknown",
                        "method": "extraction_failed",
                        "verified": False,
                        "attempts": attempt_num,
                        "regenerated": prior_attempts > 0,
                        "extracted_successfully": False,
                        "error": str(e)
                    })

        if extracted:
            return {"key": key, "extracted": extracted}
        return None

    def parse_file(
        self,
        filename: str,
        max_responses_to_parse: int,
        resume: bool = True,
    ) -> str:
        """Parse JSONL file and return state file path."""
        parsed_dir, state_path = self._paths_for(filename)
        os.makedirs(parsed_dir, exist_ok=True)

        state = self._load_state(state_path, filename, max_responses_to_parse) if resume else {
            "meta": {
                "source_file": os.path.abspath(filename),
                "responses_per_entry": max_responses_to_parse,
                "updated_at": datetime.now().isoformat(timespec="seconds"),
                "task_name": self.__class__.__name__,
                "model_name": self.model_name,
            },
            "by_key": {}
        }

        # Read file
        with open(filename, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Process in parallel
        results = []
        processed = 0

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self._process_entry, line, state, max_responses_to_parse) 
                       for line in lines]

            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
                result = future.result()
                if result:
                    results.append(result)

                processed += 1
                if processed % self.save_every == 0:
                    # Update state with current results
                    for res in results:
                        key = res["key"]
                        if key not in state["by_key"]:
                            state["by_key"][key] = []
                        state["by_key"][key].extend(res["extracted"])

                    self._save_state(state, state_path)
                    results.clear()

        # Final save
        for res in results:
            key = res["key"]
            if key not in state["by_key"]:
                state["by_key"][key] = []
            state["by_key"][key].extend(res["extracted"])

        self._save_state(state, state_path)

        # Generate parsed JSONL file
        parsed_jsonl_path = self._write_parsed_jsonl(filename, state)

        return state_path

    def _write_parsed_jsonl(self, filename: str, state: Dict[str, Any]) -> str:
        """Write parsed JSONL file with original data + parsed_response field."""
        # Use same filename as input, but in parsed directory
        parsed_dir, _ = self._paths_for(filename)
        parsed_filename = os.path.basename(filename)  # Keep same filename
        parsed_path = os.path.join(parsed_dir, parsed_filename)

        # Read original file and create mapping
        with open(filename, "r", encoding="utf-8") as f:
            original_lines = f.readlines()

        parsed_lines = []
        by_key = state.get("by_key", {})

        for line in original_lines:
            try:
                original_data = json.loads(line.strip())
                unique_key = self.make_unique_key(original_data)

                # Add parsed responses if they exist
                if unique_key in by_key:
                    # Extract just the code from each parsed response
                    parsed_codes = [item.get("code", "") for item in by_key[unique_key]]
                    original_data["parsed_response"] = parsed_codes
                else:
                    original_data["parsed_response"] = []

                parsed_lines.append(json.dumps(original_data, ensure_ascii=False))

            except Exception:
                # If parsing fails, include original line without parsed_response
                try:
                    original_data = json.loads(line.strip())
                    original_data["parsed_response"] = []
                    parsed_lines.append(json.dumps(original_data, ensure_ascii=False))
                except:
                    # Skip malformed lines
                    continue

        # Write parsed JSONL
        with open(parsed_path, "w", encoding="utf-8") as f:
            for line in parsed_lines:
                f.write(line + "\n")

        print(f"✓ Parsed JSONL written to: {parsed_path}")
        return parsed_path