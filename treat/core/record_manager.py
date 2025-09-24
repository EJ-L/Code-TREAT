from typing import Dict, List, Any, Optional
import threading
import json
import os

class RecordManager:
    """
    Minimal record manager.
    Layout: results/<task>/<dataset>/<model_name>.jsonl
    Tracks:
      - ref_index: seen ref_keys (for fast skip)
      - ref_counts: number of outputs already saved per ref_key (for resume/top-up)
    """

    def __init__(self, task_name: str, dataset_name: str, save_dir: str = "results"):
        self.task_name = task_name
        self.save_dir = save_dir
        self.record_dir = os.path.join(self.save_dir, self.task_name, dataset_name, "predictions")
        os.makedirs(self.record_dir, exist_ok=True)
        self._lock = threading.Lock()
        self.ref_index: Dict[str, bool] = {}
        self.ref_counts: Dict[str, int] = {}

    # ---------- Path helpers ----------
    def _file_path(self, model_name: str) -> str:
        model_name_fs = model_name.replace("/", "_")
        return os.path.join(self.record_dir, f"{model_name_fs}.jsonl")

    # ---------- Flexible ref key builder ----------
    @staticmethod
    def make_ref_key(*parts: Any, _sep: str = "_") -> str:
        """
        Build a composite key:
          - numbers -> str(num)
          - list/tuple -> '__'.join(str(x) for x in seq)
          - strings/other -> str(x)
          - join parts with '_'
        """
        norm: List[str] = []
        for p in parts:
            if p is None:
                norm.append("None")
            elif isinstance(p, (int, float)):
                norm.append(str(p))
            elif isinstance(p, (list, tuple)):
                norm.append("__".join(str(x) for x in p))
            else:
                norm.append(str(p))
        return _sep.join(norm)

    @staticmethod
    def _count_outputs_in_record(record: Dict[str, Any]) -> int:
        """
        Prefer 'response'.
        Each must be a list to be counted.
        """
        out = record.get("response")
        return len(out) if isinstance(out, list) else 0

    # ---------- Key helpers ----------
    def compose_ref_key_from_record_fields(self, record: Dict[str, Any]) -> Optional[str]:
        """
        Build a ref_key from typical fields if missing.
        Expected shape (whatever you persist): dataset, lang/target_lang, prompt_category/prompting_category, model_name, prompt_id.
        Falls back gracefully if fields are missing.
        """
        dataset = record.get("dataset")
        lang = record.get("lang") or record.get("target_lang")
        # allow prompt_category to be str or list
        cats = record.get("prompt_category") or record.get("prompting_category") or []
        if isinstance(cats, str):
            cats = [cats]
        model_name = record.get("model_name")
        prompt_id = record.get("prompt_id")

        # If any core piece is missing, give up (caller may supply its own key)
        if None in (dataset, lang, model_name, prompt_id):
            return None
        return self.make_ref_key(dataset, lang, cats, model_name, prompt_id)

    # ---------- Queries ----------
    def is_ref_processed(self, ref_key: str) -> bool:
        return ref_key in self.ref_index

    def get_ref_count(self, ref_key: str) -> int:
        return self.ref_counts.get(ref_key, 0)

    def mark_ref_add(self, ref_key: str, added: int):
        if added <= 0:
            return
        self.ref_counts[ref_key] = self.ref_counts.get(ref_key, 0) + added

    def mark_ref_processed(self, ref_key: str):
        self.ref_index[ref_key] = True

    def attach_ref_key(self, record: Dict[str, Any]) -> Optional[str]:
        """
        Ensure record['ref_key'] exists. If not, compute it from record fields.
        Also mark the key processed in the in-memory index.
        """
        key = record.get("ref_key")
        if not key:
            key = self.compose_ref_key_from_record_fields(record)
            if key:
                record["ref_key"] = key
        if key:
            self.mark_ref_processed(key)
        return key

    # ---------- Load / Save ----------
    def load_records(self, models: List[Any]) -> None:
        """
        Populate in-memory indices (ref_index, ref_counts) from existing jsonl logs
        under results/<task>/<dataset>/ for the given models.
        """
        self.ref_index.clear()
        self.ref_counts.clear()

        os.makedirs(self.record_dir, exist_ok=True)

        for model in models:
            model_name = getattr(model, "model_name", None)
            if not model_name:
                print("[RecordManager] Warning: model without 'model_name'; skipping.")
                continue
            path = self._file_path(model_name)
            if not os.path.exists(path):
                continue
            self._load_one_file_into_index(path)

    def _load_one_file_into_index(self, filepath: str):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                    except Exception:
                        continue

                    # obtain key
                    key = record.get("ref_key")
                    if not key:
                        key = self.compose_ref_key_from_record_fields(record)

                    if not key:
                        continue

                    self.ref_index[key] = True
                    self.ref_counts[key] = self.ref_counts.get(key, 0) + self._count_outputs_in_record(record)
        except Exception as e:
            print(f"[RecordManager] Warning: error loading '{filepath}': {e}")

    def save_record(self, model_name: str, record: Dict[str, Any], fsync: bool = False):
        """
        Append record to results/<task>/<dataset>/<model_name>.jsonl.
        Ensures ref_key is present and updates in-memory counts.
        """
        os.makedirs(self.record_dir, exist_ok=True)
        filepath = self._file_path(model_name)

        if "_schema_version" not in record:
            record["_schema_version"] = 1

        key = self.attach_ref_key(record)
        added = self._count_outputs_in_record(record)
        if key:
            self.mark_ref_add(key, added)

        line = json.dumps(record, ensure_ascii=False)
        with self._lock:
            with open(filepath, "a", encoding="utf-8") as f:
                f.write(line + "\n")
                if fsync:
                    f.flush()
                    os.fsync(f.fileno())
