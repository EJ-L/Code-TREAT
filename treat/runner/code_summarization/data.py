class CodeSumData:
    def __init__(self, data: dict, default_dataset: str | None = None):
        owner = (data.get("owner") or "").strip()
        repo_name = (data.get("repo_name") or "").strip()

        self.dataset_name = data.get("dataset") or data.get("dataset_name") or default_dataset
        self.dataset = self.dataset_name
        self.lang = data.get("lang")

        self.owner = owner
        self.repo_name = repo_name
        self.repo = f"{owner}/{repo_name}" if owner and repo_name else owner or repo_name

        self.sha = data.get("sha", "")
        self.url = data.get("url", "")
        self.func_name = data.get("func_name", "")
        self.code = data.get("code", "")
        self.docstring = data.get("docstring", "")

        parts = [self.dataset_name, self.repo, self.sha, self.func_name, self.url]
        composite = "::".join(str(p) for p in parts if p)
        self.key = composite or self.url or self.sha or self.func_name
        self.id = self.key
