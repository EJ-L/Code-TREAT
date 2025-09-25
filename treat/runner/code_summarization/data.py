class CodeSumData:
    def __init__(self, data: dict):
        self.repo = data.get("owner", "") + "/" + data.get("repo_name", "")
        self.code = data.get("code", "")
        self.docstring = data.get("docstring", "")
        self.func_name = data.get("func_name", "")
        self.url = data.get("url", "")
        self.sha = data.get("sha", "")