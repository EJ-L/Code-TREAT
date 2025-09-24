class CodeSumData:
    def __init__(self, data: dict):
        self.data = data
        self.code = data.get("code/function", "")
        self.docstring = data.get("docstring", "")
        self.func_name = data.get("func_name", "")
        self.repo = data.get("repo", "")
        self.sha = data.get("sha", "")
        self.url = data.get("url", "")
        self.start_line = data.get("start_line", "")
        self.end_line = data.get("end_line", "")