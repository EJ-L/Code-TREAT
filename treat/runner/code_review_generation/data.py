class CodeReviewData:
    def __init__(self, owner: str, repo_name: str, pr_id: int, diff_hunk: str, reviewer: str, code_review_comment: str, dataset: str, language: str):
        self.dataset = dataset
        self.language = language
        self.repo = repo_name
        self.owner = owner
        self.pr_id = pr_id
        self.diff_hunk = diff_hunk
        self.reviewer = reviewer
        self.code_review_comment = code_review_comment
        
        # Framework compatibility attributes for manifest support
        self.dataset_name = dataset  # For manifest matching
        # self.idx = idx  # For manifest matching
        
        # Create a composite identifier for code review data
        # Format: owner/repo_pr_id_diff_hunk for consistency
        self.composite_id = f"{owner}/{repo_name}_{pr_id}_{diff_hunk}"
    
    def __str__(self):
        return "Owner: " + self.owner + "\nRepo Name: " + self.repo + "\nPR ID: " + str(self.pr_id) + "\nDiff Hunk: " + self.diff_hunk + "\nReviewer: " + self.reviewer + "\nCode Review Comment: " + self.code_review_comment +"\nLanguage: " + self.language + "\n"
    
