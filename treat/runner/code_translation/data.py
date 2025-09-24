from itertools import combinations

class Data:
    def __init__(self, id, dataset, domain=None):
        self.id = id
        self.dataset = dataset
        self.domain = domain
    
    @property
    def dataset_name(self):
        return self.dataset
        
        
class PolyHumanEvalData(Data):
    def __init__(self, id, source_code):
        super().__init__(id, dataset="polyhumaneval", domain="polyhumaneval")        
        self.source_code = source_code
            
class HackerrankData(Data):
    def __init__(self, id, title, difficulty, domain, release_date, source_code):
        super().__init__(id, dataset="hackerrank", domain=domain)
        self.title = title
        self.difficulty = difficulty
        self.release_date = release_date
        self.source_code = source_code