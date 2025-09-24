
class Data:
    def __init__(self, id, dataset, title, problem_description, difficulty, release_date, func_sign, driver_code, starter_code, class_name=None, domain=None):
        self.id = id
        self.dataset = dataset
        self.title = title
        self.problem_description = problem_description
        self.difficulty = difficulty
        self.release_date = release_date
        self.func_sign = func_sign
        self.driver_code = driver_code
        self.starter_code = starter_code
        self.class_name = class_name
        self.domain = domain

class GeeksforGeeksData(Data):
    def __init__(self, id, question_title, problem_description, difficulty, release_date, func_sign, driver_code, starter_code, class_name=None):
        super().__init__(id, "geeksforgeeks", question_title, problem_description, difficulty, release_date, func_sign, driver_code, starter_code, class_name, "geeksforgeeks")
        
            
class HackerrankData(Data):
    def __init__(self, id, title, problem_description, difficulty, release_date, func_sign, driver_code, starter_code, class_name=None):
        super().__init__(id, "hackerrank", title, problem_description, difficulty, release_date, func_sign, driver_code, starter_code, class_name, "hackerrank")