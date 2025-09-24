from extractors.regex_extraction_utils.code_execution_utils import get_assertion, get_expected, get_input, mask_inputs
import json
import pickle

class Data:
    def __init__(self, _id, difficulty, language, function, dataset, test_case_info):
        self.id = _id
        self.dataset = dataset
        self.difficulty = difficulty
        self.language = language
        self.function = function
        self.test_case_info = test_case_info
        self._organize()

    def _organize(self):
        self.input_prediction_query = self.test_case_info['input_masked_code']
        self.output_prediction_query = self.test_case_info['output_masked_code']
        self.test_case_idx = self.test_case_info['test_case_idx']
    
class HackerrankData(Data):
    def __init__(self, _id, difficulty, language, function,  test_case_info):
        super().__init__(_id, difficulty, language, function, 'hackerrank', test_case_info)
        
class GeeksforGeeksData(Data):
    def __init__(self, _id, difficulty, language, function,  test_case_info):
        super().__init__(_id, difficulty, language, function, 'geeksforgeeks', test_case_info)
