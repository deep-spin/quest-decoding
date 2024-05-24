from langchain.prompts import PromptTemplate

class LanguageModel:
    def __init__(self,
            prompt_template: PromptTemplate):
        
        self.prompt_template=prompt_template

    def encode(self, prompt_data):
        prompt_txt = [self.get_prompt(**data) for data in prompt_data]
        return self.tokenize(prompt_txt)
    
    def get_prompt(self, **input_data):

        input_data = {
            k: v
            for k, v in input_data.items()
            if k in self.prompt_template.input_variables
        } # filter out relevant variables.

        prompt = self.prompt_template.format(**input_data)

        return prompt

    def continuation(self, x, prefix=None, **kwargs):
        raise NotImplementedError()
    
    def evaluate_continuation(self, x, y, temperature=1.0):
        raise NotImplementedError()


    def tokenize(self, prompt):
        raise NotImplementedError()

    def decode_tokenize(self, ids):
        raise NotImplementedError()
    
    
