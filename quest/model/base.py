from langchain.prompts import PromptTemplate

class LanguageModel:
    """
    Abstract base class for language models.

    Args:
        prompt_template (PromptTemplate): The template used to generate prompts.

    Attributes:
        prompt_template (PromptTemplate): The template used to generate prompts.
    """

    def __init__(self, prompt_template: PromptTemplate):
        self.prompt_template = prompt_template

    def encode(self, prompt_data):
        """
        Encodes the given prompt data into a tokenized format.

        Args:
            prompt_data: The data used to generate prompts.

        Returns:
            The tokenized representation of the prompts.
        """
        prompt_txt = [self.get_prompt(**data) for data in prompt_data]
        return self.tokenize(prompt_txt)

    def get_prompt(self, **input_data):
        """
        Generates a prompt using the input data.

        Args:
            **input_data: The input data used to generate the prompt.

        Returns:
            The generated prompt.
        """
        input_data = {
            k: v
            for k, v in input_data.items()
            if k in self.prompt_template.input_variables
        }  # filter out relevant variables.

        prompt = self.prompt_template.format(**input_data)

        return prompt

    def continuation(self, x, prefix=None, **kwargs):
        """
        Generates a continuation given an input sequence.

        Args:
            x: The input sequence.
            prefix: The prefix to be added to the continuation.
            **kwargs: Additional keyword arguments.

        Raises:
            NotImplementedError: This method should be implemented by subclasses.
        """
        raise NotImplementedError()

    def evaluate_continuation(self, x, y, temperature=1.0):
        """
        Evaluates the continuation of an input sequence.

        Args:
            x: The input sequence.
            y: The continuation sequence.
            temperature (float): The temperature parameter for sampling.

        Raises:
            NotImplementedError: This method should be implemented by subclasses.
        """
        raise NotImplementedError()

    def tokenize(self, prompt):
        """
        Tokenizes the given prompt.

        Args:
            prompt: The prompt to be tokenized.

        Raises:
            NotImplementedError: This method should be implemented by subclasses.
        """
        raise NotImplementedError()

    def decode_tokenize(self, ids):
        """
        Decodes and tokenizes the given IDs.

        Args:
            ids: The IDs to be decoded and tokenized.

        Raises:
            NotImplementedError: This method should be implemented by subclasses.
        """
        raise NotImplementedError()
    
    
