from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer

DEFAULT_TEMPLATE = (
    PromptTemplate.from_template("{prompt}")
)


class LanguageModel:
    """
    Abstract base class for language models.

    Args:
        prompt_template (PromptTemplate): The template used to generate prompts.

    Attributes:
        prompt_template (PromptTemplate): The template used to generate prompts.
    """

    def __init__(
        self,
        prompt_template: PromptTemplate,
        temperature=1.0,
    ):
        self.prompt_template = (
            prompt_template
        )
        self.temperature = temperature

    def encode(self, prompt_data):
        """
        Encodes the given prompt data into a tokenized format.

        Args:
            prompt_data: The data used to generate prompts.

        Returns:
            The tokenized representation of the prompts.
        """
        prompt_txt = [
            self.get_prompt(**data)
            for data in prompt_data
        ]
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
            if k
            in self.prompt_template.input_variables
        }  # filter out relevant variables.

        prompt = (
            self.prompt_template.format(
                **input_data
            )
        )

        return prompt

    def continuation(
        self, x, prefix=None, **kwargs
    ):
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

    def evaluate_continuation(
        self,
        x,
        y,
    ):
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

    def decode_tokenize(
        self, ids, skip_special_tokens=False
    ):
        """
        Decodes and tokenizes the given IDs.

        Args:
            ids: The IDs to be decoded and tokenized.

        Raises:
            NotImplementedError: This method should be implemented by subclasses.
        """
        raise NotImplementedError()


class LocalLanguageModel(LanguageModel):

    def __init__(
        self,
        model_path: str,
        prompt_template: PromptTemplate = DEFAULT_TEMPLATE,
        temperature=1.0,
        max_new_tokens=600,
        max_prompt_length=300,
        stop_tokens=[],  # ["\n"],
        skip_special_tokens=False,
    ):
        super().__init__(
            prompt_template=prompt_template,
            temperature=temperature,
        )

        self.model_path = model_path

        self.max_new_tokens = max_new_tokens
        self.max_prompt_length = (
            max_prompt_length
        )
        self.stop_tokens = stop_tokens
        self.skip_special_tokens = (
            skip_special_tokens
        )
        self.tokenizer = (
            AutoTokenizer.from_pretrained(
                model_path,
                padding_side="left",
            )
        )

        if (
            self.tokenizer.pad_token_id
            is None
        ):
            self.tokenizer.pad_token_id = (
                self.tokenizer.bos_token_id
            )  # THIS IS ACTUALLY REALLY IMPORTANT :) THIS HIDDEN NIGHTMARE DONT USE EOS. - w/ AR models in batch we may have padding in the beginig - obvious reason left to right gen.
            self.tokenizer.pad_token = (
                self.tokenizer.bos_token
            )

    def tokenize(self, prompt):

        return [
            self.tokenizer.encode(
                p,
                max_length=self.max_prompt_length,
                truncation=True,
                # return_tensors="np",q
            )
            for p in prompt
        ]

    def decode_tokenize(self, ids):
        return self.tokenizer.batch_decode(
            ids,
            skip_special_tokens=self.skip_special_tokens,
        )
