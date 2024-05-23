from langchain.prompts import PromptTemplate
from quest.model.vllm import VLLM
import os

def integrated_test():
    
    template =  PromptTemplate.from_template(
        "Translate this from {source_language} to {target_language}:\n{source_language}: {source_sentence}\n{target_language}:"
    )
    
    test_input_data = [{
        "source_language": "English",
        "target_language": "French",
        "source_sentence": "Hello, how are you?"
    }]

    model = VLLM(
        model_path="haoranxu/ALMA-7B",
        prompt_template=template,
        download_dir=os.environ["HF_HOME"],
    )


    prompt = model.encode(
        test_input_data
    )

    y1, scores1 = model.continuation(prompt)
    
    import pdb; pdb.set_trace()
    
    prefix = [ y_i[:4] for y_i in y1]

    y2, scores2 = model.continuation(prompt,prefix=prefix)


integrated_test()

print("passed all tests")