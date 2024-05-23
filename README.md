![Quest Logo](logo.png)

--------------------------------------------------------------------------------

# QUEST: Quality-Aware Metropolis-Hastings Sampling for Machine Translation

Paper: arxiv link goes here
<!-- toc -->

-----
## <div align="center">Documentation</div>

-----
## <div align="center">Quick Start Examples</div>

<details open>
<summary>Install</summary>

Install using pip (recommended):

```bash
pip install quest-decoding
```

</details>


<details open>
<summary>Length Reward</summary>

Example using Length Reward

```python

    from langchain.prompts import PromptTemplate
    from quest.model.vllm import VLLM
    from quest.index import Uniform


    template =  PromptTemplate.from_template(
        "Translate this from {source_language} to {target_language}:\n{source_language}: {source_sentence}\n{target_language}:"
    ) # a prompt template you define.
    
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

    reward = LengthReward() # a reward you define.
    
    index = Uniform()
    
    chain = Quest(
        input_data=test_input_data,
        model=model,
        reward=reward,
        dist=index,   
    )
    
    chain_outputs = chain.run(
        steps=10,
        use_tqdm=True,
    )
    
     
    print(chain_outputs.samples)
        
```

</details>


-----

## <div align="center">Contact</div>

For bugs and feature requests please visit [GitHub Issues](https://github.com/goncalorafaria/quest-decoding/issues). For business inquiries or
professional support requests please send an [e-mail](goncalofaria.research@gmail.com).

-----

## <div align="center">Citation</div>

````
@inproceedings{
    questdecoding,
    title={QUEST: Quality-Aware Metropolis-Hastings Sampling for Machine Translation},
    author={Gonçalo Faria, Sweta Agrawal, António Farinhas, Ricardo Rei, José G. C. de Souza, Andre Martins},
    booktitle={},
    year={2024},
    url={arxiv link goes here}
}
````