![Quest Logo](logo.png)

--------------------------------------------------------------------------------

# QUEST: Quality-Aware Metropolis-Hastings Sampling for Machine Translation

Gonçalo Faria, Sweta Agrawal, António Farinhas, Ricardo Rei, José G. C. de Souza, Andre Martins

**Paper**: arxiv link goes here

**TL;DR:** This paper presents a method to generate diverse and high-quality machine translations by sampling from a Gibbs distribution using the Metropolis-Hastings algorithm.

### Abstract:
An important challenge in machine translation (MT) is to generate high-quality and diverse translations. Prior work has shown that the estimated likelihood from the MT model correlates poorly with translation quality. In contrast, quality evaluation metrics (such as COMET or BLEURT) exhibit high correlations with human judgments, which has motivated their use as rerankers (such as quality-aware and minimum Bayes risk decoding). However, relying on a single translation with high estimated quality increases the chances of "gaming the metric''. In this paper, we address the problem of sampling a set of high-quality and diverse translations. We provide a simple and effective way to avoid over-reliance on noisy quality estimates by using them as the energy function of a Gibbs distribution. Instead of looking for a mode in the distribution, we generate multiple samples from high-density areas through the Metropolis-Hastings algorithm, a simple Markov chain Monte Carlo approach. The results show that our proposed method leads to high-quality and diverse outputs across multiple language pairs (English$\leftrightarrow${German, Russian}) with two strong decoder-only LLMs (Alma-7b, Tower-7b).
<!-- toc -->

-----
## <div align="center">Documentation</div>

TBD

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


```python

    from langchain.prompts import PromptTemplate
    from quest import VLLM, Uniform


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