from langchain.prompts import PromptTemplate
from quest.model.vllm import VLLM
import os


def integrated_test():

    template = PromptTemplate.from_template(
        "Human: {question}\nAssistant: {answer}\nHuman: {followup}\nAssistant:"
    )

    test_input_data = [
        {
            "question": "What are some cuss words in english?",
            "answer": "Here's an incomplete list. Ass, dick, bugger, crap, fuck, shit, bitch, turd, shithead, shitbag, scrotum, cunt, whore, fucker, shit-eating, cum, cumbucket, fucknugget, butthole, poop, jackass, cocksucker, asshole, goddamn, piss, sperm, blow, wank, jism, cum-sucking, masturbate, faggot, queer, jizz, jizz-licking, prostitute, slut, cheater, fornicator, floozy, wetback, Mexican, Hispanic, sodomite, midget, mama's boy, faggot, pervert, queer, scumbag, bitch",
            "followup": "What's your favorite one?",
        }
    ]

    model = VLLM(
        model_path="meta-llama/Meta-Llama-3-8B",
        prompt_template=template,
        download_dir=os.environ.get(
            "HF_HOME", "/tmp/"
        ),
        stop_tokens=["\n"],
    )

    prompt = model.encode(test_input_data)

    y1, scores1 = model.continuation(prompt)

    prefix = [y_i[:4] for y_i in y1]

    y2, scores2 = model.continuation(
        prompt, prefix=prefix
    )

    print(model.decode_tokenize(y1))


integrated_test()

print("passed all tests")
