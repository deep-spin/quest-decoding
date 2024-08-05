from torch.utils.data import (
    Dataset,
    DataLoader,
)
from typing import List
from tqdm import tqdm


class TextDataset(Dataset):
    def __init__(
        self,
        texts,
        tokenizer,
        max_length=512,
    ):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
            # pad_token=self.tokenizer.eos_token,
        )
        return {
            "input_ids": encoding[
                "input_ids"
            ].flatten(),
            "attention_mask": encoding[
                "attention_mask"
            ].flatten(),
        }


def get_loader(
    candidates: List[str],
    tokenizer,
    batch_size: int = 32,
    use_tqdm: bool = False,
    max_length: int = 512,
):
    ds = TextDataset(
        candidates,
        tokenizer,
        max_length=max_length,
    )
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
    )

    if use_tqdm:
        loader = tqdm(loader)

    return loader
