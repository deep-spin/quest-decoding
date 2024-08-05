from collections import OrderedDict


def join_accepted_values(
    accept, proposal, state
):
    return [
        p if a else s
        for a, p, s in zip(
            accept, proposal, state
        )
    ]


def flatten_list(lst):
    flattened = []
    for item in lst:
        if isinstance(item, list):
            flattened.extend(
                flatten_list(item)
            )
        else:
            flattened.append(item)
    return flattened


def unflatten_list(flat_data, counts):

    unflattened_translations = []
    start = 0
    for count in counts:
        end = start + count
        unflattened_translations.append(
            flat_data[start:end]
        )
        start = end

    return unflattened_translations


class OrderedSet:
    def __init__(self, elements=[]):
        self.elements = OrderedDict()

        for e in elements:
            self.add(e)

    def add(self, item):
        self.elements[item] = None

    def remove(self, item):
        del self.elements[item]

    def __iter__(self):
        return iter(self.elements)

    def __contains__(self, item):
        return item in self.elements

    def __len__(self):
        return len(self.elements)


def get_unique_mapping(si):
    vocab = {
        s: i
        for i, s in enumerate(
            OrderedSet(si)
        )
    }
    tokens = [vocab[s] for s in si]
    sorted_vocab = [
        k
        for k, v in sorted(
            vocab.items(),
            key=lambda item: item[1],
        )
    ]

    return tokens, sorted_vocab


def invert_unique_mapping(tokens, rsi):
    return [rsi[t] for t in tokens]
