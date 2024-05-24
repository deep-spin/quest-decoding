
def join_accepted_values(accept, proposal, state):
    return [p if a else s for a, p, s in zip(accept, proposal, state)]
