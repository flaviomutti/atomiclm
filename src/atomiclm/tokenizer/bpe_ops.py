def merge(ids, pair, idx):
    new_ids = []
    i = 0

    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            new_ids.append(idx)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1

    return new_ids


def get_stats(ids, counts=None):
    counts = {} if counts is None else counts
    if len(ids) < 2:
        return counts

    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1

    return counts
