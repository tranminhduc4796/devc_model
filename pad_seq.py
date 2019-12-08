import torch
from torch.nn.utils.rnn import pad_sequence


def pad_seq(batch):
    """
    Pad the data in a batch so we can train by batch.
    :param batch: Batch of sequences.
    :return: Padded sequences, their lengths and the target.
    """
    hours, users, shops, previous_shops_batch, ratings = zip(*batch)

    # Get list of length of each sequence.
    previous_shops_lens = [len(previous_shops) for previous_shops in previous_shops_batch]

    # Padding the sequence.
    padded_previous_shops_batch = pad_sequence(previous_shops_batch, batch_first=True, padding_value=0)
    return hours, users, shops, padded_previous_shops_batch, previous_shops_lens, ratings
