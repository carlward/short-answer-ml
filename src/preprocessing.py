from collections import namedtuple
from pathlib import Path

import numpy as np
import pandas as pd

import torch
from torch.utils.data import TensorDataset

SEQ_1_COL = '#1 String'
SEQ_2_COL = '#2 String'
LABEL_COL = 'Quality'

SeqPair = namedtuple('SeqPair', ['seq1', 'seq2', 'label'])


class MRPCDataset(TensorDataset):
    """Microsoft Research Paraphrase Challenge dataset."""

    def __init__(self, root_dir, tokenizer, max_seq_length=200, split='train'):
        """
        Args:
            root_dir (string): Directory containing the datasets.
            tokenizer (Tokenizer): Spacy Tokenizer object associated with word embeddings.
            max_seq_length (int): Max length that sequences will be trimmed/padded to.
            split: (string): Dataset to select. One of {‘train’, ‘test’}
        """
        file_name = 'msr_paraphrase_train.tsv' if split == 'train' else 'msr_paraphrase_test.tsv'
        self.file_path = Path.cwd() / root_dir / file_name
        self.seq_1_col = '#1 String'
        self.seq_2_col = '#2 String'
        self.label_col = 'Quality'
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer
        self.samples = pd.read_csv(self.file_path, sep='\t', quoting=3)
        self.samples[self.label_col] = self.samples[self.label_col].map(lambda v: -1 if v == 0 else v)

        # Build Tensors
        self.tensors = SeqPair(
            self.seqs_to_tensor(self.samples[self.seq_1_col]),
            self.seqs_to_tensor(self.samples[self.seq_2_col]),
            torch.tensor(self.samples[self.label_col], dtype=torch.long)
        )

    def _prepare_sequence(self, seq):
        idxs = self._seq_to_ix(seq)
        return torch.tensor(idxs, dtype=torch.long)

    def _seq_to_ix(self, seq):
        return np.array([token.lex_id for token in self.tokenizer(seq)])

    def seqs_to_tensor(self, seq_array):
        n_seq = len(seq_array)
        seq_ix_stack = (self._prepare_sequence(seq) for seq in seq_array)
        ix_tensor = torch.zeros(n_seq, self.max_seq_length, dtype=torch.long)

        for i, ix in enumerate(seq_ix_stack):
            ix_tensor[i, :min(len(ix), self.max_seq_length)] = ix

        return ix_tensor
