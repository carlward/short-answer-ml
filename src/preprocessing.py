from pathlib import Path

import pandas as pd

import torch
from torch.utils.data import TensorDataset
SEQ_1_COL = '#1 String'
SEQ_2_COL = '#2 String'
LABEL_COL = 'Quality'


class MRPCDataset(TensorDataset):
    """Microsoft Research Paraphrase Challenge dataset."""

    def __init__(self, root_dir, tokenizer, split='train'):
        """
        Args:
            root_dir (string): Directory containing the datasets.
            tokenizer (Tokenizer): Spacy Tokenizer object associated with word embeddings.
            split: (string): Dataset to select. One of {‘train’, ‘test’}
        """
        file_name = 'msr_paraphrase_train.tsv' if split == 'train' else 'msr_paraphrase_test.tsv'
        self.file_path = Path.cwd() / root_dir / file_name
        self.seq_1_col = '#1 String'
        self.seq_2_col = '#2 String'
        self.label_col = 'Quality'
        self.tokenizer = tokenizer
        self.samples = pd.read_csv(self.file_path, sep='\t', quoting=3)
        self.samples[self.label_col] = self.samples[self.label_col].map(lambda v: -1 if v == 0 else v)

        # Build Tensors
        left_seq, left_lens = self.seqs_to_tensor(self.samples[self.seq_1_col])
        right_seq, right_lens = self.seqs_to_tensor(self.samples[self.seq_2_col])

        self.tensors = (
            left_seq,
            left_lens,
            right_seq,
            right_lens,
            torch.tensor(self.samples[self.label_col], dtype=torch.long)
        )

    def __len__(self):
        return self.samples.shape[0]

    def _seq_to_ix(self, seq):
        return torch.tensor([token.lex_id for token in self.tokenizer(seq)], dtype=torch.long)

    def seqs_to_tensor(self, seq_array):
        n_seq = len(seq_array)
        seq_ix_stack = (self._seq_to_ix(seq) for seq in seq_array)
        max_seq_length = seq_array.map(len).max()

        ix_tensor = torch.zeros((n_seq, max_seq_length), dtype=torch.long)
        seq_len_tensor = torch.zeros(n_seq, dtype=torch.long)
        for i, ix in enumerate(seq_ix_stack):
            ix_tensor[i, :min(ix.size(0), max_seq_length)] = ix
            seq_len_tensor[i] = ix.size(0)

        return ix_tensor, seq_len_tensor
