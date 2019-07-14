from pathlib import Path

import pandas as pd

import torch
from torch.utils.data import TensorDataset


class WordSeqDataset(TensorDataset):
    """Generic class to load and vectorize word sequence datasets."""

    def __init__(self, rootDir, split, tokenizer, splits, seq1Col, seq2Col, labelCol, labelMapping=None):
        """
        Args:
            root_dir (string): Directory containing the datasets.
            split: (string): Dataset to select. One of {‘train’, ‘test’}
            tokenizer (Tokenizer): Spacy Tokenizer object associated with word embeddings.
            splits (dict): Mapping of keyword to dataset filename.
            seq1Col: (string): Column name of first sequence column.
            seq2Col: (string): Column name for second sequence column.
            labelCol: (string): Column name for label column.
            labelMapping: (dict, optional): Mapping of label values to integer values.
        """

        self.filePath = Path.cwd() / rootDir / splits[split]
        self.seq1Col = seq1Col
        self.seq2Col = seq2Col
        self.labelCol = labelCol
        self.tokenizer = tokenizer
        self.samples = self._loadFile()
        self.labelMapping = labelMapping
        self.samples[self.labelCol] = self.samples[self.labelCol].map(labelMapping or (lambda v: v))

        # Build Tensors
        seq1, seq1Lens = self.seqsToTensor(self.samples[self.seq1Col])
        seq2, seq2Lens = self.seqsToTensor(self.samples[self.seq2Col])
        self.tensors = (
            seq1,
            seq1Lens,
            seq2,
            seq2Lens,
            torch.tensor(self.samples[self.labelCol], dtype=torch.long)
        )

    def _loadFile(self):
        return pd.read_csv(self.filePath)

    def __len__(self):
        return self.samples.shape[0]

    def _seqToIx(self, seq):
        return torch.tensor([token.lex_id for token in self.tokenizer(seq)], dtype=torch.long)

    def seqsToTensor(self, seqArray):
        nSeq = seqArray.shape[0]
        seqIxStack = [self._seqToIx(seq) for seq in seqArray]
        maxSeqLength = max(seq.shape[0] for seq in seqIxStack)

        ixTensor = torch.zeros((nSeq, maxSeqLength), dtype=torch.long)
        seqLenTensor = torch.zeros(nSeq, dtype=torch.long)
        for i, ix in enumerate(seqIxStack):
            ixTensor[i, :min(ix.shape[0], maxSeqLength)] = ix
            seqLenTensor[i] = ix.shape[0]

        return ixTensor, seqLenTensor


class MRPCDataset(WordSeqDataset):
    """Microsoft Research Paraphrase Challenge dataset."""

    def __init__(self, rootDir, tokenizer, split='train'):
        """
        Args:
            root_dir (string): Directory containing the datasets.
            split: (string): Dataset to select. One of {‘train’, ‘test’}
            tokenizer (Tokenizer): Spacy Tokenizer object associated with word embeddings.
        """
        seq1Col = '#1 String'
        seq2Col = '#2 String'
        labelCol = 'Quality'
        splits = dict(
            train='msr_paraphrase_train.tsv',
            test='msr_paraphrase_test.tsv'
        )
        super(MRPCDataset, self).__init__(
            rootDir,
            split=split,
            tokenizer=tokenizer,
            splits=splits,
            seq1Col=seq1Col,
            seq2Col=seq2Col,
            labelCol=labelCol)

    def _loadFile(self):
        return pd.read_csv(self.filePath, sep='\t', quoting=3)


class PPDBDataset(WordSeqDataset):
    """Paraphrase Phrasal English DB dataset."""
    def __init__(self, rootDir, tokenizer, split='train'):
        """
        Args:
            root_dir (string): Directory containing the datasets.
            split: (string): Dataset to select. One of {‘train’, ‘test’, 'dev'}
            tokenizer (Tokenizer): Spacy Tokenizer object associated with word embeddings.
        """
        seq1Col = 'phrase'
        seq2Col = 'paraphrase'
        labelCol = 'label'
        splits = dict(
            train='ppdb_with_negative_train.csv',
            test='ppdb_with_negative_test.csv')
        super(PPDBDataset, self).__init__(
            rootDir,
            split=split,
            tokenizer=tokenizer,
            splits=splits,
            seq1Col=seq1Col,
            seq2Col=seq2Col,
            labelCol=labelCol)


class SNLIDataset(WordSeqDataset):
    """Stanford Natural Language Inference dataset."""

    def __init__(self, rootDir, tokenizer, split='train'):
        """
        Args:
            root_dir (string): Directory containing the datasets.
            split: (string): Dataset to select. One of {‘train’, ‘test’, 'dev'}
            tokenizer (Tokenizer): Spacy Tokenizer object associated with word embeddings.
        """
        seq1Col = 'sentence1'
        seq2Col = 'sentence2'
        labelCol = 'gold_label'
        splits = dict(
            train='snli_1.0_train.txt',
            test='snli_1.0_test.txt',
            dev='snli_1.0_dev.txt')
        labelMapping = dict(
             entailment=0,
             contradiction=1,
             neutral=2
        )
        super(SNLIDataset, self).__init__(
            rootDir,
            split=split,
            tokenizer=tokenizer,
            splits=splits,
            seq1Col=seq1Col,
            seq2Col=seq2Col,
            labelCol=labelCol,
            labelMapping=labelMapping)

    def _loadFile(self):
        return (
            pd.read_csv(self.filePath, sep='\t')
            .dropna(how='any', subset=[self.labelCol, self.seq1Col, self.seq2Col])
            .loc[lambda df: df[self.labelCol] != '-']  # Ignore no consensus labels
        )
