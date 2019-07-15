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


class ASAPSASDataset(WordSeqDataset):
    """Automated Student Assessment Prize Phase 2 dataset."""
    def __init__(self, rootDir, tokenizer, split='train', refresh=False, seed=42):
        """
        Args:
            root_dir (string): Directory containing the datasets.
            tokenizer (Tokenizer): Spacy Tokenizer object associated with word embeddings.
            split: (string): Dataset to select. One of {‘train’, ‘test’, 'dev'}
            refresh: (bool): Flag to rebuild derived datasets.
            seed: (int): Random seed when sampling the derived datasets.
        """
        seq1Col = 'referencetext'
        seq2Col = 'essaytext'
        labelCol = 'score'
        splits = dict(
            train='train_with_reference_text_.csv',
            test='test_with_reference_text_.csv',
            dev='dev_with_reference_text_.csv')

        self.seed = seed
        if not all((Path.cwd() / rootDir / f).exists() for f in splits):
            self.buildDatasets(rootDir, splits)

        super(ASAPSASDataset, self).__init__(
            rootDir,
            split=split,
            tokenizer=tokenizer,
            splits=splits,
            seq1Col=seq1Col,
            seq2Col=seq2Col,
            labelCol=labelCol)

    def _selectReferenceTextId(self, essaySet):
        # Pick median length top scoring answers as the reference for each essay set
        topScoring = essaySet[essaySet['score'] == essaySet['score'].max()]
        medianIdx = topScoring.shape[0] // 2
        essaySet['referenceId'] = topScoring.sort_values('essaylength').iloc[medianIdx]['id']
        return essaySet

    def buildDatasets(self, rootDir, splits, trainFrac=0.7, testFrac=0.2):
        devFrac = 1-trainFrac-testFrac
        assert devFrac + trainFrac + testFrac == 1

        # Format and combine all labeled datasets
        trainSet = (
            pd.read_csv(Path.cwd() / rootDir / 'train_rel_2.tsv', sep='\t')
            .rename(columns=lambda col: col.lower())
            .rename(columns=dict(score1='score'))
            .loc[:, ['id', 'essayset', 'score', 'essaytext']]
        )
        leaderBoardLabels = pd.read_csv(Path.cwd() / rootDir / 'public_leaderboard_solution.csv').set_index('id')
        leaderBoardSet = (
            pd.read_csv(Path.cwd() / rootDir / 'public_leaderboard_rel_2.tsv', sep='\t')
            .rename(columns=lambda col: col.lower())
            .join(leaderBoardLabels, on='id', how='inner')
            .rename(columns=dict(essay_score='score'))
            .loc[:, ['id', 'essayset', 'score', 'essaytext']]
        )
        allLabeled = pd.concat([trainSet, leaderBoardSet], ignore_index=True)

        # Calculate reference answers to act as ground truth teacher labels
        allLabeled = (
            allLabeled
            .assign(essaylength=lambda df: df['essaytext'].map(len))
            .groupby('essayset').apply(self._selectReferenceTextId)
            .set_index('id')
            .assign(referencetext=lambda df: df.loc[df['referenceId'], 'essaytext'].values)
            .loc[
                lambda df: ~df.index.isin(df['referenceId'].unique()),  # Drop reference essays
                ['essayset', 'score', 'essaytext', 'referencetext']
            ]
        )

        # Sample new datasets and persist
        newTrain = allLabeled.sample(frac=trainFrac, random_state=self.seed)
        remaining = allLabeled.drop(newTrain.index)
        newTest = remaining.sample(frac=testFrac, random_state=self.seed)
        newDev = remaining.drop(newTest.index)

        newTrain.to_csv(Path.cwd() / rootDir / splits['train'])
        newTest.to_csv(Path.cwd() / rootDir / splits['test'])
        newDev.to_csv(Path.cwd() / rootDir / splits['dev'])


class MRPCDataset(WordSeqDataset):
    """Microsoft Research Paraphrase Challenge dataset."""

    def __init__(self, rootDir, tokenizer, split='train'):
        """
        Args:
            root_dir (string): Directory containing the datasets.
            tokenizer (Tokenizer): Spacy Tokenizer object associated with word embeddings.
            split: (string): Dataset to select. One of {‘train’, ‘test’}
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
            tokenizer (Tokenizer): Spacy Tokenizer object associated with word embeddings.
            split: (string): Dataset to select. One of {‘train’, ‘test’, 'dev'}
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
            tokenizer (Tokenizer): Spacy Tokenizer object associated with word embeddings.
            split: (string): Dataset to select. One of {‘train’, ‘test’, 'dev'}
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
