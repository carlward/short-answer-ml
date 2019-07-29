from pathlib import Path
from collections import Counter, namedtuple

import numpy as np
import torch
from spacy.attrs import ORTH
from spacy.tokenizer import Tokenizer
from spacy.vocab import Vocab
from spacy.vectors import Vectors
from spacy.lang.en import English

SeqPair = namedtuple('SeqPair', ['leftSeq', 'leftLen', 'rightSeq', 'rightLen', 'label'], defaults=(-1, ))


class Seq2IdxTransformer(object):
    """Transform text sequence to array of word vector indices"""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def seqToIx(self, seq):
        return torch.tensor([token.lex_id for token in self.tokenizer(seq)], dtype=torch.long)

    def seqsToTensor(self, seqArray):
        nSeq = seqArray.shape[0]
        seqIxStack = [self.seqToIx(seq) for seq in seqArray]
        maxSeqLength = max(seq.shape[0] for seq in seqIxStack)

        ixTensor = torch.zeros((nSeq, maxSeqLength), dtype=torch.long)
        seqLenTensor = torch.zeros(nSeq, dtype=torch.long)
        for i, ix in enumerate(seqIxStack):
            ixTensor[i, :min(ix.shape[0], maxSeqLength)] = ix
            seqLenTensor[i] = ix.shape[0]

        return ixTensor, seqLenTensor


class VocabBuilder(object):
    def __init__(self, rootDir='.cache', vectorPath='vectors', tokenizerPath='tokenizer'):
        self.vectorPath = Path.cwd() / rootDir / vectorPath
        self.tokenizerPath = Path.cwd() / rootDir / tokenizerPath
        self.tokenizer = Tokenizer(Vocab())
        self.vectors = Vectors(shape=(41299, 300))

    def _countWords(self, sequences, tokenizer):
        self.tokenCounts = Counter()
        for seq in sequences:
            tokens = tokenizer(seq)
            for t in tokens:
                self.tokenCounts[t.text] += 1

    def fromDisk(self):
        self.tokenizer.from_disk(self.tokenizerPath)
        self.vectors.from_disk(self.vectorPath)

    def learnVocab(self, sequences, tokenizer, vectors, padToken='<pad>'):
        nlp = English()
        self._countWords(sequences, tokenizer=tokenizer)
        nlp.vocab = Vocab()
        nlp.vocab.set_vector(padToken, np.zeros(vectors.data.shape[1]))
        for word in self.tokenCounts:
            idx = tokenizer(word)[0].lex_id
            nlp.vocab.set_vector(word, vectors.data[idx])

        self.tokenizer = Tokenizer(
            nlp.vocab,
            rules={padToken: [{ORTH: padToken}]},
            prefix_search=nlp.tokenizer.prefix_search,
            suffix_search=nlp.tokenizer.suffix_search,
            token_match=nlp.tokenizer.token_match,
            infix_finditer=nlp.tokenizer.infix_finditer)
        self.vectors = nlp.vocab.vectors

    def toDisk(self, tokenizerPath=None, vectorPath=None):
        self.tokenizer.to_disk(tokenizerPath or self.tokenizerPath)
        self.vectors.to_disk(vectorPath or self.vectorPath)
