import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class MalLSTMModel(nn.Module):
    def __init__(self, embeddings, hiddenSize, layers=1, dropProb=0.05):
        super(MalLSTMModel, self).__init__()
        vocabSize, embeddingSize = embeddings.shape
        self.embeddings = self._setupEmbeddings(embeddings, vocabSize, embeddingSize)
        self.biLSTM = nn.LSTM(embeddingSize, hiddenSize, num_layers=layers, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(p=dropProb)

    def _setupEmbeddings(self, embeddings, vocabSize, embeddingSize):
        embeddingLayer = nn.Embedding(vocabSize, embeddingSize)
        embeddingLayer.from_pretrained(embeddings, freeze=True)
        return embeddingLayer

    def _expManDistance(self, h1, h2):
        return torch.exp(-(h1-h2).abs().sum(dim=1))

    def lstmFeat(self, seq, seqLen, maxSeqLen=None):
        # Sort batch
        lens, sortedIdx = seqLen.sort(0, descending=True)
        seqSorted = seq[sortedIdx]

        # LSTM Features
        embeds = self.embeddings(seqSorted)
        packed, _ = self.biLSTM(pack_padded_sequence(embeds, lens, batch_first=True))
        unPacked, _ = pad_packed_sequence(packed, batch_first=True, total_length=maxSeqLen)
        pooled = F.adaptive_max_pool1d(unPacked.permute(0, 1, 2), 1).view(unPacked.size(0), -1)

        # Unsort Output
        _, idx = sortedIdx.sort(0)
        return self.dropout(pooled[idx])

    def forward(self, inputs):
        maxLen = max(inputs.leftLen.max(), inputs.rightLen.max())  # Longest seq length overall
        leftFeat = self.lstmFeat(inputs.leftSeq, inputs.leftLen, maxSeqLen=maxLen)
        rightFeat = self.lstmFeat(inputs.rightSeq, inputs.rightLen, maxSeqLen=maxLen)

        return self._expManDistance(leftFeat, rightFeat)

    def predict(self, inputs):
        with torch.no_grad():
            return self.forward(inputs)
