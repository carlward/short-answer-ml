import torch
from torch import nn
from torch.nn import functional as F


class ParaLSTMModel(nn.Module):
    def __init__(self, embeddings, hiddenSize, dropProb=0.05):
        super(ParaLSTMModel, self).__init__()
        vocabSize, embeddingSize = embeddings.shape
        self.embeddings = self._setupEmbeddings(embeddings, vocabSize, embeddingSize)
        self.biLSTM = nn.LSTM(embeddingSize, hiddenSize, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(p=dropProb)

    def _setupEmbeddings(self, embeddings, vocabSize, embeddingSize):
        embeddingLayer = nn.Embedding(vocabSize, embeddingSize)
        embeddingLayer.weight.data = embeddings
        for param in embeddingLayer.parameters():
            param.requires_grad = False

        return embeddingLayer

    def forward(self, leftSeq, rightSeq):
        leftEmbeds = self.embeddings(F.pad(leftSeq, pad=(1, 0, 0, 0)))
        rightEmbeds = self.embeddings(F.pad(rightSeq, pad=(1, 0, 0, 0)))
        left, em = self.biLSTM(leftEmbeds)
        right, _ = self.biLSTM(rightEmbeds)
        pooledLeft = left.max(dim=1).values
        pooledRight = right.max(dim=1).values

        return self.dropout(pooledLeft), self.dropout(pooledRight)

    def predict(self, leftSeq, rightSeq):
        with torch.no_grad():
            outputLeft, outputRight = self.forward(leftSeq, rightSeq)
            return F.cosine_similarity(outputLeft, outputRight)
