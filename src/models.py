import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


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

    def forward(self, inputs):
        # Sort batch
        leftLens, sorted_idx_left = inputs.seq1_len.sort(0, descending=True)
        leftSeq = inputs.seq1[sorted_idx_left]
        rightLens, sorted_idx_right = inputs.seq2_len.sort(0, descending=True)
        rightSeq = inputs.seq2[sorted_idx_right]

        # Embeddings
        leftEmbeds = self.embeddings(leftSeq)
        rightEmbeds = self.embeddings(rightSeq)

        # BiLSTM
        leftPacked, _ = self.biLSTM(pack_padded_sequence(leftEmbeds, leftLens, batch_first=True))
        rightPacked, _ = self.biLSTM(pack_padded_sequence(rightEmbeds, rightLens, batch_first=True))

        total_length = max(leftLens[0], rightLens[0])  # Longest seq length overall
        left, _ = pad_packed_sequence(leftPacked, batch_first=True, total_length=total_length)
        right, _ = pad_packed_sequence(rightPacked, batch_first=True, total_length=total_length)

        # Max pooling
        pooledLeft = F.adaptive_max_pool1d(left.permute(0, 1, 2), 1).view(left.size(0), -1)
        pooledRight = F.adaptive_max_pool1d(right.permute(0, 1, 2), 1).view(right.size(0), -1)

        # Unsort Output
        _, idx_left = sorted_idx_left.sort(0)
        _, idx_right = sorted_idx_right.sort(0)

        return self.dropout(pooledLeft[idx_left]), self.dropout(pooledRight[idx_right])

    def predict(self, inputs):
        with torch.no_grad():
            outputLeft, outputRight = self.forward(inputs)
            return F.cosine_similarity(outputLeft, outputRight)
