import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BiLSTMEncoder(nn.Module):
    def __init__(self, embeddingSize, hiddenSize, layers=1, dropProb=0.05):
        super(BiLSTMEncoder, self).__init__()
        self.biLSTM = nn.LSTM(
            embeddingSize,
            hiddenSize,
            num_layers=layers,
            batch_first=True,
            dropout=dropProb if layers > 1 else 0,
            bidirectional=True)

    def forward(self, embeddings, seq, seqLen):
        # Sort batch
        lens, sortedIdx = seqLen.sort(0, descending=True)
        seqSorted = seq[sortedIdx]

        # LSTM Features
        embeds = embeddings(seqSorted)
        packed, _ = self.biLSTM(pack_padded_sequence(embeds, lens, batch_first=True))
        unPacked, _ = pad_packed_sequence(packed, batch_first=True)
        pooled = F.adaptive_max_pool1d(unPacked.permute(0, 2, 1), 1).view(unPacked.size(0), -1)

        # Unsort Output
        _, idx = sortedIdx.sort(0)
        return pooled[idx]


class SeqClassifier(nn.Module):
    def __init__(self, embeddingSize, hiddenSize, nClasses, dropProb=0.05):
        super(SeqClassifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Dropout(p=dropProb),
            nn.Linear(embeddingSize, hiddenSize),
            nn.ReLU(),
            nn.Dropout(p=dropProb),
            nn.Linear(hiddenSize, hiddenSize),
            nn.ReLU(),
            nn.Linear(hiddenSize, nClasses)
        )

    def forward(self, leftFeat, rightFeat):
        combinedFeat = torch.cat([leftFeat, rightFeat, (leftFeat - rightFeat).abs(), leftFeat*rightFeat], 1)
        return self.fc(combinedFeat)


class BiLSTMModel(nn.Module):
    def __init__(self, embeddings, nClasses=2, hiddenSizeEncoder=141,
                 hiddenSizeCls=128, layers=2, dropProb=0.05):
        super(BiLSTMModel, self).__init__()

        vocabSize, embeddingSize = embeddings.shape
        self.embeddings = self._setupEmbeddings(embeddings, vocabSize, embeddingSize)
        self.encoder = BiLSTMEncoder(
            embeddingSize,
            hiddenSize=hiddenSizeEncoder,
            layers=layers,
            dropProb=dropProb)
        self.classifer = SeqClassifier(hiddenSizeEncoder*8, hiddenSize=hiddenSizeCls, nClasses=nClasses)

    def _setupEmbeddings(self, embeddings, vocabSize, embeddingSize):
        embeddingLayer = nn.Embedding(vocabSize, embeddingSize)
        embeddingLayer.from_pretrained(embeddings, freeze=True)
        return embeddingLayer

    def forward(self, inputs):
        leftFeat = self.encoder(self.embeddings, inputs.leftSeq, inputs.leftLen)
        rightFeat = self.encoder(self.embeddings, inputs.rightSeq, inputs.rightLen)
        return self.classifer(leftFeat, rightFeat)

    def predict(self, inputs):
        with torch.no_grad():
            return F.softmax(self.forward(inputs), dim=1)


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
