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
        seqSorted = seq[sortedIdx, 0:seqLen.max()]  # Trim to longest sequence length to reduce padding

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
        self.nClasses = nClasses
        self.embeddingSize = embeddingSize
        self.hiddenSize = hiddenSize
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
    def __init__(self, embeddings=None, nClasses=2, hiddenSizeEncoder=141,
                 hiddenSizeCls=128, layers=2, dropProb=0.05, freezeEmbeds=False):
        super(BiLSTMModel, self).__init__()
        vocabSize, embeddingSize = (1999995, 300) if embeddings is None else embeddings.shape
        self.embeddingSize = embeddingSize
        self.vocabSize = vocabSize
        self.hiddenSizeCls = hiddenSizeCls
        self.hiddenSizeEncoder = hiddenSizeEncoder
        self.embeddings = self._setupEmbeddings(embeddings, freeze=freezeEmbeds)
        self.nClasses = nClasses

        self.encoder = BiLSTMEncoder(
            self.embeddingSize,
            hiddenSize=self.hiddenSizeEncoder,
            layers=layers,
            dropProb=dropProb)
        self.classifier = SeqClassifier(
            self.hiddenSizeEncoder*8,
            hiddenSize=self.hiddenSizeCls,
            nClasses=self.nClasses)

    def _setupEmbeddings(self, embeddings, freeze=False):
        embeddingLayer = nn.Embedding(self.vocabSize, self.embeddingSize)
        if embeddings is not None:
            embeddingLayer.from_pretrained(embeddings, freeze=freeze)
        return embeddingLayer

    def freezeEncoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.embeddings.parameters():
            param.requires_grad = False

    def forward(self, inputs):
        leftFeat = self.encoder(self.embeddings, inputs.leftSeq, inputs.leftLen)
        rightFeat = self.encoder(self.embeddings, inputs.rightSeq, inputs.rightLen)
        return self.classifier(leftFeat, rightFeat)

    def predict(self, inputs):
        with torch.no_grad():
            return F.softmax(self.forward(inputs), dim=1)
