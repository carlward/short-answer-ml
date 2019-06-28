import copy
from pathlib import Path
import time

import torch

import torch.optim as optim
from torch.utils.data import DataLoader
from torch import nn

import spacy

from models import ParaLSTMModel
from preprocessing import MRPCDataset

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 512
MODELS_PATH = 'models'
WORD_EMBED_NAME = 'en_core_web_lg'
CLS_EPOCHS = 1000

DATA_DIR = 'data/MRPC/'
TRAIN_DIR = 'msr_paraphrase_train.tsv'
TEST_DIR = 'msr_paraphrase_test.tsv'
WARM_START = True


def trainCls(model, tokenizer, numEpochs=10, batch_size=BATCH_SIZE, lossFunc=nn.CosineEmbeddingLoss,
             optimizer=optim.Adam, scheduler=optim.lr_scheduler, dataDir=DATA_DIR, device=DEVICE):

    # Preprocess datasets
    trainDataLoader = DataLoader(
        MRPCDataset(dataDir, tokenizer=tokenizer, split='train'),
        batch_size=batch_size,
        shuffle=True,
        num_workers=4)
    trainSize = len(trainDataLoader.dataset)
    testLoader = DataLoader(
        MRPCDataset(dataDir, tokenizer=tokenizer, split='test'),
        batch_size=batch_size,
        shuffle=True,
        num_workers=4)
    testSize = len(testLoader.dataset)

    # Train/test loop
    bestAcc = 0
    startTime = time.time()
    bestWeights = copy.deepcopy(model.state_dict())
    for epoch in range(numEpochs):
        print('Epoch {}/{}'.format(epoch, numEpochs - 1))
        print('-' * 10)

        # Train
        trainLoss, trainCorrect = 0, 0
        model.train()

        for batch_left, batch_right, batchLabels in trainDataLoader:
            batch_left = batch_left.to(device)
            batch_right = batch_right.to(device)
            batchLabels = batchLabels.to(device)

            # Forward
            optimizer.zero_grad()
            output_left, output_right = model(batch_left, batch_right)
            cos_sim = model.predict(batch_left, batch_right)
            # print('Cos stats', cos_sim.mean(), cos_sim.min(), 'Values ', cos_sim[:5])
            preds = (cos_sim.sign() * cos_sim.abs().ceil()).type(torch.long)

            # Backward
            loss = lossFunc(output_left, output_right, batchLabels.type(torch.float32))
            loss.backward()
            optimizer.step()

            trainLoss += loss.item()
            trainCorrect += torch.sum((preds == batchLabels.data))

        # Eval
        evalLoss, evalCorrect = 0, 0
        model.eval()

        for batch_left, batch_right, batchLabels in testLoader:
            batch_left = batch_left.to(device)
            batch_right = batch_right.to(device)
            batchLabels = batchLabels.to(device)

            with torch.no_grad():
                optimizer.zero_grad()
                output_left, output_right = model(batch_left, batch_right)
                loss = lossFunc(output_left, output_right, batchLabels.type(torch.float32))
                cos_sim = model.predict(batch_left, batch_right)
                preds = (cos_sim.sign() * cos_sim.abs().ceil()).type(torch.long)

            evalLoss += loss.item()
            evalCorrect += torch.sum((preds == batchLabels.data))

        # Epoch Loss
        scheduler.step(evalLoss)
        epochLossTrain = trainLoss / trainSize
        epochAccTrain = trainCorrect.double() / trainSize
        epochLossEval = evalLoss / testSize
        epochAccEval = evalCorrect.double() / testSize
        print('Train| Loss: {:.4f} Acc: {:.4f}\t| Eval | Loss: {:.4f} Acc: {:.4f} EvalCorrect:{:d}'.format(
                epochLossTrain, epochAccTrain, epochLossEval, epochAccEval, evalCorrect))

        if epochAccEval > bestAcc:
            bestAcc = epochAccEval
            bestWeights = copy.deepcopy(model.state_dict())

    trainTime = time.time() - startTime
    print('Training complete in {:.0f}m {:.0f}s'.format(
        trainTime // 60, trainTime % 60))
    print('Best eval Acc: {:4f}'.format(bestAcc))

    # load best model weights
    model.load_state_dict(bestWeights)
    return model


if __name__ == "__main__":
    nlp = spacy.load(WORD_EMBED_NAME)
    embeddings = torch.Tensor(nlp.vocab.vectors.data)
    tokenizer = nlp.tokenizer

    model = ParaLSTMModel(embeddings, hiddenSize=141, dropProb=0.1)
    model = model.to(DEVICE)

    lossFunc = nn.CosineEmbeddingLoss(margin=0.05)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # optimizer = optim.SGD(model.parameters(), lr=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=100, factor=0.1)

    if WARM_START:
        bestWeights = torch.load(Path.cwd() / MODELS_PATH / 'clsParams.pt')
        model.load_state_dict(bestWeights)

    model = trainCls(
        model,
        tokenizer=tokenizer,
        numEpochs=CLS_EPOCHS,
        optimizer=optimizer,
        scheduler=scheduler,
        lossFunc=lossFunc)
    torch.save(model, Path.cwd() / MODELS_PATH / 'clsModel.pt')
    torch.save(model.state_dict(), Path.cwd() / MODELS_PATH / 'clsParams.pt')
