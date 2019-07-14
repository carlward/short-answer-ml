import copy
import time
from collections import namedtuple
from pathlib import Path

import spacy
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models import BiLSTMModel
from preprocessing import SNLIDataset
# from models import MalLSTMModel
# from preprocessing import MalLSTMModel

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 128
MODELS_PATH = 'models'
WORD_EMBED_NAME = 'en_vectors_web_lg'
CLS_EPOCHS = 20

DATA_DIR = 'data/snli_1.0/'
WARM_START = False

SeqPair = namedtuple('SeqPair', ['leftSeq', 'leftLen', 'rightSeq', 'rightLen', 'label'])


def trainSNLI(model, tokenizer, numEpochs=10, batchSize=BATCH_SIZE, lossFunc=nn.CosineEmbeddingLoss,
              optimizer=optim.Adam, scheduler=optim.lr_scheduler, dataDir=DATA_DIR, device=DEVICE):

    # Preprocess datasets
    trainDataLoader = DataLoader(
        SNLIDataset(dataDir, tokenizer=tokenizer, split='train'),
        batch_size=batchSize,
        shuffle=True,
        num_workers=4)
    evalLoader = DataLoader(
        SNLIDataset(dataDir, tokenizer=tokenizer, split='dev'),
        batch_size=batchSize,
        shuffle=True,
        num_workers=4)
    trainSize = len(trainDataLoader.dataset)
    evalSize = len(evalLoader.dataset)

    # Train/test loop
    writer = SummaryWriter()
    bestAcc = 0
    startTime = time.time()
    bestWeights = copy.deepcopy(model.state_dict())
    for epoch in range(1, numEpochs+1):

        # Train
        trainLoss, trainCorrect = 0, 0
        model.train()

        for batch in trainDataLoader:
            batch = SeqPair(*[t.to(device) for t in batch])

            # Forward
            optimizer.zero_grad()
            output = model(batch)
            scores = model.predict(batch)
            preds = scores.argmax(dim=1)

            # Backward
            loss = lossFunc(output, batch.label)
            loss.backward()
            optimizer.step()

            trainLoss += loss.item()
            trainCorrect += torch.sum((preds == batch.label.data))

        # Eval
        evalLoss, evalCorrect = 0, 0
        model.eval()

        for batch in evalLoader:
            batch = SeqPair(*[t.to(device) for t in batch])

            with torch.no_grad():
                optimizer.zero_grad()
                output = model(batch)
                loss = lossFunc(output, batch.label)
                scores = model.predict(batch)
                preds = scores.argmax(dim=1)

            evalLoss += loss.item()
            evalCorrect += torch.sum((preds == batch.label.data))

        # Epoch Loss
        scheduler.step(evalLoss)
        epochLossTrain = trainLoss / trainSize
        epochAccTrain = trainCorrect.double() / trainSize
        epochLossEval = evalLoss / evalSize
        epochAccEval = evalCorrect.double() / evalSize
        print('[Epoch:\t{}/{}] | train loss: {:.4f} acc: {:.4f}\t| eval loss: {:.4f} acc: {:.4f} nCorrect: {:d}'.format(
            epoch, numEpochs, epochLossTrain, epochAccTrain, epochLossEval, epochAccEval, evalCorrect))

        # Save state
        if epochAccEval > bestAcc:
            bestAcc = epochAccEval
            bestWeights = copy.deepcopy(model.state_dict())
            torch.save(bestWeights, Path.cwd() / MODELS_PATH / 'clsParams.pt')

        # Write state to tensorboard
        writer.add_scalar('trainLoss', trainLoss, global_step=epoch)
        writer.add_scalar('trainAcc', epochAccTrain, global_step=epoch)
        writer.add_scalar('evalLoss', evalLoss, global_step=epoch)
        writer.add_scalar('evalAcc', epochAccEval, global_step=epoch)
        writer.add_histogram('preds', scores, global_step=epoch)

    trainTime = time.time() - startTime
    print('Training complete in {:.0f}m {:.0f}s'.format(
        trainTime // 60, trainTime % 60))
    print('Best eval Acc: {:4f}'.format(bestAcc))

    # load best model weights
    model.load_state_dict(bestWeights)
    return model


def testSNLI(model, tokenizer, batchSize=1024, dataDir=DATA_DIR, device=DEVICE):
    """Output Results"""
    testLoader = DataLoader(
            SNLIDataset(dataDir, tokenizer=tokenizer, split='test'),
            batch_size=batchSize,
            shuffle=True,
            num_workers=4)
    testSize = len(testLoader.dataset)

    nCorrect = 0
    model.eval()
    for batch in testLoader:
        batch = SeqPair(*[t.to(device) for t in batch])
        with torch.no_grad():
            preds = model.predict(batch).argmax(dim=1)
        nCorrect += torch.sum((preds == batch.label.data))

    acc = 100 * nCorrect.double() / testSize
    return acc


def main():
    # Initialize
    nlp = spacy.load(WORD_EMBED_NAME)
    embeddings = torch.Tensor(nlp.vocab.vectors.data)
    tokenizer = nlp.tokenizer

    # Model parameters
    model = BiLSTMModel(
        embeddings,
        nClasses=3,
        hiddenSizeEncoder=2048,
        hiddenSizeCls=512,
        layers=1,
        dropProb=0.1)
    model = model.to(DEVICE)
    if WARM_START:
        bestWeights = torch.load(Path.cwd() / MODELS_PATH / 'clsParamsPPBest.pt', map_location=DEVICE)
        model.load_state_dict(bestWeights)

    # Training parameters
    lossFunc = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00025)
    # optimizer = optim.SGD(model.parameters(), lr=0.05)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1, factor=0.2)

    # Train
    model = trainSNLI(
        model,
        tokenizer=tokenizer,
        numEpochs=CLS_EPOCHS,
        optimizer=optimizer,
        scheduler=scheduler,
        lossFunc=lossFunc)

    # Test and persist
    torch.save(model, Path.cwd() / MODELS_PATH / 'clsModel.pt')
    torch.save(model.state_dict(), Path.cwd() / MODELS_PATH / 'clsParams.pt')
    print('Test Accuracy:\t{:3.3f}'.format(testSNLI(model)))


if __name__ == "__main__":
    main()
