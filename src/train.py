import copy
import time
from argparse import ArgumentParser
from pathlib import Path

import spacy
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader

from models import BiLSTMModel, SeqClassifier
from preprocessing import (ASAPSASDataset, MRPCDataset, SNLIDataset,
                           loadSequences)
from utils import SeqPair, VocabBuilder

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def trainEpoch(model, dataLoader, optimizer, lossFunc, device=DEVICE):
    trainLoss, trainCorrect = 0, 0
    model.train()

    for batch in dataLoader:
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

    return trainLoss, trainCorrect


def evalEpoch(model, dataLoader, optimizer, lossFunc, device=DEVICE):
    evalLoss, evalCorrect = 0, 0
    model.eval()

    for batch in dataLoader:
        batch = SeqPair(*[t.to(device) for t in batch])

        with torch.no_grad():
            optimizer.zero_grad()
            output = model(batch)
            loss = lossFunc(output, batch.label)
            scores = model.predict(batch)
            preds = scores.argmax(dim=1)

        evalLoss += loss.item()
        evalCorrect += torch.sum((preds == batch.label.data))

    return evalLoss, evalCorrect, scores, preds


def train(params):
    print('=' * 89)
    print('Initializing...')

    # Initialize NLP tools
    vectorPath = Path(params.vector_cache_dir) / 'snli_1.0' / params.word_embeds
    vocab = VocabBuilder(Path(vectorPath))
    if params.rebuild_vocab or not (Path.is_file(vocab.tokenizerPath) and Path.is_dir(vocab.vectorPath)):
        print('No vocabulary found. Rebuilding from dataset')
        if not Path.is_dir(Path.cwd() / vectorPath):
            Path.mkdir(Path.cwd() / vectorPath, parents=True)
        nlp = spacy.load(params.word_embeds)
        fullTokenizer = nlp.tokenizer
        fullVectors = nlp.vocab.vectors

        # Combine all sequences in SNLI dataset
        sequences = loadSequences(
            Path.cwd() / params.data_dir / 'snli_1.0',
            filenames=('snli_1.0_train.txt', 'snli_1.0_test.txt', 'snli_1.0_dev.txt'),
            seq1Col='sentence1',
            seq2Col='sentence2',
            filterPred=lambda r: r['gold_label'] != '-',
            sep='\t')

        # Learn vocabulary from sequences
        vocab.learnVocab(sequences, fullTokenizer, fullVectors)
        vocab.toDisk()
    vocab.fromDisk()
    torch.save(vocab.tokenizer, Path.cwd() / params.model_dir / '{0}Tokenizer.pt'.format(params.word_embeds))

    # Preprocess datasets
    print('Preprocessing dataset...')
    datasetConsMap = {
        'snli_1.0': SNLIDataset,
        'asap-sas': ASAPSASDataset,
        'MRPC': MRPCDataset
    }
    datasetDir = Path.cwd() / params.data_dir / params.dataset
    datasetCons = datasetConsMap[params.dataset]
    trainDataLoader = DataLoader(
        datasetCons(datasetDir, tokenizer=vocab.tokenizer, split='train'),
        batch_size=params.batch_size,
        shuffle=True,
        num_workers=4)
    evalLoader = DataLoader(
        datasetCons(datasetDir, tokenizer=vocab.tokenizer, split='dev'),
        batch_size=params.batch_size,
        shuffle=True,
        num_workers=4)
    trainSize = len(trainDataLoader.dataset)
    evalSize = len(evalLoader.dataset)
    nClasses = trainDataLoader.dataset.nClasses

    # Model parameters
    modelName = 'encoder' if params.mode == 'train_encoder' else 'cls'
    if params.mode == 'train_encoder':
        model = BiLSTMModel(
            torch.Tensor(vocab.vectors.data),
            nClasses=nClasses,
            hiddenSizeEncoder=params.hidden_size_encoder,
            hiddenSizeCls=params.hidden_size_cls,
            layers=params.lstm_layers,
            dropProb=params.dropout)

    elif params.mode == 'train_cls':
        model = BiLSTMModel(
            torch.zeros(vocab.vectors.data.shape),
            nClasses=3,
            hiddenSizeEncoder=params.hidden_size_encoder,
            hiddenSizeCls=params.hidden_size_cls,
            layers=params.lstm_layers,
            dropProb=0.0)
        bestWeights = torch.load(Path.cwd() / params.model_dir / 'encoderParams.pt', map_location=DEVICE)
        model.load_state_dict(bestWeights)
        model.classifier = SeqClassifier(
            params.hidden_size_encoder*8,
            hiddenSize=params.hidden_size_cls,
            nClasses=nClasses,
            dropProb=params.dropout)
        model.freezeEncoder()
    model = model.to(DEVICE)

    # Training parameters
    lossFunc = nn.CrossEntropyLoss()
    optimizerCons = optim.SGD if params.optimizer == 'sgd' else optim.Adam
    optimizer = optimizerCons(model.parameters(), lr=params.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=params.lr_patience, factor=0.2)

    # Train/test loop
    bestAcc = 0
    bestLoss = 1E9
    startTime = time.time()
    bestWeights = copy.deepcopy(model.state_dict())
    stopTraining = False
    stopCount = 0

    print('-' * 89)
    print('Beginning training...')
    for epoch in range(1, params.epochs+1):
        epochStartTime = time.time()
        trainLoss, trainCorrect = trainEpoch(model, trainDataLoader, optimizer, lossFunc)
        evalLoss, evalCorrect, scores, preds = evalEpoch(model, evalLoader, optimizer, lossFunc)

        # Epoch Loss
        epochLossTrain = trainLoss / trainSize
        epochAccTrain = trainCorrect.double() / trainSize
        epochLossEval = evalLoss / evalSize
        epochAccEval = evalCorrect.double() / evalSize
        print('[Epoch:\t{}/{}] | time {:5.2f}s | train loss: {:.4f} acc: {:.4f}\t'
              '| eval loss: {:.4f} acc: {:.4f} nCorrect: {:d}'.format(epoch, params.epochs, time.time()-epochStartTime,
                                                                      epochLossTrain, epochAccTrain, epochLossEval,
                                                                      epochAccEval, evalCorrect))

        # Update learning rate
        scheduler.step(epochLossEval)
        if epochLossEval < bestLoss:
            bestLoss = epochLossEval
            stopCount = 0
        elif params.optimizer == 'sgd':
            optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] / 5
            if optimizer.param_groups[0]['lr'] < 1e-5:
                stopTraining = True
        else:
            stopCount += 1
            if stopCount >= 4:
                stopTraining = True
        optimizer.param_groups[0]['lr'] *= (1 if params.optimizer != 'sgd' else 0.99)

        # Save state
        if epochAccEval > bestAcc:
            bestAcc = epochAccEval
            bestWeights = copy.deepcopy(model.state_dict())
            torch.save(bestWeights, Path.cwd() / params.model_dir / '{0}Params.pt'.format(modelName))
            torch.save(model, Path.cwd() / params.model_dir / '{0}Model.pt'.format(modelName))

        # Check for early stopping
        if stopTraining:
            break

    trainTime = time.time() - startTime
    print('Training complete in {:.0f}m {:.0f}s'.format(
        trainTime // 60, trainTime % 60))
    print('Best eval Acc: {:4f}'.format(bestAcc))

    # Load best model weights
    model.load_state_dict(bestWeights)
    return model, vocab.tokenizer


def test(model, tokenizer, params):
    """Output test set results"""
    datasetConsMap = {
        'snli_1.0': SNLIDataset,
        'asap-sas': ASAPSASDataset,
        'MRPC': MRPCDataset
    }
    datasetCons = datasetConsMap[params.dataset]
    datasetDir = Path.cwd() / params.data_dir / params.dataset
    testLoader = DataLoader(
            datasetCons(datasetDir, tokenizer=tokenizer, split='test'),
            batch_size=params.batch_size,
            shuffle=True,
            num_workers=4)
    testSize = len(testLoader.dataset)

    nCorrect = 0
    model.eval()
    for batch in testLoader:
        batch = SeqPair(*[t.to(DEVICE) for t in batch])
        with torch.no_grad():
            preds = model.predict(batch).argmax(dim=1)
        nCorrect += torch.sum((preds == batch.label.data))

    acc = 100 * nCorrect.double() / testSize
    return acc


def parseArgs():
    parser = ArgumentParser(description='Model training params')

    parser.add_argument(
        '--mode',
        type=str,
        choices=['train_encoder', 'train_cls'],
        required=True
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=['snli_1.0', 'asap-sas', 'MRPC']
    )
    parser.add_argument(
        "--rebuild_vocab",
        action='store_true',
        help='Re-learn vocabulary/vectors from dataset if already present'
    )

    # Training params
    parser.add_argument(
        '--epochs',
        type=int,
        default=20
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64
    )
    parser.add_argument(
        '--optimizer',
        type=str,
        choices=['adam', 'sgd'],
        default='adam',
    )

    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.0025
    )
    parser.add_argument(
        '--lr_decay',
        type=float,
        help='only applies for sgd',
        default=0.99
    )
    parser.add_argument(
        '--lr_patience',
        type=int,
        default=1,
        help='reduce lr after n epochs on plateau'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )

    # Model params
    parser.add_argument(
        '--word_embeds',
        type=str,
        choices=['crawl-300d-2M', 'glove.840B.300d'],
        default='crawl-300d-2M'
    )
    parser.add_argument(
        '--hidden_size_encoder',
        type=int,
        default=2048
    )
    parser.add_argument(
        '--lstm_layers',
        type=int,
        default=1
    )
    parser.add_argument(
        '--hidden_size_cls',
        type=int,
        default=512
    )
    parser.add_argument(
        '--n_classes',
        type=int,
        default=3
    )
    parser.add_argument(
        '--dropout',
        type=float,
        default=0.1
    )

    # Path args
    parser.add_argument(
        '--data_dir',
        type=str,
        default='data'
    )
    parser.add_argument(
        '--model_dir',
        type=str,
        default='models'
    )
    parser.add_argument(
        '--vector_cache_dir',
        type=str,
        default='.cache'
    )
    return parser.parse_args()


def main():
    params = parseArgs()
    torch.manual_seed(params.seed)
    torch.cuda.manual_seed(params.seed)

    model, tokenizer = train(params)
    testAcc = test(model, tokenizer, params)
    print('-' * 89)
    print('Test Accuracy:\t{:3.3f}'.format(testAcc))

    # # Persist best model
    modelName = 'encoder' if params.mode == 'train_encoder' else 'cls'
    torch.save(model, Path.cwd() / params.model_dir / '{0}Model.pt'.format(modelName))
    torch.save(model.state_dict(), Path.cwd() / params.model_dir / '{0}Params.pt'.format(modelName))
    torch.save(tokenizer, Path.cwd() / params.model_dir / '{0}Tokenizer.pt'.format(params.word_embeds))


if __name__ == "__main__":
    main()
