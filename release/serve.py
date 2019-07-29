import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from spacy.tokenizer import Tokenizer
from spacy.vocab import Vocab
from torch.utils.data import DataLoader, TensorDataset

from models import BiLSTMModel
from utils import Seq2IdxTransformer, SeqPair

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

MODEL_NAME = 'sagModel'
TOKENIZER = 'tokenizer'
BATCH_SIZE = 1024

JSON_CONTENT_TYPE = 'application/json'
CSV_CONTENT_TYPE = 'text/csv'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def model_fn(model_dir):
    model = BiLSTMModel(
        torch.zeros((41299, 300)),
        nClasses=4,
        hiddenSizeEncoder=2048,
        hiddenSizeCls=512,
        layers=1,
        dropProb=0.0)
    weights = torch.load(Path(model_dir) / '{}.pt'.format(MODEL_NAME), map_location=DEVICE)
    model.load_state_dict(weights)
    model.to(DEVICE)
    model.eval()

    tokenizer = Tokenizer(Vocab())
    tokenizer.from_disk(Path(model_dir) / '{}'.format(TOKENIZER))
    return {'model': model, 'tokenizer': tokenizer}


def input_fn(inputData, contentType=JSON_CONTENT_TYPE):
    logger.info('Deserializing the input data.')
    if contentType == JSON_CONTENT_TYPE:
        return json.loads(inputData)
    raise Exception('Requested unsupported ContentType in content_type: {}'.format(contentType))


def predict_fn(inputObject, modelDict):
    logger.info("Calling model")
    model = modelDict['model']
    tokenizer = modelDict['tokenizer']
    transformer = Seq2IdxTransformer(tokenizer)
    nClasses = model.nClasses

    # Preprocess input
    seq, refSeq = inputObject['seq'], inputObject['refSeq']
    seq = pd.Series(seq)
    refSeq = pd.Series(refSeq)

    seqIdx, seqLen = transformer.seqsToTensor(seq)
    refSeqIdx, refSeqLen = transformer.seqsToTensor(refSeq)
    dummyLabels = torch.zeros_like(seqIdx, dtype=torch.long)

    # Inference
    tensors = (refSeqIdx, refSeqLen, seqIdx, seqLen, dummyLabels)
    model.eval()
    with torch.no_grad():
        batch = SeqPair(*[t.to(DEVICE) for t in tensors])
        probs = model.predict(batch)
        preds = probs.argmax(dim=1).cpu().numpy()

    #  Format output
    preds += 1  # Use 1 indexed scoring system
    probDf = pd.DataFrame(
        probs.cpu().numpy().round(5),
        columns=['probScore{}'.format(n) for n in range(1, nClasses+1)]
    )
    results = [
        {'estimScore': int(s), 'scoreProbs': p}
        for s, p in zip(preds, probDf.to_dict(orient='records'))
    ]
    return {'data': results}


def output_fn(prediction, accept=JSON_CONTENT_TYPE):
    logger.info('Serializing the generated output.')
    if accept == JSON_CONTENT_TYPE:
        return json.dumps(prediction), accept
    raise Exception('Requested unsupported ContentType in Accept: {}'.format(accept))
