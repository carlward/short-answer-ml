# Short Answer Autograder

A model to automatically assess short answers given a reference answer. Answers are graded on a 4 point scale by default. Built with `PyTorch` and `spaCy`. The encoder portion of model is based on the [InferSent](https://github.com/facebookresearch/InferSent) architecture and the classifier head is trained on the [asap-sas dataset](https://www.kaggle.com/c/asap-sas). A demo of model in action is [available here](https://carlward.github.io/short-answer-web/).

## Setup

- Install required packages (Python 3.7) `pip install -r requirements.txt`
- Download datasets and place enclosing folder in data dir
    - https://nlp.stanford.edu/projects/snli/snli_1.0.zip
    - https://www.kaggle.com/c/asap-sas
- Download and serialize word vectors:

    `wget https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip`

    `python -m spacy init-model en crawl-300d-2M.vec.zip --vectors-loc crawl-300d-2M.vec.zip`

## Training

Train the encoder first on the `snli_1.0` dataset. Then transfer learn a new classifier on the `asap-sas` dataset or alternative. Run `python src/train.py --help` to see available options.

### Encoder training example

```bash
python src/train.py \
    --mode train_encoder \
    --dataset snli_1.0 \
    --optimizer sgd \
    --learning_rate 0.10 \
```

### Classifier training example

```bash
python src/train.py \
    --mode train_cls \
    --dataset asap-sas \
    --dropout 0.1 \
    --epochs 40 \
```

### Deploy

Deploy to AWS Sagemaker​ with `./deploy.sh`. `SageMakerRole` IAM role must be defined first.
