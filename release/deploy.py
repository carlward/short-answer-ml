import tarfile
from pathlib import Path

import sagemaker
from sagemaker.pytorch import PyTorchModel

from predictors import ShortAnswerPredictor

MODEL_DIR = 'models'
MODEL_NAME = 'sagModel'
PARAM_PATH = 'clsParams.pt'
TOKENIZER = 'tokenizer'
INSTANCE_TYPE = 'ml.m4.xlarge'
VERSION = 'v1'


def uploadModel(session, modelPath=PARAM_PATH, tokenizerPath=TOKENIZER,
                modelFilename=MODEL_NAME, tokenizerFilename='tokenizer', resourceName='model'):
    pathOut = Path.cwd() / MODEL_DIR / 'model.tar.gz'

    # Compress
    with tarfile.open(pathOut, 'w:gz') as f:
        t = tarfile.TarInfo('models')
        t.type = tarfile.DIRTYPE
        f.addfile(t)
        f.add(Path.cwd() / MODEL_DIR / modelPath, arcname='{}.pt'.format(modelFilename))
        f.add(Path.cwd() / MODEL_DIR / tokenizerPath, arcname='{}'.format(tokenizerFilename))

    # Upload to S3
    return session.upload_data(
        path=str(pathOut),
        bucket=session.default_bucket(),
        key_prefix='sagemaker/{}-{}'.format(resourceName, VERSION)
    )


def deploy():
    session = sagemaker.Session()
    modelArtifacts = uploadModel(
        session,
        modelPath=PARAM_PATH,
        tokenizerPath=TOKENIZER,
        resourceName=MODEL_NAME)

    model = PyTorchModel(
        model_data=modelArtifacts,
        name='{}-{}'.format(MODEL_NAME, VERSION),
        role='SageMakerRole',  # Needs to defined beforehand
        framework_version='1.1.0',
        entry_point='serve.py',
        source_dir='release',
        predictor_cls=ShortAnswerPredictor)
    model.deploy(
        initial_instance_count=1,
        instance_type=INSTANCE_TYPE)


if __name__ == "__main__":
    deploy()
