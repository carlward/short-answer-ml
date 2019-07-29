import sagemaker
from sagemaker.predictor import json_deserializer, json_serializer


class ShortAnswerPredictor(sagemaker.RealTimePredictor):
    def __init__(self, endpoint_name, session):
        super().__init__(
            endpoint_name,
            sagemaker_session=session,
            serializer=json_serializer,
            deserializer=json_deserializer,
            content_type='json')
