from common_code.config import get_settings
from common_code.logger.logger import get_logger, Logger
from common_code.service.models import Service
from common_code.service.enums import ServiceStatus
from common_code.common.enums import FieldDescriptionType, ExecutionUnitTagName, ExecutionUnitTagAcronym
from common_code.common.models import FieldDescription, ExecutionUnitTag
from common_code.tasks.models import TaskData
# Imports required by the service's model
import io
import json
import numpy as np
from PIL import Image
from retinaface import RetinaFace

settings = get_settings()
api_description = """
This service detects faces in images and returns the coordinates of the bounding boxes.
"""
api_summary = """
Detects faces in images and returns the coordinates of the bounding boxes.
"""

api_title = "Face Detection API."
version = "1.0.0"


class MyService(Service):
    """
    Face detection service model
    """

    # Any additional fields must be excluded for Pydantic to work
    _model: object
    _logger: Logger

    def __init__(self):
        super().__init__(
            name="Face Detection",
            slug="face-detection",
            url=settings.service_url,
            summary=api_summary,
            description=api_description,
            status=ServiceStatus.AVAILABLE,
            data_in_fields=[
                FieldDescription(name="image", type=[FieldDescriptionType.IMAGE_PNG, FieldDescriptionType.IMAGE_JPEG]),
            ],
            data_out_fields=[
                FieldDescription(name="result", type=[FieldDescriptionType.APPLICATION_JSON]),
            ],
            tags=[
                ExecutionUnitTag(
                    name=ExecutionUnitTagName.IMAGE_RECOGNITION,
                    acronym=ExecutionUnitTagAcronym.IMAGE_RECOGNITION
                ),
            ],
            has_ai=True,
            docs_url="https://docs.swiss-ai-center.ch/reference/services/face-detection/",
        )
        self._logger = get_logger(settings)

    def process(self, data):
        # Get raw image data
        raw = data["image"].data
        buff = io.BytesIO(raw)
        img_pil = Image.open(buff)
        img = np.array(img_pil)

        faces = RetinaFace.detect_faces(img)
        self._logger.info(f"Found {len(faces)} faces")

        # https://stackoverflow.com/a/57915246
        class NpEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super(NpEncoder, self).default(obj)

        if type(faces) is dict:
            faces_areas = {'areas': [[facial_area for facial_area in f[1]['facial_area']] for f in faces.items()]}
        else:
            faces_areas = {'areas': []}

        task_data = TaskData(
            data=json.dumps(
                faces_areas,
                cls=NpEncoder,
            ),
            type=FieldDescriptionType.APPLICATION_JSON
        )

        return {
            "result": task_data
        }
