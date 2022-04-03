FROM tensorflow/tensorflow:2.7.1

ARG MODEL_NAME
ARG CLASS_NAME

WORKDIR /workdir

COPY ./docker-requirements.txt /workdir/docker-requirements.txt

RUN pip install --no-cache-dir -r docker-requirements.txt

COPY ./app /workdir/app

COPY ./${MODEL_NAME} /workdir/app/model/tensorflow.h5
COPY ./${CLASS_NAME} /workdir/app/model/class_names.z

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]
