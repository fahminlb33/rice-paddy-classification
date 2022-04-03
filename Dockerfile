FROM tensorflow/tensorflow:2.7.1

ENV MODEL_NAME "deploy/mobilenet-20220326-173030.h5"
ENV CLASS_NAME "deploy/class_names-20220326-173030.z"

WORKDIR /workdir

COPY ./docker-requirements.txt /workdir/docker-requirements.txt

RUN pip install --no-cache-dir -r docker-requirements.txt

COPY ./app /workdir/app

COPY ./${MODEL_NAME} /workdir/model/${MODEL_NAME}
COPY ./${CLASS_NAME} /workdir/model/${CLASS_NAME}

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]
