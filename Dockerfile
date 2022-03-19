FROM tensorflow/tensorflow:2.7.1

ENV MODEL_NAME "mobilenet-20220312-062819"
ENV CLASS_NAME "class_names-20220312-062819.z"

WORKDIR /workdir

COPY ./docker-requirements.txt /workdir/docker-requirements.txt

RUN pip install --no-cache-dir -r docker-requirements.txt

COPY ./app /workdir/app

COPY ./model/${MODEL_NAME} /workdir/model/${MODEL_NAME}
COPY ./model/${CLASS_NAME} /workdir/model/${CLASS_NAME}

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]
