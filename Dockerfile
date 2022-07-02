FROM tensorflow/tensorflow:2.8.1

ARG MODEL_URL
ARG CLASS_NAMES_URL

WORKDIR /workdir

COPY ./docker-requirements.txt /workdir/docker-requirements.txt

RUN pip install --no-cache-dir -r docker-requirements.txt

COPY ./app /workdir/app

ADD ${MODEL_URL} /workdir/app/model/tensorflow.h5
ADD ${CLASS_NAMES_URL} /workdir/app/model/class_names.z

CMD ["uvicorn", "app:app", "--proxy-headers", "--host", "0.0.0.0", "--port", "80"]
