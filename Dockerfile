FROM tensorflow/tensorflow:2.7.1

WORKDIR /workdir

RUN yes | pip install --no-cache-dir fastapi==0.75.0

COPY ./app /workdir/app
COPY ./model /workdir/model

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]
