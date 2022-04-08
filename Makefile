# Variables
MODEL_NAME=deploy/mobilenet-20220326-151658.h5
CLASS_NAME=deploy/class_names-20220326-151658.z
IMAGE_NAME=skripsi:latest

.PHONY: run docker-build docker-run

.DEFAULT_GOAL := docker-run

run:
	MODEL_NAME="../../${MODEL_NAME}" CLASS_NAME="../../${CLASS_NAME}" uvicorn app:app --port 8080 --reload

docker-build:
	docker build --build-arg MODEL_NAME="${MODEL_NAME}" --build-arg CLASS_NAME="${CLASS_NAME}" -t ${IMAGE_NAME} .

docker-run:
	-docker rm -f skripsi
	docker run --name skripsi -d -p '8080:80' ${IMAGE_NAME}
