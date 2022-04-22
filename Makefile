# Variables
TF_LOG_LEVEL=2
MODEL_NAME=deploy/mobilenet-20220326-151658.h5
CLASS_NAME=deploy/class_names-20220326-151658.z

IMAGE_REPO=fahminlb33
IMAGE_TAG=latest
IMAGE_NAME=skripsi
IMAGE_FULLNAME=${IMAGE_REPO}/${IMAGE_NAME}:${IMAGE_TAG}

# Directives
.PHONY: run debug docker-build docker-run docker-stop docker-rm

.DEFAULT_GOAL := docker-run

# Commands
run:
	MODEL_NAME="../../${MODEL_NAME}" \
	CLASS_NAME="../../${CLASS_NAME}" \
	TF_CPP_MIN_LOG_LEVEL=${TF_LOG_LEVEL} \
	uvicorn app:app --port 8080

debug:
	MODEL_NAME="../../${MODEL_NAME}" \
	CLASS_NAME="../../${CLASS_NAME}" \
	TF_CPP_MIN_LOG_LEVEL=${TF_LOG_LEVEL} \
	uvicorn app:app --port 8080 --reload

docker-build:
	docker build --build-arg MODEL_NAME="${MODEL_NAME}" --build-arg CLASS_NAME="${CLASS_NAME}" -t ${IMAGE_FULLNAME} .

docker-run: docker-stop docker-rm docker-build
	docker run --name skripsi -d -p 8080:80 ${IMAGE_FULLNAME}

docker-stop:
	-docker stop skripsi

docker-rm: docker-stop
	-docker rm -f skripsi
