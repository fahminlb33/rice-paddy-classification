# Variables
TF_LOG_LEVEL=2
MODEL_URL="https://kodesianastorage.blob.core.windows.net/kodesiana-public-assets/skripsi/model-289106b3e245440c97dedef10cafd8c1l.h5"
CLASS_NAMES_URL="https://kodesianastorage.blob.core.windows.net/kodesiana-public-assets/skripsi/class_names-289106b3e245440c97dedef10cafd8c1l.z"

IMAGE_REPO=fahminlb33
IMAGE_TAG=latest
IMAGE_NAME=skripsi
IMAGE_FULLNAME=${IMAGE_REPO}/${IMAGE_NAME}:${IMAGE_TAG}

# Directives
.PHONY: run debug docker-build docker-run docker-stop docker-rm

.DEFAULT_GOAL := docker-run

# Commands
run:
	MODEL_URL="../../${MODEL_URL}" \
	CLASS_NAMES_URL="../../${CLASS_NAMES_URL}" \
	TF_CPP_MIN_LOG_LEVEL=${TF_LOG_LEVEL} \
	uvicorn app:app --port 8080

debug:
	MODEL_URL="../../${MODEL_URL}" \
	CLASS_NAMES_URL="../../${CLASS_NAMES_URL}" \
	TF_CPP_MIN_LOG_LEVEL=${TF_LOG_LEVEL} \
	uvicorn app:app --port 8080 --reload

docker-build:
	docker build --build-arg MODEL_URL="${MODEL_URL}" --build-arg CLASS_NAMES_URL="${CLASS_NAMES_URL}" -t ${IMAGE_FULLNAME} .

docker-run: docker-stop docker-rm docker-build
	docker run --name skripsi -d -p 8080:80 ${IMAGE_FULLNAME}

docker-stop:
	-docker stop skripsi

docker-rm: docker-stop
	-docker rm -f skripsi
