# Variables
TF_LOG_LEVEL=2
MODEL_URL="https://kodesianastorage.blob.core.windows.net/kodesiana-public-assets/skripsi/model-289106b3e245440c97dedef10cafd8c1l.h5"
CLASS_NAMES_URL="https://kodesianastorage.blob.core.windows.net/kodesiana-public-assets/skripsi/class_names-289106b3e245440c97dedef10cafd8c1l.z"

IMAGE_REPO=fahminlb33
IMAGE_TAG=latest
IMAGE_NAME=skripsi
IMAGE_FULLNAME=${IMAGE_REPO}/${IMAGE_NAME}:${IMAGE_TAG}

# Directives
.PHONY: download run debug docker-build docker-run docker-stop docker-rm

.DEFAULT_GOAL := docker-run

# Commands
download:
	curl -L -o model.h5 ${MODEL_URL}
	curl -L -o class_names.z ${CLASS_NAMES_URL}
	echo "Model and class names downloaded"

run: download
	MODEL_NAME="model.h5" \
	CLASS_NAME="class_names.z" \
	TF_CPP_MIN_LOG_LEVEL=${TF_LOG_LEVEL} \
	uvicorn app:app --port 8080

debug: download
	MODEL_NAME="model.h5" \
	CLASS_NAME="class_names.z" \
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
