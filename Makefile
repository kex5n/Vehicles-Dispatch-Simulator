IMAGE_NAME=simulator

build:
	docker build -t $(IMAGE_NAME):experiment --target experiment .

build-dev:
	docker build -t $(IMAGE_NAME):dev --target=dev .

train:
	docker run -it --rm -v `pwd`:/usr/src --entrypoint=python3 $(IMAGE_NAME):dev src/train.py
