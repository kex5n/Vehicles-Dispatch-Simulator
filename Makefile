IMAGE_NAME=simulator

build:
	docker build -t $(IMAGE_NAME):experiment --target experiment .

train:
	docker run -it --rm -v `pwd`:/usr/src --entrypoint=python3 $(IMAGE_NAME):experiment src/train.py

test:
	docker run -it --rm -v `pwd`:/usr/src --entrypoint=python3 $(IMAGE_NAME):experiment src/test.py
