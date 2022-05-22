IMAGE_NAME=simulator

build:
	docker build -t $(IMAGE_NAME):experiment .

train:
	docker run -it --rm -v `pwd`:/usr/src --entrypoint=python3 $(IMAGE_NAME):experiment src/train.py

test:
	docker run -it --rm -v `pwd`:/usr/src --entrypoint=python3 $(IMAGE_NAME):experiment src/test.py

run-dev:
	docker run -it --rm -v `pwd`:/usr/src -p 8888:8888 --ip 0.0.0.0 --entrypoint=/bin/bash $(IMAGE_NAME):experiment
