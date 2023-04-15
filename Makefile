.DEFAULT_GOAL := help

APPLICATION?=ai-train-kb-chatbot
COMMIT_SHA?=$(shell git rev-parse --short HEAD)
DOCKER?=docker
DOCKERHUB_OWNER?=jonascheng
DOCKER_IMG_NAME=${DOCKERHUB_OWNER}/${APPLICATION}
PWD?=$(shell pwd)

.PHONY: setup
setup: ## setup project
	python -m pip install -U pip
	pip install -r requirements.txt

.PHONY: docker-build
docker-build: ## build docker image
	${DOCKER} build -t ${DOCKER_IMG_NAME}:${COMMIT_SHA} .

.PHONY: docker-run
docker-run: docker-build ## run docker image
ifdef OPENAI_API_KEY
	${DOCKER} run -it -p 8080:8080 -e OPENAI_API_KEY=${OPENAI_API_KEY} ${DOCKER_IMG_NAME}:${COMMIT_SHA}
else
	$(error OPENAI_API_KEY not set on env)
endif

.PHONY: docker-debug
docker-debug: docker-build ## run docker image
ifdef OPENAI_API_KEY
	${DOCKER} run -it -v ${PWD}:/app -p 8080:8080 -e OPENAI_API_KEY=${OPENAI_API_KEY} ${DOCKER_IMG_NAME}:${COMMIT_SHA}
else
	$(error OPENAI_API_KEY not set on env)
endif

.PHONY: help
help: ## prints this help message
	@echo "Usage: \n"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
