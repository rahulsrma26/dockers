IMAGE := welcometors/scrap_downloader
TAG := latest
PLATFORMS := linux/amd64,linux/arm64

build:
	docker build -t $(IMAGE):$(TAG) .

push:
	docker push $(IMAGE):$(TAG)

build-push:
	docker buildx build \
		--platform $(PLATFORMS) \
		--push \
		-t $(IMAGE):$(TAG) \
		.

.PHONY: build push build-push
