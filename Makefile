help:
	@cat Makefile
SRC?=$(shell pwd)/src
DATASETS?=${SRC}/../../datasets
torch:
	docker build -t torch .
notebook: torch
	docker run -it -v ${SRC}:/data -v ${DATASETS}:/datasets --net=host torch
