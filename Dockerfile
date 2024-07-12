FROM jupyter/scipy-notebook:python-3.10.5

RUN conda install --yes pytorch torchvision -c soumith
RUN pip install typeguard
RUN pip install pynndescent
RUN conda install --yes numba
RUN conda install --yes python-graphviz
RUN pip install pycodestyle flake8 pycodestyle_magic umap-learn
RUN pip install squarify
RUN pip install gensim
RUN pip install pycode_similar
RUN pip install pyastsim
RUN pip install zss

USER root
RUN mkdir -p /src && \
    chown root /src

USER $NB_UID
WORKDIR /data
EXPOSE 8888

CMD jupyter notebook --port=8888 --ip=0.0.0.0
