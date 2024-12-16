FROM quay.io/jupyter/minimal-notebook:afe30f0c9ad8

COPY conda-linux-64.lock /tmp/conda-linux-64.lock
COPY requirements.txt /tmp/requirements.txt

USER root

RUN sudo apt update \
    && sudo apt install -y \
    lmodern

RUN apt-get update && apt-get install -y build-essential make

USER $NB_UID

RUN mamba update --quiet --file /tmp/conda-linux-64.lock \
    && mamba clean --all -y -f \
    && pip install --no-cache-dir -r /tmp/requirements.txt \
    && pip cache purge \ 
    && fix-permissions "${CONDA_DIR}" \
    && fix-permissions "/home/${NB_USER}"