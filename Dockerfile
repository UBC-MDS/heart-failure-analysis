FROM quay.io/jupyter/minimal-notebook:afe30f0c9ad8

COPY conda-linux-64.lock /tmp/conda-linux-64.lock
COPY requirements.txt /tmp/requirements.txt

USER root

# install lmodern for Quarto PDF rendering
RUN sudo apt update \
    && sudo apt install -y lmodern

USER $NB_UID

RUN mamba update --quiet --file /tmp/conda-linux-64.lock
RUN mamba clean --all -y -f

RUN pip install --no-cache-dir -r /tmp/requirements.txt
RUN pip cache purge

RUN fix-permissions "${CONDA_DIR}"
RUN fix-permissions "/home/${NB_USER}"

RUN pip install deepchecks==0.18.1

