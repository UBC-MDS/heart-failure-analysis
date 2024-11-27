FROM quay.io/jupyter/minimal-notebook:afe30f0c9ad8

COPY conda-linux-64.lock /tmp/conda-linux-64.lock
COPY requirements.txt /tmp/requirements.txt

RUN pip install --no-cache-dir -r /tmp/requirements.txt
RUN pip cache purge

RUN mamba update --quiet --file /tmp/conda-linux-64.lock
RUN mamba clean --all -y -f

RUN fix-permissions "${CONDA_DIR}"
RUN fix-permissions "/home/${NB_USER}"