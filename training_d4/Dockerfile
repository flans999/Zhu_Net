FROM pytorch/pytorch@sha256:c16f4c749e2d9e96878875cdf6cc45cddda1d1a36fddd371dd6f2360f1b6e2a2
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
    XDG_CACHE_HOME=/scratch/cache TORCH_HOME=/scratch/torch \
    CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=4
WORKDIR /app
COPY requirements.lock /app/requirements.lock
RUN python -m pip install --no-cache-dir --no-deps --only-binary=:all: --require-hashes -r /app/requirements.lock
COPY . /app
RUN python /app/runtime_info.py --require-build \
    && python -m pip freeze --all > /app/environment.lock \
    && python -c "import platform,json;json.dump({'python':platform.python_version()},open('/app/python-version.json','w'))" \
    && mkdir -p /datasets /outputs /scratch \
    && chmod 0777 /outputs /scratch
USER 1000:1000
ENTRYPOINT []
CMD ["python", "-u", "/app/train.py", "--help"]
