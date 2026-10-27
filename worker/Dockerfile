FROM python:3.12-slim@sha256:cb33f7f771ecec2ef1225cf39b159591a040391bd73e4c01f00fdfc4c47611d2

# With SOURCE_DATE_EPOCH set, pip writes hash-based .pyc files, so a rebuild repeats the layer.
ARG SOURCE_DATE_EPOCH
RUN --mount=type=bind,source=requirements.txt,target=/requirements.txt \
    pip install --no-cache-dir --disable-pip-version-check --root-user-action=ignore \
        --require-hashes --no-deps --only-binary=:all: --requirement /requirements.txt

# env -i drops the image and runtime environment. The container gets one CPU, so OpenBLAS is held
# to one thread instead of one per host core counted against the pids and memory limits.
ENTRYPOINT ["/usr/bin/env", "-i", "PATH=/usr/local/bin:/usr/bin:/bin", "OPENBLAS_NUM_THREADS=1", "/bin/sh", "-c", "python -I /job/candidate.py >/dev/null && exec tar -C /output -cf - ."]
