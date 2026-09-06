FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libegl1 \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        libosmesa6 \
        libxrender1 \
        # GLFW/X11 client libs for mujoco.viewer's interactive window (GUI display mode).
        # Everything above is offscreen EGL/OSMesa rendering only and cannot open a window.
        libglfw3 \
        libx11-6 \
        libxrandr2 \
        libxinerama1 \
        libxcursor1 \
        libxi6 \
        libxxf86vm1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
RUN mkdir -p src/mujoco_warp_locomotion_sim \
    && touch src/mujoco_warp_locomotion_sim/__init__.py \
    && python -m pip install --no-cache-dir ".[mjwarp]"

COPY src ./src
COPY config ./config
COPY assets ./assets
COPY policies ./policies
RUN rm -rf build \
    && python -m pip uninstall -y mujoco-warp-locomotion-sim \
    && python -m pip install --no-cache-dir --no-deps .

EXPOSE 8010
HEALTHCHECK --interval=15s --timeout=5s --start-period=180s --retries=5 \
  CMD python -c "import json,urllib.request; assert json.load(urllib.request.urlopen('http://127.0.0.1:8010/health'))['ready']"

CMD ["mujoco-warp-locomotion-sim", "serve", "--config", "/app/config/default.toml"]
