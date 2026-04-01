#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="${1:-ms3d_training}"
LABELCLOUD_VERSION="${LABELCLOUD_VERSION:-1.1.1}"

docker exec "${CONTAINER_NAME}" bash -lc "
set -euo pipefail

python -m pip show virtualenv >/dev/null 2>&1 || python -m pip install virtualenv

if [ ! -x /opt/labelcloud-venv/bin/python ]; then
  python -m virtualenv /opt/labelcloud-venv
fi

apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y \
  libglu1-mesa \
  libfontconfig1 \
  libfreetype6 \
  libxcb-icccm4 \
  libxcb-image0 \
  libxcb-shm0 \
  libxcb-keysyms1 \
  libxcb-randr0 \
  libxcb-render-util0 \
  libxcb-render0 \
  libxcb-shape0 \
  libxcb-xfixes0 \
  libxcb-xinerama0 \
  libxcb-xkb1 \
  libxkbcommon0 \
  libxkbcommon-x11-0 \
  libdbus-1-3

. /opt/labelcloud-venv/bin/activate
python -m pip install --upgrade pip
python -m pip install \"labelCloud==${LABELCLOUD_VERSION}\"
"

docker exec "${CONTAINER_NAME}" bash -lc ". /opt/labelcloud-venv/bin/activate && labelCloud --version"
