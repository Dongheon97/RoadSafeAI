#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="${1:-ms3d_training}"

if [ -z "${DISPLAY:-}" ]; then
  echo "DISPLAY is not set on the host." >&2
  exit 1
fi

if ! docker exec "${CONTAINER_NAME}" bash -lc "test -x /opt/labelcloud-venv/bin/labelCloud"; then
  echo "labelCloud is not installed in ${CONTAINER_NAME}. Run tools/labelcloud/install_in_ms3d_container.sh first." >&2
  exit 1
fi

if command -v xhost >/dev/null 2>&1; then
  xhost +SI:localuser:root >/dev/null 2>&1 || true
fi

docker exec \
  -e DISPLAY="${DISPLAY}" \
  -e QT_X11_NO_MITSHM=1 \
  -e XDG_RUNTIME_DIR=/tmp/labelcloud-runtime \
  "${CONTAINER_NAME}" \
  bash -lc "
set -euo pipefail
mkdir -p /tmp/labelcloud-runtime
chmod 700 /tmp/labelcloud-runtime
mkdir -p /MS3D/data/custom_dataset/labelcloud/labels
mkdir -p /MS3D/data/custom_dataset/labelcloud/segmentation
cd /MS3D/tools/labelcloud
export PYTHONPATH=/MS3D
. /opt/labelcloud-venv/bin/activate
exec labelCloud
"
