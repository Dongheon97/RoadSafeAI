#!/usr/bin/env bash
set -euo pipefail

if [[ -f /MS3D/setup.py && -d /MS3D/pcdet/ops ]]; then
    if ! compgen -G "/MS3D/pcdet/ops/iou3d_nms/iou3d_nms_cuda*.so" > /dev/null; then
        echo "[entrypoint] CUDA ops not found in mounted workspace. Building extensions in /MS3D..."
        original_pwd="$(pwd)"
        cd /MS3D
        python setup.py build_ext --inplace
        cd "${original_pwd}"
    fi
fi

exec "$@"
