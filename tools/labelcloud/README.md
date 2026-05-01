# labelCloud in `ms3d_training`

This setup runs `labelCloud` inside the existing `ms3d_training` container with X11 forwarding and uses the RoadSafeAI custom dataset at `/MS3D/data/custom_dataset/training/velodyne`.

## What this setup does

- Uses a dedicated virtual environment at `/opt/labelcloud-venv` inside the container.
- Keeps the MS3D workflow unchanged: use `PYTHONPATH=/MS3D` for training and testing as before.
- Uses `tools/labelcloud/config.ini` as the working config by launching `labelCloud` from `/MS3D/tools/labelcloud`.
- Defaults to one class, `Pedestrian`, with export format `kitti_untransformed`.
- Writes annotations to `/MS3D/data/custom_dataset/labelcloud/labels/`.

## Install into the running container

From the repo root on the host:

```bash
bash tools/labelcloud/install_in_ms3d_container.sh
```

This installs:

- `labelCloud==1.1.1`
- required Qt/X11/OpenGL runtime libraries
- the container-local virtualenv `/opt/labelcloud-venv`

If you recreate `ms3d_training`, run the install script again.

## Launch the GUI

From the repo root on the host:

```bash
bash tools/labelcloud/run_in_ms3d_container.sh
```

The launcher expects:

- the `ms3d_training` container to be running
- host `DISPLAY` to be set
- the container to have `/tmp/.X11-unix` mounted and `DISPLAY` forwarded

The current `ms3d_training` container was validated with:

- `DISPLAY=:1`
- bind mount `/tmp/.X11-unix:/tmp/.X11-unix`
- bind mount `/home/dongheon/workspace/c2i/RoadSafeAI:/MS3D`

## Dataset paths

- Point clouds: `/MS3D/data/custom_dataset/training/velodyne/`
- Labels: `/MS3D/data/custom_dataset/labelcloud/labels/`
- Segmentation output: `/MS3D/data/custom_dataset/labelcloud/segmentation/`
- Label class definition: `/MS3D/tools/labelcloud/labels/_classes.json`

## Notes

- `labelCloud` does not change the MS3D training entrypoints. Continue to use `tools/train.py` and `tools/test.py`.
- The default box dimensions in `config.ini` were changed from the upstream cart-sized defaults to pedestrian-sized defaults.
- If GUI startup fails after container recreation, check host X11 permissions first. Running the launcher from the same host shell session as your Docker/X11 workflow is the intended path.
