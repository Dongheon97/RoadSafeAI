import os
from pathlib import Path

_ACTIVE_WANDB_RUN = None


def _safe_scalar(value):
    if value is None:
        return None
    if hasattr(value, 'item'):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _collect_run_config(args, cfg, output_dir):
    return {
        'cfg_file': getattr(args, 'cfg_file', None),
        'extra_tag': getattr(args, 'extra_tag', None),
        'pretrained_model': getattr(args, 'pretrained_model', None),
        'batch_size': getattr(args, 'batch_size', None),
        'epochs': getattr(args, 'epochs', None),
        'output_dir': str(output_dir),
        'exp_group_path': getattr(cfg, 'EXP_GROUP_PATH', None),
        'tag': getattr(cfg, 'TAG', None),
        'class_names': list(getattr(cfg, 'CLASS_NAMES', [])) if hasattr(cfg, 'CLASS_NAMES') else None,
    }


def get_active_wandb_run():
    return _ACTIVE_WANDB_RUN


def init_wandb_if_available(args, cfg, output_dir, logger=None, job_type='train'):
    global _ACTIVE_WANDB_RUN

    if _ACTIVE_WANDB_RUN is not None:
        return _ACTIVE_WANDB_RUN

    project = os.getenv('WANDB_PROJECT')
    if not project:
        if logger is not None:
            logger.info('wandb disabled: WANDB_PROJECT is not set')
        return None

    try:
        import wandb
    except ImportError:
        if logger is not None:
            logger.info('wandb disabled: package is not installed')
        return None
    except Exception as exc:
        if logger is not None:
            logger.info(f'wandb disabled: import failed ({exc})')
        return None

    run_name = os.getenv('WANDB_RUN_NAME') or getattr(args, 'extra_tag', None) or Path(output_dir).name
    entity = os.getenv('WANDB_ENTITY') or None
    tags_env = os.getenv('WANDB_TAGS', '')
    tags = [tag.strip() for tag in tags_env.split(',') if tag.strip()]
    mode = os.getenv('WANDB_MODE', 'online')

    try:
        _ACTIVE_WANDB_RUN = wandb.init(
            project=project,
            entity=entity,
            name=run_name,
            tags=tags,
            job_type=job_type,
            dir=str(output_dir),
            mode=mode,
            config=_collect_run_config(args, cfg, output_dir),
            reinit=False,
        )
        if logger is not None and _ACTIVE_WANDB_RUN is not None:
            logger.info(f'wandb enabled: project={project}, run={_ACTIVE_WANDB_RUN.name}, mode={mode}')
    except Exception as exc:
        _ACTIVE_WANDB_RUN = None
        if logger is not None:
            logger.info(f'wandb disabled: init failed ({exc})')

    return _ACTIVE_WANDB_RUN


def finish_wandb(logger=None):
    global _ACTIVE_WANDB_RUN
    if _ACTIVE_WANDB_RUN is None:
        return
    try:
        _ACTIVE_WANDB_RUN.finish()
        if logger is not None:
            logger.info('wandb run finished')
    except Exception as exc:
        if logger is not None:
            logger.info(f'wandb finish failed ({exc})')
    finally:
        _ACTIVE_WANDB_RUN = None


class MultiScalarLogger:
    def __init__(self, tensorboard_writer=None, wandb_run=None):
        self.tensorboard_writer = tensorboard_writer
        self.wandb_run = wandb_run

    def add_scalar(self, tag, value, global_step=None):
        if self.tensorboard_writer is not None:
            self.tensorboard_writer.add_scalar(tag, value, global_step)

        if self.wandb_run is not None:
            try:
                payload = {tag: _safe_scalar(value)}
                if global_step is not None:
                    self.wandb_run.log(payload, step=int(global_step))
                else:
                    self.wandb_run.log(payload)
            except Exception:
                pass

    def close(self):
        if self.tensorboard_writer is not None:
            self.tensorboard_writer.close()
