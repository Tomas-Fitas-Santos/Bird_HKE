"""Ask an active training run to stop safely after its current epoch."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import yaml


STOP_REQUEST_FILENAME = 'stop_after_epoch.request'


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cfg', required=True, type=Path, help='Experiment YAML')
    return parser.parse_args()


def _load_run_directory(config_path: Path) -> Path:
    with config_path.open('r', encoding='utf-8') as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f'{config_path}: configuration root must be a mapping')

    train = config.get('TRAIN')
    if not isinstance(train, dict):
        raise ValueError(f'{config_path}: TRAIN section is missing')
    run_directory = train.get('LOG_DIR')
    if not run_directory:
        raise ValueError(f'{config_path}: TRAIN.LOG_DIR is missing or empty')
    path = Path(os.path.expanduser(str(run_directory)))
    return path if path.is_absolute() else Path.cwd() / path


def _atomic_request(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f'.{path.name}.', suffix='.tmp', dir=str(path.parent), text=True
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary_path), str(path))
    except BaseException:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass
        raise


def main() -> int:
    args = parse_args()
    config_path = args.cfg.resolve()
    run_directory = _load_run_directory(config_path).resolve()
    state_path = run_directory / 'training_state.json'
    request_path = run_directory / STOP_REQUEST_FILENAME

    current_state = None
    try:
        with state_path.open('r', encoding='utf-8') as handle:
            current_state = json.load(handle)
    except FileNotFoundError:
        pass

    _atomic_request(
        request_path,
        {
            'config_file': str(config_path),
            'requested_at_utc': datetime.now(timezone.utc).isoformat(),
            'training_status_when_requested': (
                current_state.get('status')
                if isinstance(current_state, dict)
                else None
            ),
        },
    )
    print(f'Safe stop requested for: {config_path}')
    print(f'Request file: {request_path}')
    print('The trainer will finish and checkpoint its current epoch, then exit.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
