import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPOSITORY_ROOT / 'Bird_HKE' / 'tools' / 'request_training_stop.py'
SPEC = importlib.util.spec_from_file_location('request_training_stop', SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class RequestTrainingStopTests(unittest.TestCase):
    def test_load_run_directory_from_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / 'experiment.yaml'
            config.write_text(
                'TRAIN:\n  LOG_DIR: outputs/model/seed_2026\n',
                encoding='utf-8',
            )
            self.assertEqual(
                MODULE._load_run_directory(config),
                Path.cwd() / 'outputs/model/seed_2026',
            )

    def test_atomic_request_writes_valid_json(self):
        with tempfile.TemporaryDirectory() as directory:
            request = Path(directory) / MODULE.STOP_REQUEST_FILENAME
            payload = {'config_file': 'example.yaml'}
            MODULE._atomic_request(request, payload)
            with request.open('r', encoding='utf-8') as handle:
                self.assertEqual(json.load(handle), payload)
            self.assertEqual(list(request.parent.glob('*.tmp')), [])


if __name__ == '__main__':
    unittest.main()
