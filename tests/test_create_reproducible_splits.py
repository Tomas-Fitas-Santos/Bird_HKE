import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    REPOSITORY_ROOT / 'Bird_HKE' / 'tools' / 'create_reproducible_splits.py'
)
SPEC = importlib.util.spec_from_file_location('create_reproducible_splits', SCRIPT_PATH)
SPLITTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SPLITTER)


class ReproducibleSplitTests(unittest.TestCase):
    def _create_dataset(self, root: Path):
        records = []
        for folder, count in (('species_a', 12), ('species_b', 6), ('species_c', 3)):
            image_dir = root / 'images' / folder
            image_dir.mkdir(parents=True, exist_ok=True)
            for index in range(count):
                relative = f'{folder}/image_{index:02d}.jpg'
                (root / 'images' / relative).write_bytes(b'image')
                records.append({'image': relative, 'record_id': f'{folder}-{index}'})

        annotation_dir = root / 'annot'
        annotation_dir.mkdir(parents=True)
        partitions = {
            'train': records[::3],
            'test': records[1::3],
            'val': records[2::3],
        }
        for name, values in partitions.items():
            (annotation_dir / f'{name}.json').write_text(
                json.dumps(values), encoding='utf-8'
            )
        return records

    def test_load_pool_and_split_without_loss_or_overlap(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original = self._create_dataset(root)
            pooled, source_info = SPLITTER.load_source_annotations(root)
            outputs, manifest = SPLITTER.split_annotations(pooled, seed=2026)

            self.assertEqual(sum(len(values) for values in outputs.values()), len(original))
            images = [record['image'] for values in outputs.values() for record in values]
            self.assertEqual(len(images), len(set(images)))
            self.assertEqual(set(source_info), {'train', 'test', 'val'})
            self.assertEqual(manifest['folders']['species_a']['train'], 10)
            self.assertEqual(manifest['folders']['species_a']['val'], 1)
            self.assertEqual(manifest['folders']['species_a']['calibration'], 1)
            self.assertEqual(manifest['folders']['species_b']['train'], 5)
            self.assertEqual(manifest['folders']['species_c']['train'], 3)
            self.assertEqual(manifest['folders']['species_c']['val'], 0)
            self.assertEqual(manifest['folders']['species_c']['calibration'], 0)

    def test_split_is_independent_of_input_record_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._create_dataset(root)
            pooled, _ = SPLITTER.load_source_annotations(root)
            first, _ = SPLITTER.split_annotations(pooled, seed=2026)
            second, _ = SPLITTER.split_annotations(reversed(pooled), seed=2026)
            for name in SPLITTER.OUTPUT_SPLITS:
                self.assertEqual(
                    [record['image'] for record in first[name]],
                    [record['image'] for record in second[name]],
                )

    def test_write_creates_manifest_and_preserves_source_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._create_dataset(root)
            pooled, source_info = SPLITTER.load_source_annotations(root)
            outputs, manifest = SPLITTER.split_annotations(pooled)
            output_dir = root / 'annot_repro_v1'
            SPLITTER.write_outputs(output_dir, outputs, manifest, source_info)

            for name in SPLITTER.OUTPUT_SPLITS:
                self.assertTrue((output_dir / f'{name}.json').is_file())
            written_manifest = json.loads(
                (output_dir / 'split_manifest.json').read_text(encoding='utf-8')
            )
            self.assertEqual(written_manifest['seed'], 2026)
            self.assertTrue(written_manifest['outputs']['train']['sha256'])
            self.assertTrue((root / 'annot' / 'test.json').is_file())

    def test_duplicate_image_across_sources_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._create_dataset(root)
            train_path = root / 'annot' / 'train.json'
            test_path = root / 'annot' / 'test.json'
            train = json.loads(train_path.read_text(encoding='utf-8'))
            test = json.loads(test_path.read_text(encoding='utf-8'))
            test.append(train[0])
            test_path.write_text(json.dumps(test), encoding='utf-8')
            with self.assertRaises(SPLITTER.SplitError):
                SPLITTER.load_source_annotations(root)

    def test_images_prefix_is_rejected(self):
        with self.assertRaises(SPLITTER.SplitError):
            SPLITTER.normalize_image_path('images/species/example.jpg')


if __name__ == '__main__':
    unittest.main()
