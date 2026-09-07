import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    REPOSITORY_ROOT / 'Bird_HKE' / 'tools' / 'reorganize_dataset_sources.py'
)
SPEC = importlib.util.spec_from_file_location('reorganize_dataset_sources', SCRIPT_PATH)
FORMATTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FORMATTER)


class ReorganizeDatasetSourcesTests(unittest.TestCase):
    def _create_dataset(self, root: Path):
        paths = {
            'animal_kingdom': 'QWERTY/animal.jpg',
            'ebird': 'FINETUNE/ebird.jpg',
            'nabirds': '12345/nabirds.jpg',
            'birdsnap': 'Turdus_merula/birdsnap.jpg',
            'unannotated': 'Turdus_merula/unannotated.png',
        }
        for relative in paths.values():
            path = root / 'images' / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f'image:{relative}'.encode('utf-8'))
        (root / 'images' / 'Turdus_merula' / 'notes.txt').write_text(
            'not an image', encoding='utf-8'
        )

        annotation_dir = root / 'annot'
        annotation_dir.mkdir(parents=True)
        partitions = {
            'train.json': [
                {'image': paths['animal_kingdom'], 'id': 1},
                {'image': paths['birdsnap'], 'id': 2},
            ],
            'test.json': [{'image': paths['ebird'], 'id': 3}],
            'val.json': [{'image': paths['nabirds'], 'id': 4}],
        }
        for filename, records in partitions.items():
            (annotation_dir / filename).write_text(
                json.dumps(records), encoding='utf-8'
            )
        return paths

    def test_classification_precedence_and_rules(self):
        self.assertEqual(FORMATTER.classify_folder('FINETUNE'), 'eBird')
        self.assertEqual(FORMATTER.classify_folder('ABCXYZ'), 'Animal_Kingdom')
        self.assertEqual(FORMATTER.classify_folder('00742'), 'NABirds')
        self.assertEqual(FORMATTER.classify_folder('Corvus_corax'), 'Birdsnap')
        self.assertEqual(FORMATTER.classify_folder('MixedCase'), 'Birdsnap')

    def test_plan_counts_and_rewrites_every_annotation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / 'dataset'
            paths = self._create_dataset(root)
            entries, rewritten, manifest = FORMATTER.build_plan(root)

            self.assertEqual(len(entries), 5)
            self.assertEqual(manifest['counts']['original_images'], 5)
            self.assertEqual(manifest['counts']['annotated_unique_images'], 4)
            self.assertEqual(manifest['counts']['unannotated_images'], 1)
            self.assertEqual(manifest['counts']['ignored_non_image_files'], 1)
            rewritten_paths = {
                record['image']
                for records in rewritten.values()
                for record in records
            }
            self.assertEqual(
                rewritten_paths,
                {
                    'Animal_Kingdom/' + paths['animal_kingdom'],
                    'eBird/' + paths['ebird'],
                    'NABirds/' + paths['nabirds'],
                    'Birdsnap/' + paths['birdsnap'],
                },
            )

    def test_create_dataset_copies_all_images_and_preserves_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            root = parent / 'dataset'
            self._create_dataset(root)
            entries, rewritten, manifest = FORMATTER.build_plan(root)
            output = parent / 'formatted'
            completed = FORMATTER.create_dataset(
                root,
                output,
                entries,
                rewritten,
                manifest,
                show_progress=False,
            )

            self.assertTrue((root / 'images' / 'QWERTY' / 'animal.jpg').is_file())
            self.assertTrue(
                (output / 'images' / 'Animal_Kingdom' / 'QWERTY' / 'animal.jpg').is_file()
            )
            self.assertTrue(
                (output / 'images' / 'eBird' / 'FINETUNE' / 'ebird.jpg').is_file()
            )
            self.assertTrue(
                (output / 'images' / 'NABirds' / '12345' / 'nabirds.jpg').is_file()
            )
            self.assertTrue(
                (
                    output
                    / 'images'
                    / 'Birdsnap'
                    / 'Turdus_merula'
                    / 'unannotated.png'
                ).is_file()
            )
            output_train = json.loads(
                (output / 'annot' / 'train.json').read_text(encoding='utf-8')
            )
            self.assertEqual(output_train[0]['image'], 'Animal_Kingdom/QWERTY/animal.jpg')
            self.assertEqual(completed['counts']['output_images'], 5)
            self.assertTrue((output / 'source_format_manifest.json').is_file())
            self.assertFalse(output.with_name(output.name + '.partial').exists())

    def test_missing_annotated_image_is_rejected_before_copy(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / 'dataset'
            self._create_dataset(root)
            train_path = root / 'annot' / 'train.json'
            records = json.loads(train_path.read_text(encoding='utf-8'))
            records[0]['image'] = 'QWERTY/missing.jpg'
            train_path.write_text(json.dumps(records), encoding='utf-8')

            with self.assertRaises(FORMATTER.DatasetFormatError):
                FORMATTER.build_plan(root)

    def test_duplicate_annotation_reference_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / 'dataset'
            self._create_dataset(root)
            train = json.loads((root / 'annot' / 'train.json').read_text(encoding='utf-8'))
            test_path = root / 'annot' / 'test.json'
            test = json.loads(test_path.read_text(encoding='utf-8'))
            test.append(train[0])
            test_path.write_text(json.dumps(test), encoding='utf-8')

            with self.assertRaises(FORMATTER.DatasetFormatError):
                FORMATTER.build_plan(root)

    def test_existing_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            root = parent / 'dataset'
            self._create_dataset(root)
            entries, rewritten, manifest = FORMATTER.build_plan(root)
            output = parent / 'formatted'
            output.mkdir()

            with self.assertRaises(FORMATTER.DatasetFormatError):
                FORMATTER.create_dataset(
                    root,
                    output,
                    entries,
                    rewritten,
                    manifest,
                    show_progress=False,
                )


if __name__ == '__main__':
    unittest.main()
