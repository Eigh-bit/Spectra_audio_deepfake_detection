"""CPU-only contract checks; stub scoring is not a model accuracy evaluation."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from audio_spectra import production


class StubEngine:
    loading = {'stub': True}

    def score_file(self, path, mode):
        return {'fake_score': {'a.wav': -2., 'm.wav': 0., 'z.wav': 2.}[path.name]}

    def resources(self):
        return {}


class InferenceContractTests(unittest.TestCase):
    def test_default_and_named_profiles(self):
        self.assertEqual(production.load_config()['mode'], 'V2-Candidate')
        for name, mode, fusion in [('v2', 'V2-Candidate', 'off'),
                                   ('top2', 'V3-Top2', 'off'),
                                   ('domain_moe', 'V3-Top2', 'domain_moe')]:
            config = production.load_config(ROOT / 'configs' / (name + '.json'))
            self.assertEqual(config['mode'], mode)
            self.assertEqual(config.get('score_fusion', 'off'), fusion)
            manifest = ROOT / config['resource_manifest']
            self.assertTrue(manifest.is_file())
            resources = json.loads(manifest.read_text())
            checkpoint = next(x['sha256'] for x in resources if x['path'] == 'spectra0/model.safetensors')
            self.assertEqual(checkpoint, '830d05e5ff3fe6860858fdfee2bfdf61e0287bbf26892731e846ff7bbef5546b'
                             if name == 'v2' else 'd355d9175ead0a95856e6310e1d669d1cb86324dae830e079bbbc208d0707668')

    def test_resource_integrity_and_tampering(self):
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d)
            (folder / 'checkpoint').write_bytes(b'fixture only')
            manifest = folder / 'resources.json'
            manifest.write_text(json.dumps([{'path': 'checkpoint', 'sha256': hashlib.sha256(b'fixture only').hexdigest()}]))
            production.verify_weights(folder, manifest)
            (folder / 'checkpoint').write_bytes(b'corrupted fixture')
            with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
                production.verify_weights(folder, manifest)

    def test_csv_matches_frozen_baseline(self):
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d); inputs = folder / 'input'; inputs.mkdir()
            for name in ['z.wav', 'm.wav', 'a.wav']:
                (inputs / name).write_bytes(b'stub engine does not decode these files')
            output = folder / 'result.csv'
            production.run_directory(inputs, output, production.load_config(ROOT / 'configs/v2.json'), engine=StubEngine())
            self.assertEqual(output.read_bytes(), (ROOT / 'tests/fixtures/predictions.csv').read_bytes())
            status = json.loads((folder / 'result.csv.state/status.json').read_text())
            self.assertEqual((status['status'], status['completed']), ('Completed', 3))
            with self.assertRaises(FileExistsError):
                production.run_directory(inputs, output, production.load_config(ROOT / 'configs/v2.json'), engine=StubEngine())

    def test_failure_does_not_publish_partial_csv(self):
        class BrokenEngine(StubEngine):
            def score_file(self, path, mode):
                if path.name == 'z.wav':
                    raise ValueError('controlled scoring failure')
                return super().score_file(path, mode)
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d); inputs = folder / 'input'; inputs.mkdir()
            for name in ['a.wav', 'z.wav']:
                (inputs / name).write_bytes(b'fixture')
            output = folder / 'result.csv'
            with self.assertRaisesRegex(ValueError, 'controlled scoring failure'):
                production.run_directory(inputs, output, production.load_config(ROOT / 'configs/v2.json'), engine=BrokenEngine())
            self.assertFalse(output.exists())
            status = json.loads((folder / 'result.csv.state/status.json').read_text())
            self.assertEqual((status['status'], status['completed']), ('Failed', 1))

    def test_help_and_missing_resources_need_no_torch(self):
        help_result = subprocess.run([sys.executable, str(ROOT / 'predict.py'), '--help'], capture_output=True, text=True)
        self.assertEqual(help_result.returncode, 0)
        self.assertIn('--weights', help_result.stdout)
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d)
            result = subprocess.run([sys.executable, str(ROOT / 'predict.py'), str(folder), str(folder / 'out.csv'),
                                     '--weights', str(folder / 'missing')], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Model resource missing or SHA256 mismatch', result.stderr)
            self.assertNotIn('No module named', result.stderr)
            self.assertFalse((folder / 'out.csv').exists())


if __name__ == '__main__':
    unittest.main()
