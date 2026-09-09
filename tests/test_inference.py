import json
import logging
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
from shapely.geometry import Polygon
import torch

from floortrans.inference import load_model, predict, prepare_image, resolve_device
from floortrans.loaders.augmentations import RotateNTurns
from floortrans.metrics import get_evaluation_tensors, polygons_to_tensor
from floortrans.models import get_model
from floortrans.plotting import discrete_cmap, discrete_cmap_furukawa
from floortrans.post_prosessing import get_polygons, merge_rectangles, split_prediction
from infer import write_interface_vectors, write_vectors


torch.set_num_threads(2)


class BackgroundModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def forward(self, image):
        out = image.new_zeros((1, 44, *image.shape[-2:]))
        out[:, 21] = 10  # background room
        out[:, 33] = 10  # empty icon
        return out + self.anchor


class InferenceTests(unittest.TestCase):
    def test_cpu_fallback_and_explicit_cuda_error(self):
        with patch('torch.cuda.is_available', return_value=False):
            self.assertEqual(resolve_device('auto'), torch.device('cpu'))
            with self.assertRaisesRegex(ValueError, 'CUDA is unavailable'):
                resolve_device('cuda')
        with self.assertRaises(ValueError):
            resolve_device('meta')

    def test_checkpoint_formats_and_strict_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.pth'
            for payload in ({'anchor': torch.tensor(3.)},
                            {'model_state': {'anchor': torch.tensor(3.)}},
                            {'model_state': {'module.anchor': torch.tensor(3.)}}):
                torch.save(payload, path)
                with patch('floortrans.inference.get_model', return_value=BackgroundModel()) as factory:
                    model = load_model(path, 'cpu')
                factory.assert_called_once_with('hg_furukawa_original', 44, pretrained=False)
                self.assertEqual(model.anchor.item(), 3.)
                self.assertFalse(model.training)
            torch.save({'wrong_key': torch.zeros(2)}, path)
            with patch('floortrans.inference.get_model', return_value=BackgroundModel()):
                with self.assertRaises(RuntimeError):
                    load_model(path, 'cpu')

    def test_checkpoint_load_is_restricted_and_mapped_to_cpu(self):
        with patch('floortrans.inference.torch.load', return_value={'anchor': torch.tensor(0.)}) as loader:
            with patch('floortrans.inference.get_model', return_value=BackgroundModel()):
                load_model('example.pth', 'cpu')
        loader.assert_called_once_with(Path('example.pth'), map_location='cpu', weights_only=True)

    def test_no_auxiliary_backbone_load(self):
        with patch('torch.load', side_effect=AssertionError('unexpected backbone load')):
            model = get_model('hg_furukawa_original', 44, pretrained=False)
        self.assertEqual(model.conv4_.out_channels, 44)
        with torch.inference_mode():
            out = model.eval()(torch.zeros(1, 3, 65, 97))
        self.assertEqual(out.shape[:2], (1, 44))
        self.assertTrue(torch.isfinite(out).all())

    def test_rgb_normalization_and_transparent_background(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'input.png'
            pixels = np.zeros((64, 128, 4), dtype=np.uint8)
            pixels[:, :64] = [255, 0, 0, 255]
            Image.fromarray(pixels).save(path)
            image, tensor, original = prepare_image(path)
            self.assertEqual(original, (128, 64))
            self.assertEqual(image.size, original)
            torch.testing.assert_close(tensor[0, :, 0, 0], torch.tensor([1., -1., -1.]))
            torch.testing.assert_close(tensor[0, :, 0, 127], torch.ones(3))
            with self.assertRaises(ValueError):
                prepare_image(path, 32)

    def test_aspect_ratio_and_minimum_size(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'input.png'
            Image.new('RGB', (1024, 512)).save(path)
            image, _, original = prepare_image(path, 256)
            self.assertEqual(image.size, (256, 128))
            self.assertEqual(original, (1024, 512))
            with self.assertRaises(ValueError):
                prepare_image(path, 64)

    def test_rotation_permutations_and_roundtrips(self):
        rot = RotateNTurns()
        channels = torch.arange(44.).reshape(1, 44, 1, 1)
        twice = rot(rot(channels, 'points', 1), 'points', 1)
        torch.testing.assert_close(rot(channels, 'points', 2), twice)
        for turns, back in [(1, -1), (-1, 1), (2, 2)]:
            torch.testing.assert_close(rot(rot(channels, 'points', turns), 'points', back), channels)
            spatial = torch.arange(64 * 96).reshape(1, 1, 64, 96)
            torch.testing.assert_close(rot(rot(spatial, 'tensor', turns), 'tensor', back), spatial)

    def test_rotation_and_single_pass_sizes(self):
        model = BackgroundModel().eval()
        for rotate in (False, True):
            out = predict(model, torch.zeros(1, 3, 65, 97), size=(67, 99), rotate=rotate)
            self.assertEqual(out.shape, (1, 44, 67, 99))
            self.assertFalse(out.requires_grad)
            self.assertEqual(out.device.type, 'cpu')
            heatmaps, rooms, icons = split_prediction(out, (67, 99), (21, 12, 11))
            np.testing.assert_allclose(rooms.sum(0), 1, atol=1e-6)
            np.testing.assert_allclose(icons.sum(0), 1, atol=1e-6)
            polygons, types, room_polygons, room_types = get_polygons((heatmaps, rooms, icons), .4, [1, 2])
            self.assertEqual(len(polygons), 0)
            self.assertEqual(len(room_polygons), 0)

    def test_evaluation_cpu_postprocessing_both_rotation_modes(self):
        val = {'image': torch.zeros(1, 3, 64, 96), 'label': torch.zeros(1, 23, 64, 96)}
        for rotate in (False, True):
            outputs = get_evaluation_tensors(val, BackgroundModel().eval(), (21, 12, 11), logging.getLogger(), rotate)
            for result in outputs:
                self.assertEqual(result.shape, (2, 64, 96))
                self.assertTrue(np.all(result == 0))

    def test_shapely_multipart_and_missing_classes(self):
        rectangles = [np.array([[0, 0], [2, 0], [2, 2], [0, 2]]),
                      np.array([[4, 0], [6, 0], [6, 2], [4, 2]])]
        polygons, types = merge_rectangles(rectangles, [{'class': 3}, {'class': 3}])
        self.assertEqual(len(polygons), 2)
        self.assertTrue(all(isinstance(p, Polygon) for p in polygons))
        self.assertEqual([t['class'] for t in types], [3, 3])
        self.assertEqual(sum(p.area for p in polygons), 8.)
        self.assertEqual(merge_rectangles([], []), ([], []))

    def test_polygon_rasterization_clips_image_boundary(self):
        polygons = np.array([[[-2, -2], [10, -2], [10, 10], [-2, 10]]])
        result = polygons_to_tensor(polygons, [{'type': 'wall', 'class': 2}], [], [], (8, 8))
        self.assertTrue(np.all(result[2] == 1))

    def test_colormaps_can_be_registered_repeatedly(self):
        discrete_cmap()
        discrete_cmap()
        discrete_cmap_furukawa()
        discrete_cmap_furukawa()

    def test_svg_and_json_geometry_preserve_holes(self):
        room = Polygon([(0, 0), (9, 0), (9, 9), (0, 9)], holes=[[(2, 2), (4, 2), (4, 4), (2, 4)]])
        with tempfile.TemporaryDirectory() as directory:
            elements = write_vectors(Path(directory), (10, 10), [], [], [room], [{'class': 3}])
            self.assertEqual(len(elements[0]['holes']), 1)
            json.dumps(elements, allow_nan=False)
            self.assertIn('fill-rule="evenodd"', (Path(directory) / 'prediction.svg').read_text())

    def test_interface_export_keeps_dimensions_and_opening_labels(self):
        elements = [
            {'type': 'wall', 'class': 2, 'name': 'Wall', 'points': [[0, 2], [10, 2], [10, 4], [0, 4]]},
            {'type': 'icon', 'class': 1, 'name': 'Window', 'points': [[4, 3], [6, 3], [6, 9], [4, 9]]},
            {'type': 'room', 'class': 3, 'name': 'Kitchen', 'points': []},
        ]
        with tempfile.TemporaryDirectory() as directory:
            write_interface_vectors(Path(directory), (64, 96), elements)
            lines = (Path(directory) / 'floorplan.txt').read_text().splitlines()
            self.assertEqual(lines, ['64\t96', '1', '0\t3\t10\t3\t0\t0', '5\t3\t5\t9\twindow\t1\t1'])

    def test_single_entry_lmdb_evaluation(self):
        import lmdb
        import pickle
        from types import SimpleNamespace
        from eval import evaluate

        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / 'test.txt').write_text('sample\n')
            sample = {'image': torch.zeros(3, 64, 96), 'label': torch.zeros(2, 64, 96),
                      'heatmaps': {}, 'scale': 1}
            with lmdb.open(str(folder / 'cubi_lmdb'), map_size=2**22) as database:
                with database.begin(write=True) as transaction:
                    transaction.put(b'sample', pickle.dumps(sample))
            args = SimpleNamespace(data_path=directory, split_file='test.txt', data_format='lmdb',
                                   lmdb_folder='cubi_lmdb/', n_classes=44, weights='unused.pth',
                                   device='cpu', arch='hg_furukawa_original', rotate=True)
            with patch('eval.load_model', return_value=BackgroundModel().eval()), np.errstate(invalid='ignore'):
                evaluate(args, directory, None, logging.getLogger())

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA hardware/build unavailable')
    def test_cuda_rotations_and_numpy_conversion(self):
        out = predict(BackgroundModel().cuda(), torch.zeros(1, 3, 64, 96), rotate=True)
        self.assertEqual(out.device.type, 'cuda')
        parts = split_prediction(out, (64, 96), (21, 12, 11))
        self.assertTrue(all(isinstance(part, np.ndarray) for part in parts))


if __name__ == '__main__':
    unittest.main()
