# Modern inference and evaluation

The inference path targets Python 3.11 and PyTorch 2.14.0. CPU operation is tested.
CUDA and Docker are optional. The trained hourglass architecture and 44-channel
layout are unchanged: 21 junction heatmaps, 12 room classes and 11 icon classes.
Training, its loss functions, and the historical Dockerfile have not been migrated.

## Environment

An isolated `.venv` is already installed in this checkout. To reproduce it on
Linux with Python 3.11 installed, run from the CubiCasa5k directory:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements-inference.txt
.venv/bin/python -m pip check
```

If Python 3.11 is unavailable and you have [uv](https://docs.astral.sh/uv/):

```bash
uv venv --python 3.11 --seed .venv
uv pip sync --python .venv/bin/python requirements-inference.txt --index-strategy unsafe-best-match
```

`requirements-inference.txt` pins all resolved dependencies, including the CPU
PyTorch wheel. The two indexes are PyPI and the official PyTorch wheel index.
`requirements-inference.in` holds the direct dependencies; regenerate the lock
with the command in the `.txt` header. `requirements.txt` remains the historical
training environment and must not be used for the modern inference setup.
Jupyter is optional and is not part of this runtime lock.

## Trained checkpoint

The [authors' checkpoint](https://drive.google.com/file/d/1gRB7ez1e4H7a9Y09lLqRuna0luZO5VRK/view)
is already downloaded here as `checkpoints/model_best_val_loss_var.pkl` (209 MB).
On another machine, download that file from the linked page and put it at the same
path. Its SHA-256 in this checkout is:

```text
dd20b4e1bf1d670f2125107b079df06958b1ccd36e49a464ab739aeb00b8e7a2
```

Loading uses `map_location='cpu'`, `weights_only=True`, and strict state-dictionary
validation, then transfers the complete model to the selected device. Raw state
dictionaries and the authors' `model_state` wrapper are supported, including a
uniform `module.` prefix from DataParallel. Incompatible checkpoints fail; there
is no unrestricted-pickle fallback. A complete checkpoint no longer loads the
unrelated `model_1427.pth` pose-estimation weights first.

## Run one image

```bash
.venv/bin/python infer.py \
  --input ../FloorplanTransformation/frontend/public/sample.png \
  --weights checkpoints/model_best_val_loss_var.pkl \
  --output outputs/sample --device cpu
```

Replace `--input` with your own PNG/JPEG/WebP image. Paths also work when invoking
the script from another directory. `--device auto` selects CUDA only when PyTorch
can use it, otherwise CPU; `--device cuda` or `cuda:0` explicitly requests a GPU
and fails clearly when unavailable. `--threads 2` is the default CPU thread count.

The image is orientation-corrected, flattened against white, converted to RGB,
resized without stretching to a maximum side of 512, and normalized to `[-1, 1]`.
Images are not enlarged or square-padded. Use `--max-size 0` for native resolution,
or a larger limit when detail matters. Both processed dimensions must be at least
64 pixels. Resolution changes can affect accuracy and runtime. Four-rotation
averaging is enabled by default; use `--no-rotate` for a faster single pass.

Outputs:

- `input.png`: the actual processed image; all output coordinates refer to it.
- `rooms.png`, `icons.png`: class-ID masks, not colour previews. JSON includes the
  class-name lists. Do not infer classes from a PNG viewer's brightness.
- `prediction.json`: exact polygons, including room holes, image dimensions,
  original dimensions, device, rotation setting and processing time.
- `prediction.svg`: vector rendering of those polygons.
- `floorplan.txt`: a simplified view for the existing Floorplan Lab interface.
  Walls/openings become centre lines and fixtures become bounding boxes. This
  format omits room polygons, holes, wall thickness and adjacency labels; use
  JSON/SVG for the complete geometry.
- `progress.json` and, if inference fails, `error.json`.

Use a distinct output directory for each image/run. To map coordinates back to the
orientation-corrected original, multiply x by `original_size[0] / width` and y by
`original_size[1] / height`.

## Existing local interface

The sibling FloorplanTransformation service now accepts a second engine. From
that repository, run:

```bash
.venv/bin/python backend/server.py --engine cubicasa --port 8766
```

Open <http://127.0.0.1:8766>. The original engine remains available with
`--engine floorplan` (the default) on port 8765. This permits the same image to be
run in both browser tabs. The result identifies the engine and actual image size.
CubiCasa runs in its own Python 3.11 subprocess, with the existing upload limits,
authentication and three-minute job timeout. It does not alter the original
engine's environment. Use `--cubicasa-root /path/to/CubiCasa5k` for another checkout
and `--checkpoint /path/to/weights.pkl` to override its checkpoint. The interface
currently displays the simplified centre-line export described above.

## Dataset evaluation

The full benchmark requires the dataset and annotations. It has not been run in
this checkout. An LMDB database is optional; SVGs can be read directly:

```bash
.venv/bin/python eval.py --weights checkpoints/model_best_val_loss_var.pkl \
  --data-path data/cubicasa5k --data-format txt --split-file test.txt --device cpu
```

For an existing LMDB database, use `--data-format lmdb` (the default) and optionally
`--lmdb-folder cubi_lmdb/`. Evaluation runs at dataset resolution and writes
metrics to `runs_cubi/<timestamp>/eval.log`. `samples.ipynb` uses the same device-aware
loader and predictor; it still requires the dataset and an optional Jupyter kernel.
Notebook saved outputs were cleared because they predate these changes.

## Validation and limitations

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m pip check
```

The regression suite covers CPU fallback, explicit CUDA rejection, strict and
restricted checkpoint loading, RGB normalization, image sizing, rotation
permutations, the original network's forward pass, evaluation with/without
rotations, Shapely multipart geometry, boundary clipping and repeated colormap
registration. The CUDA runtime test skips when hardware is unavailable.

The authors' checkpoint completed inference and polygon reconstruction on the
existing 425 × 425 sample on this CPU in about 4.1 seconds (model loading through
postprocessing, excluding interpreter startup), with four rotations. It returned
26 wall polygons, five doors, three windows and five room polygons. These counts
verify execution, not reconstruction quality or the paper's benchmark accuracy.
The local-service upload-to-result smoke test also passed in about 5.1 seconds,
and the frontend parser retained the 425 × 425 coordinate system, 26 walls and
eight openings. There are 15 passing inference/evaluation tests, seven backend
tests and five frontend tests; one CUDA test is skipped. The frontend build and
`pip check` pass. Browser interaction was not tested. No failing-example accuracy
comparison or GPU validation has been completed.

Compatibility changes include explicit SciPy mode dimensions, Shapely 2 `.geoms`,
modern Matplotlib colormap registration, and detached CPU conversion before NumPy.
A pre-existing typo in the 180° junction permutation is corrected (channel 3 maps
to channel 1); consequently rotated predictions can differ from the original code.

For future GPU use, replace the CPU torch wheel with the matching release from
[PyTorch's official installation instructions](https://pytorch.org/get-started/locally/)
and record the CUDA environment separately. The host needs a compatible NVIDIA
driver and a visible GPU; VM passthrough may be necessary. Docker and the NVIDIA
Container Toolkit are needed only if choosing GPU containers.
