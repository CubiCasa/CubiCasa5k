"""Run the trained CubiCasa5k model on one image, on CPU or CUDA."""
import argparse
import json
from pathlib import Path
import time
from xml.etree import ElementTree as ET

import numpy as np
from PIL import Image
import torch

from floortrans.inference import ICON_CLASSES, ROOM_CLASSES, SPLIT, load_model, predict, prepare_image
from floortrans.post_prosessing import get_polygons, split_prediction


def write_vectors(output, image_size, polygons, types, rooms, room_types):
    width, height = image_size
    elements = []
    for polygon, info in zip(polygons, types):
        classes = ICON_CLASSES if info['type'] == 'icon' else ROOM_CLASSES
        elements.append({"type": info['type'], "class": int(info['class']),
                         "name": classes[info['class']], "points": polygon.tolist(),
                         "holes": []})
    for polygon, info in zip(rooms, room_types):
        elements.append({"type": "room", "class": int(info['class']),
                         "name": ROOM_CLASSES[info['class']],
                         "points": list(map(list, polygon.exterior.coords)),
                         "holes": [list(map(list, ring.coords)) for ring in polygon.interiors]})
    svg = ET.Element("svg", xmlns="http://www.w3.org/2000/svg", viewBox=f"0 0 {width} {height}",
                     width=str(width), height=str(height))
    # Rooms first so walls and openings remain visible.
    for item in sorted(elements, key=lambda item: item['type'] != 'room'):
        rings = [item['points'], *item['holes']]
        d = " ".join("M " + " L ".join(f"{x:g},{y:g}" for x, y in ring) + " Z" for ring in rings)
        color = '#1e293b' if item['type'] == 'wall' else '#38bdf8' if item['type'] == 'icon' else '#cbd5e1'
        path = ET.SubElement(svg, "path", d=d, fill=color, **{"fill-rule": "evenodd", "stroke": "#475569", "stroke-width": "0.5"})
        ET.SubElement(path, "title").text = item['name']
    ET.ElementTree(svg).write(output / 'prediction.svg', encoding='utf-8', xml_declaration=True)
    return elements


def write_interface_vectors(output, image_size, elements):
    """Export a simplified centre-line view for Floorplan Lab's popup parser.

    Exact wall outlines, room polygons and holes remain in prediction.json/SVG.
    The legacy popup format has no representation for those geometries.
    """
    walls, objects = [], []
    for item in elements:
        if item['type'] == 'room':
            continue
        points = np.asarray(item['points'])
        x1, y1 = points.min(0)
        x2, y2 = points.max(0)
        opening = item['type'] == 'icon' and item['class'] in (1, 2)
        if item['type'] == 'wall' or opening:
            if x2 - x1 >= y2 - y1:
                y1 = y2 = (y1 + y2) / 2
            else:
                x1 = x2 = (x1 + x2) / 2
        coords = f'{x1:g}\t{y1:g}\t{x2:g}\t{y2:g}'
        if item['type'] == 'wall':
            walls.append(coords + '\t0\t0')
        else:
            name = item['name'].lower().replace(' ', '_')
            objects.append(coords + f'\t{name}\t1\t1')
    width, height = image_size
    text = '\n'.join([f'{width}\t{height}', str(len(walls)), *walls, *objects]) + '\n'
    (output / 'floorplan.txt').write_text(text)


def infer(input_path, output_dir, checkpoint, device='auto', max_size=512, rotate=True):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    def progress(message):
        pending = output / 'progress.tmp'
        pending.write_text(json.dumps({'message': message}))
        pending.replace(output / 'progress.json')

    start = time.monotonic()
    progress('Loading the trained CubiCasa model…')
    model = load_model(checkpoint, device)
    image, tensor, original_size = prepare_image(input_path, max_size)
    image.save(output / 'input.png')
    progress('Detecting rooms, wall junctions, openings and fixtures…')
    prediction = predict(model, tensor, rotate=rotate)
    heatmaps, rooms, icons = split_prediction(prediction, (image.height, image.width), SPLIT)
    # The legacy polygon routine modifies room probabilities. Save masks first.
    room_mask, icon_mask = rooms.argmax(0).astype(np.uint8), icons.argmax(0).astype(np.uint8)
    Image.fromarray(room_mask).save(output / 'rooms.png')
    Image.fromarray(icon_mask).save(output / 'icons.png')
    progress('Reconstructing polygons…')
    polygons, types, room_polygons, room_types = get_polygons((heatmaps, rooms, icons), 0.4, [1, 2])
    elements = write_vectors(output, image.size, polygons, types, room_polygons, room_types)
    write_interface_vectors(output, image.size, elements)
    result = {'engine': 'cubicasa', 'device': str(next(model.parameters()).device),
              'width': image.width, 'height': image.height, 'original_size': list(original_size),
              'rotate': rotate, 'elapsed': time.monotonic() - start,
              'room_classes': ROOM_CLASSES, 'icon_classes': ICON_CLASSES, 'elements': elements}
    (output / 'prediction.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    progress('Reconstruction complete.')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--weights', '--checkpoint', dest='checkpoint', type=Path, required=True)
    parser.add_argument('--device', default='auto', help='auto, cpu, cuda or cuda:N')
    parser.add_argument('--max-size', type=int, default=512, help='Maximum input side; 0 keeps original size')
    parser.add_argument('--rotate', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('--threads must be positive')
    torch.set_num_threads(args.threads)
    try:
        result = infer(args.input, args.output, args.checkpoint, args.device, args.max_size, args.rotate)
    except Exception as exc:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / 'error.json').write_text(json.dumps({'error': str(exc)}))
        raise
    print(json.dumps({key: result[key] for key in ('engine', 'device', 'width', 'height', 'elapsed')}))


if __name__ == '__main__':
    main()
