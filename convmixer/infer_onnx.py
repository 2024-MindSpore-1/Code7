# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
Run prediction on folder or single image, output results and save them to
JSON file.
"""
import argparse
import json
from pathlib import Path

import onnxruntime as ort

from src.data.imagenet import get_validation_transforms


def parse_args():
    """
    Create and parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, add_help=False,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '-h', '--help', action='help',
        default=argparse.SUPPRESS,
        help='Show this help message and exit.'
    )
    parser.add_argument(
        'data', type=Path,
        help='Path to dataset for prediction.'
    )
    parser.add_argument(
        '--onnx_path', type=Path,
        help='Path to ONNX file.'
    )
    parser.add_argument(
        '-o', '--output', type=Path, default=Path('predictions.json'),
        help='Path to output PKL file.'
    )
    parser.add_argument(
        '--image_size', type=int, default=224,
        help='Image size.'
    )
    parser.add_argument(
        '--device_target', default='CPU', choices=['GPU', 'CPU'],
        help='Target computation platform.'
    )

    return parser.parse_args()


def data_loader(path: Path, image_size: int):
    """Load image or images from folder in generator."""
    preprocess = get_validation_transforms(
        image_size=image_size, crop_pct=0.96
    )

    def apply(img):
        for p in preprocess:
            img = p(img)
        return img

    extensions = ('.png', '.jpg', '.jpeg')
    if path.is_dir():
        print('=' * 5, ' Load directory ', '=' * 5)
        for item in path.iterdir():
            if item.is_dir():
                continue
            if item.suffix.lower() not in extensions:
                continue
            with open(item, 'rb') as f:
                image_data = f.read()
            image = apply(image_data)
            yield str(item), image[None]
    else:
        print('=' * 5, ' Load single image ', '=' * 5)
        assert path.suffix.lower() in extensions

        with open(path, 'rb') as f:
            image_data = f.read()
        image = apply(image_data)
        yield str(path), image[None]


def create_session(onnx_path, device_target):
    """Create ONNX inference session."""
    if device_target == 'GPU':
        providers = ['CUDAExecutionProvider']
    elif device_target == 'CPU':
        providers = ['CPUExecutionProvider']
    else:
        raise ValueError(
            f'Unsupported target device {device_target}, '
            f'Expected one of: "CPU", "GPU"'
        )
    session = ort.InferenceSession(onnx_path, providers=providers)
    input_name = session.get_inputs()[0].name
    return session, input_name


def main():
    """Entry point."""
    args = parse_args()

    loader = data_loader(args.data, args.image_size)
    session, input_name = create_session(
        str(args.onnx_path), args.device_target
    )

    d = {}

    for (name, img) in loader:
        res = session.run(None, {input_name: img})[0].argmax()
        print(name, f'(class: {res})')
        d[name] = int(res)

    with args.output.open(mode='w') as f:
        json.dump(d, f, indent=1)


if __name__ == '__main__':
    main()
