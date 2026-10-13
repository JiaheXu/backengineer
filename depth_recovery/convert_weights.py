"""Convert recovered ONNX initializers or NPZ arrays to a PyTorch state_dict."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
import onnx
import torch


def convert_weights(source, output=None, *, overwrite=False):
    source = Path(source).resolve()
    target = Path(output).resolve() if output else source.with_suffix('.pt')
    if source == target:
        raise ValueError('Input and checkpoint output must be different files')
    if target.exists() and not overwrite:
        raise FileExistsError(f'{target} exists; use --overwrite to replace it')
    if source.suffix.lower() == '.onnx':
        graph = onnx.load(str(source)).graph
        arrays = {t.name: onnx.numpy_helper.to_array(t) for t in graph.initializer}
        if len(arrays) != len(graph.initializer):
            raise ValueError('ONNX contains duplicate initializer names')
    elif source.suffix.lower() == '.npz':
        with np.load(source, allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
    else:
        raise ValueError('Input must be a recovered .onnx model or .npz weight archive')
    if not arrays:
        raise ValueError('Input contains no embedded weight tensors')
    state = {name: torch.from_numpy(np.array(value, copy=True)) for name, value in arrays.items()}
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, prefix=f'.{target.name}.', delete=False) as temp:
        temp_path = Path(temp.name)
    try:
        torch.save(state, temp_path)
        saved = torch.load(temp_path, map_location='cpu', weights_only=True)
        if set(saved) != set(arrays):
            raise ValueError('Checkpoint verification failed: tensor names differ')
        for name, original in arrays.items():
            actual = saved[name].numpy()
            if actual.dtype != original.dtype or actual.shape != original.shape or actual.tobytes() != original.tobytes():
                raise ValueError(f'Checkpoint verification failed for {name}')
        temp_path.replace(target)
    finally:
        temp_path.unlink(missing_ok=True)
    return {'source': str(source), 'checkpoint': str(target), 'tensors': len(state),
            'tensor_elements': sum(t.numel() for t in state.values()),
            'sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
            'verification': 'all tensor names, shapes, dtypes and bytes match'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', help='Recovered ONNX file or NPZ weight archive')
    parser.add_argument('-o', '--output', help='Output checkpoint; default: input name with .pt suffix')
    parser.add_argument('--overwrite', action='store_true', help='Replace an existing output checkpoint')
    args = parser.parse_args()
    try:
        print(json.dumps(convert_weights(args.source, args.output, overwrite=args.overwrite), indent=2))
    except (OSError, ValueError) as error:
        parser.exit(1, f'Weight conversion failed: {error}\n')


if __name__ == '__main__':
    main()
