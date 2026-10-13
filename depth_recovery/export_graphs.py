"""Export recovered ONNX graphs and initializers; verify every exported array."""
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
from google.protobuf.json_format import MessageToDict
from onnx import numpy_helper

ROOT = Path(__file__).resolve().parent
manifest = []
for path in sorted(ROOT.glob('*.onnx')):
    model = onnx.load(str(path))
    arrays = {t.name: numpy_helper.to_array(t) for t in model.graph.initializer}
    assert len(arrays) == len(model.graph.initializer), 'Duplicate initializer names'
    target = path.with_suffix('.weights.npz')
    np.savez(target, **arrays)
    with np.load(target, allow_pickle=False) as saved:
        assert set(saved.files) == set(arrays)
        for name, value in arrays.items():
            assert saved[name].dtype == value.dtype
            assert saved[name].shape == value.shape
            assert saved[name].tobytes() == value.tobytes()

    graph = MessageToDict(model, preserving_proto_field_name=True)
    # Tensor values remain in ONNX and NPZ; JSON records topology and metadata.
    for tensor in graph['graph'].get('initializer', []):
        for field in ['raw_data', 'float_data', 'int32_data', 'int64_data',
                      'double_data', 'uint64_data', 'string_data']:
            tensor.pop(field, None)
    path.with_suffix('.graph.json').write_text(json.dumps(graph, indent=2) + '\n')
    listing = []
    for index, node in enumerate(model.graph.node):
        attrs = MessageToDict(node, preserving_proto_field_name=True).get('attribute', [])
        listing.append(f'{index:04d} {node.name}: {node.domain or "ai.onnx"}::{node.op_type}'
                       f'\n  inputs: {list(node.input)}\n  outputs: {list(node.output)}'
                       f'\n  attributes: {json.dumps(attrs)}')
    path.with_suffix('.graph.txt').write_text('\n\n'.join(listing) + '\n')
    weights = [{"name": k, "shape": list(v.shape), "dtype": str(v.dtype),
                "elements": int(v.size), "bytes": int(v.nbytes)} for k, v in arrays.items()]
    path.with_suffix('.weights.json').write_text(json.dumps(weights, indent=2) + '\n')
    try:
        onnx.checker.check_model(str(path))
        validation = 'PASS (structure; custom sl operators are not schema-checked)'
    except onnx.checker.ValidationError as error:
        validation = str(error)
        assert 'No Op registered for CensusXnorCostVolume' in validation, validation
        # Verify the remainder on a temporary in-memory copy, keeping the original intact.
        check_copy = onnx.load(str(path))
        for node in check_copy.graph.node:
            if node.op_type == 'CensusXnorCostVolume':
                node.domain = 'sl'
        onnx.checker.check_model(check_copy)
        validation += '\nPASS after assigning CensusXnorCostVolume to sl in memory only; semantics untested.'
    item = {'file': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'nodes': len(model.graph.node), 'initializers': len(arrays),
            'initializer_elements': sum(int(x.size) for x in arrays.values()),
            'initializer_bytes': sum(int(x.nbytes) for x in arrays.values()),
            'operators': dict(Counter(n.op_type for n in model.graph.node)),
            'inputs': [{"name": x.name, "type": onnx.helper.printable_type(x.type)} for x in model.graph.input],
            'outputs': [{"name": x.name, "type": onnx.helper.printable_type(x.type)} for x in model.graph.output],
            'validation': validation, 'weights_roundtrip': 'PASS: every dtype, shape, and byte matches'}
    manifest.append(item)
    print(path.name, item['nodes'], 'nodes;', len(arrays), 'initializers;',
          item['initializer_elements'], 'initializer elements; NPZ roundtrip PASS')
ROOT.joinpath('manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
