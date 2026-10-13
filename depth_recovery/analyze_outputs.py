"""Create a PDF and shared-scale visualizations from a saved model comparison."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Normalize
import numpy as np
from PIL import Image


def _range(*arrays):
    valid = np.concatenate([a[np.isfinite(a)].ravel() for a in arrays])
    low, high = (float(valid.min()), float(valid.max())) if valid.size else (0., 1.)
    return low, high if high > low else low + 1.


def generate_report(directory):
    directory = Path(directory).resolve()
    report = json.loads((directory / 'comparison.json').read_text())
    with np.load(directory / 'onnx_outputs.npz', allow_pickle=False) as archive:
        reference = {key: archive[key].astype(np.float64) for key in archive.files}
    with np.load(directory / 'pytorch_outputs.npz', allow_pickle=False) as archive:
        actual = {key: archive[key].astype(np.float64) for key in archive.files}
    if set(reference) != set(actual):
        raise ValueError('Saved output names differ')
    for name in reference:
        if reference[name].shape != actual[name].shape:
            raise ValueError(f'Saved output shapes differ for {name}')
        # Visuals show the first batch/channel; metrics describe every element.
        left, right = reference[name][0, 0], actual[name][0, 0]
        low, high = (0., 1.) if name == 'confidence' else _range(left, right)
        cmap = matplotlib.colormaps['viridis'].copy()
        cmap.set_bad('magenta')
        for backend, array in [('onnx', left), ('pytorch', right)]:
            rgba = cmap(Normalize(low, high)(np.ma.masked_invalid(array)))
            Image.fromarray((rgba * 255).astype(np.uint8)).save(directory / f'{backend}_{name}.png')

    pdf_path = directory / 'analysis.pdf'
    with PdfPages(pdf_path) as pdf:
        pdf.infodict()['Title'] = 'ONNX and PyTorch depth comparison'
        pdf.infodict()['Subject'] = 'Independent model output comparison; no ground-truth depth accuracy claim'
        figure = plt.figure(figsize=(11.69, 8.27))
        figure.text(.05, .94, 'ZED depth: ONNX vs PyTorch', fontsize=22, weight='bold')
        status = 'PASS: outputs within tolerance' if report['allclose'] else 'FAIL: outputs differ beyond tolerance'
        figure.text(.05, .89, status, fontsize=14, color='#176940' if report['allclose'] else '#aa2525')
        figure.text(.05, .85, f"Model: {Path(report['model']).name} | atol={report['atol']:g}, rtol={report['rtol']:g}", fontsize=10)
        for index, (key, title) in enumerate([('left_image', 'Left RGB input'), ('right_image', 'Right RGB input')]):
            axis = figure.add_axes([.05 + index * .47, .53, .43, .29])
            with Image.open(report[key]) as image:
                axis.imshow(image.convert('RGB'))
            axis.set_title(f'{title}: {Path(report[key]).name}', fontsize=11)
            axis.axis('off')
        axis = figure.add_axes([.05, .32, .90, .16])
        axis.axis('off')
        rows = []
        for name, metrics in report['outputs'].items():
            fmt = lambda value: 'n/a' if value is None else f'{value:.6g}'
            rows.append([name, fmt(metrics['mean_abs_error']), fmt(metrics['max_abs_error']),
                         fmt(metrics['rmse']), fmt(metrics['p99_abs_error']),
                         f"{metrics['fraction_outside_tolerance'] * 100:.4g}%",
                         'PASS' if metrics['allclose'] else 'FAIL'])
        table = axis.table(cellText=rows,
            colLabels=['Output', 'Mean abs.', 'Max abs.', 'RMSE', 'P99 abs.', 'Outside tol.', 'Result'],
            cellLoc='center', loc='center', colWidths=[.19, .14, .14, .13, .14, .15, .11])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.8)
        for (row, column), cell in table.get_celld().items():
            if row == 0:
                cell.set_facecolor('#e8eef5')
                cell.set_text_props(weight='bold')
        meta = report['preprocessing']
        lines = [
            f"Input HxW: {meta['original_hw']} | network HxW: {meta['network_hw']} | bottom/right padding: {meta['padding_bottom_right']}",
            f"Output shape: {report['outputs']['disparity']['shape']} | comparison includes padding; disparity is not metric depth.",
            'Reference: ONNX Runtime standard layers + independent NumPy custom operators. No PyTorch tensors are injected.',
            'disparity_right is a zero-filled placeholder; matching zeros does not validate right-view depth.',
            'This measures implementation agreement on one stereo pair. No ground truth or native ZED CUDA parity was tested.',
            f"Timing: PyTorch forward {report['pytorch_forward_seconds']:.3f}s; reference setup + run {report['onnx_reference_setup_and_run_seconds']:.3f}s.",
        ]
        figure.text(.05, .26, '\n'.join(lines), fontsize=9, va='top', linespacing=1.8)
        pdf.savefig(figure)
        plt.close(figure)

        for name in ['disparity', 'confidence']:
            left, right = reference[name][0, 0], actual[name][0, 0]
            difference = np.abs(left - right)
            finite = np.isfinite(difference)
            low, high = (0., 1.) if name == 'confidence' else _range(left, right)
            maximum = float(difference[finite].max()) if finite.any() else 0.
            figure, axes = plt.subplots(2, 2, figsize=(11.69, 8.27))
            figure.suptitle(f'{name.title()} output analysis', fontsize=20, weight='bold')
            for axis, array, title in zip(axes[0], [left, right], ['ONNX reference', 'PyTorch']):
                image = axis.imshow(np.ma.masked_invalid(array), cmap='viridis', vmin=low, vmax=high)
                axis.set_title(f'{title} | same color scale')
                axis.axis('off')
                figure.colorbar(image, ax=axis, fraction=.035, pad=.025)
            image = axes[1, 0].imshow(np.ma.masked_invalid(difference), cmap='magma', vmin=0, vmax=maximum or 1.)
            axes[1, 0].set_title(f'Absolute difference | maximum {maximum:.6g}')
            axes[1, 0].axis('off')
            figure.colorbar(image, ax=axes[1, 0], fraction=.035, pad=.025)
            valid = difference[finite]
            if valid.size:
                axes[1, 1].hist(valid, bins=60, color='#426c91')
                axes[1, 1].set_yscale('log')
            else:
                axes[1, 1].text(.5, .5, 'No finite differences', ha='center')
            axes[1, 1].set_title(f'Error distribution | nonfinite: {int((~finite).sum())}')
            axes[1, 1].set_xlabel('Absolute difference (native network units)')
            axes[1, 1].set_ylabel('Pixel count (log scale)')
            axes[1, 1].grid(alpha=.2)
            metrics = report['outputs'][name]
            figure.text(.05, .035,
                f"Out-of-tolerance: {metrics['fraction_outside_tolerance'] * 100:.6g}% | "
                'Difference colors are magnified to the maximum error; inspect numeric values, not color alone.', fontsize=9)
            figure.subplots_adjust(top=.89, bottom=.12, hspace=.35, wspace=.40)
            pdf.savefig(figure)
            plt.close(figure)
    return pdf_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', help='Directory containing comparison.json and both output NPZ files')
    print(generate_report(parser.parse_args().directory))
