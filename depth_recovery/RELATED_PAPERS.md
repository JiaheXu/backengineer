# Papers related to the recovered ZED depth network

Research date: **2026-10-13**. Target: the depth artifacts recovered from **ZED SDK 5.5.0**, particularly `neural_depth_5.3.onnx`.

The papers below share operators or architectural patterns with the recovered network. **An exact match to a published architecture has not been confirmed.** Shared components do not establish authorship, derivation, identical training, or identical weights.

## Findings at a glance

| Paper | Publication | Shared component | Strength of match |
|---|---|---|---|
| [Group-Wise Correlation Stereo Network (GwcNet)](https://arxiv.org/abs/1903.04025) | CVPR 2019 | Group-wise correlation and 3D cost processing | Same mathematical correlation operator; different complete architecture |
| [Anytime Stereo Image Depth Estimation on Mobile Devices (AnyNet)](https://arxiv.org/abs/1810.11408) | ICRA 2019 | Coarse-to-fine refinement with five local disparity offsets | Same local hypothesis set; different feature costs and refinement networks |
| [StereoNet: Guided Hierarchical Refinement for Real-Time Edge-Aware Depth Prediction](https://arxiv.org/abs/1807.08865) | ECCV 2018 | Low-resolution stereo matching followed by hierarchical refinement | Related overall strategy |
| [CFNet: Cascade and Fused Cost Volume for Robust Stereo Matching](https://arxiv.org/abs/2104.04314) | CVPR 2021 | Group-wise correlation and cascaded cost volumes | Related design; adaptive search differs from the recovered fixed offsets |

## Recovered architecture used for comparison

The local `neural_depth_5.3.onnx` graph contains:

- Two RGB inputs and 2×2 spatial packing into 12 channels.
- Shared convolutional feature extraction, including a `[64, 12, 7, 7]` initial kernel and residual blocks with 64 and 128 channels.
- A 32-group correlation cost volume with 24 disparity hypotheses.
- A spatial 3D encoder–decoder with channel stages 64 → 128 → 256 and three transposed-convolution layers.
- Softmax-based disparity estimation.
- Three local offset-volume refinement stages, each searching `−2, −1, 0, 1, 2` around the current disparity and using 2D convolutional processing.
- Confidence prediction.

The graph has 77 convolution nodes: 70 use 2D kernels and seven use 3D kernels. It also has three transposed-convolution nodes. The reconstructed inference module registers 7,432,743 learnable parameter elements.

Local evidence: [graph topology](neural_depth_5.3.graph.json), [custom operators](torch_custom_ops.py), [operator evidence notes](evidence/custom_ops_notes.txt), and [PyTorch manifest](torch_manifest.json).

## 1. GwcNet — strongest correlation-operator match

**Paper:** *Group-Wise Correlation Stereo Network*.

[CVPR publication page](https://openaccess.thecvf.com/content_CVPR_2019/html/Guo_Group-Wise_Correlation_Stereo_Network_CVPR_2019_paper.html) · [arXiv](https://arxiv.org/abs/1903.04025) · [Authors’ code](https://github.com/xy-guo/GwcNet)

The paper’s equation (3) divides feature channels into groups and averages products between left features and horizontally shifted right features. This is the same mathematical construction recovered from ZED’s `BuildGWCVolume` kernel. Both designs also use 3D convolutions to process a correlation volume.

**Differences:** the published GwcNet configuration uses 40 correlation groups and three stacked hourglass modules. The recovered ZED 5.3 graph uses 32 groups and a different 3D encoder–decoder followed by local 2D refinement. The operator match does not establish that ZED implements stock GwcNet.

## 2. AnyNet — strongest local-refinement pattern match

**Paper:** *Anytime Stereo Image Depth Estimation on Mobile Devices*.

[arXiv](https://arxiv.org/abs/1810.11408) · [PDF](https://arxiv.org/pdf/1810.11408) · [Authors’ code](https://github.com/mileyan/AnyNet)

Section III(c) describes upsampling an existing disparity estimate, warping features, and computing residual corrections using a five-entry search window: `−2, −1, 0, 1, 2`. This hypothesis set matches the recovered ZED offset-volume kernel.

**Differences:** AnyNet uses L1 feature distances, a small U-Net feature extractor, 3D convolutional residual-cost processing, and a final spatial-propagation stage. ZED’s feature widths, parameter count, and refinement networks differ. Matching offsets alone do not identify the complete architecture.

## 3. StereoNet — related hierarchical design

**Paper:** *StereoNet: Guided Hierarchical Refinement for Real-Time Edge-Aware Depth Prediction*.

[Authors’ publication page](https://research.google/pubs/stereonet-guided-hierarchical-refinement-for-edge-aware-depth-prediction/) · [arXiv](https://arxiv.org/abs/1807.08865)

StereoNet uses shared stereo feature extraction, an initial disparity estimate from a low-resolution cost volume, and learned hierarchical upsampling/refinement. This resembles the recovered network’s broad strategy.

**Limit:** this is a weaker identity match than GwcNet’s correlation operator or AnyNet’s five-offset search window. It does not establish an exact match to ZED’s offset-correlation cascade.

## 4. CFNet — related correlation and cascade design

**Paper:** *CFNet: Cascade and Fused Cost Volume for Robust Stereo Matching*.

[arXiv](https://arxiv.org/abs/2104.04314) · [PDF](https://arxiv.org/pdf/2104.04314)

CFNet combines group-wise correlation with cascaded cost-volume search and fuses multiple low-resolution cost volumes.

**Difference:** CFNet uses variance-based uncertainty to adapt subsequent disparity-search ranges. The recovered ZED 5.3 custom kernel exposes fixed local offsets. CFNet is therefore a related reference, without a demonstrated exact architecture match.

## Manufacturer attribution

Stereolabs’ [May 23, 2022 neural-depth announcement](https://www.stereolabs.com/blog/neural-depth-sensing) describes a custom CNN trained on stereo images for extraction, matching, and cost aggregation. It does not name a paper for that design. The announcement concerns the original neural-depth release and does not document the detailed architecture of the recovered SDK 5.5.0 artifacts.

## Other recovered model variants

- **Light 5.3:** one correlation group and 2D cost aggregation.
- **5.4 and 5.6:** initial product correlation is replaced by sign-bit XNOR/Hamming similarity over learned features.

The reviewed papers do not establish the exact origin of these combined designs. The name `CensusXnorCostVolume` alone is not evidence of classical spatial-neighborhood Census or a particular paper.

## Research limits

The comparison checked manufacturer material and primary sources for GwcNet, AnyNet, StereoNet, and CFNet, with additional candidate searches. It was not an exhaustive literature review.

The supported conclusion is that the recovered ZED network shares established stereo-matching building blocks. Identifying its exact provenance would require a matching detailed architecture, source topology, or manufacturer attribution.
