# Gap-CAM Implementation Instructions

This document is a **step-by-step implementation brief** for an AI agent.  
The goal is to implement **Gap-CAM**, a Grad-CAM-style visualization method for localizing the image regions that contribute to **high-gap, low-semantic embedding dimensions**.

The method is built around the same quantities already computed in the FewDeep pipeline:

- per-dimension gap score `g_j`
- normalized gap score `g_tilde_j`
- per-dimension semantic importance `s_j`
- normalized semantic importance `s_tilde_j`
- thresholds `tau_gap` and `tau_imp`

---

## 0. Objective

For an image-text pair `(x, t)`, compute a heatmap over the image showing:

> which spatial regions most increase the contribution of the dimensions that have **high modality gap** and **low semantic importance**.

This is **not** standard class Grad-CAM.

The scalar target to backpropagate is a **bad-dimension discrepancy score**, built from the same dimension-level statistics used in FewDeep.

---

# 1. Primary implementation path

At every stage below, follow the **primary path** unless the codebase makes it impossible.

## 1.1 Primary path summary

- **Encoder type**: ViT-based CLIP / OpenCLIP image encoder
- **Spatial representation**: final patch tokens before pooling / final projection
- **Dimension set for visualization**: strong bad dimensions only
- **Target scalar**: signed pair-conditioned bad-dimension target
- **Patch saliency formula**: gradient-times-activation at patch level
- **Comparison heatmap**: good-dimension target
- **Qualitative ranking**: rank samples by bad target score
- **Quantitative evaluation**: heatmap entropy, optional object focus ratio if masks exist

If ViT tokens are not accessible, use the CNN fallback described later.

---

# 2. Required inputs

The implementation assumes the following inputs can be produced.

## 2.0 Pre-defined inputs
Model to use (name, pretrained both string like in other files )
The Thresholds `tau_gap` and `tau_imp` already selected by the FewDeep scoring procedure


## 2.1 Compute embeddings Embeddings

You must have normalized image and text embeddings from the selected model:

- `X = {x_i}` image embeddings, shape `(N, D)`
- `Y = {y_i}` text embeddings, shape `(N, D)`


### Primary path
Use **L2-normalized final CLIP embeddings**.

### Fallback
If the pipeline currently stores pre-normalization embeddings only, normalize them first:
```python
z = z / (z.norm(dim=-1, keepdim=True) + 1e-8)
```

---

## 2.2 Alignment outputs

You must have the aligned subspace matrices:

- `W_X`, shape `(D, d_sub)`
- `W_Y`, shape `(D, d_sub)`

These come from the subspace alignment stage already used by FewDeep.

### Primary path
Reuse the exact `W_X`, `W_Y` already computed in the current pipeline.

### Fallback
If they are not cached, recompute them exactly as in the FewDeep implementation.


---

## 2.4 Paired samples

You need image-text pairs `(image, caption)` from the dataset.

### Primary path
Use the same paired evaluation samples used for retrieval / clustering analysis.

### Fallback
Use any valid image-caption pair from the dataset split used to compute the embedding statistics.

---

# 3. Compute the dataset-level statistics

This stage must run **once** before generating heatmaps.

---

## Step 3.1 Compute modality means

Compute the centroid of image embeddings and text embeddings:

\[
\mu_X = \frac{1}{N}\sum_{i=1}^{N} x_i
\]

\[
\mu_Y = \frac{1}{N}\sum_{i=1}^{N} y_i
\]

### Output
- `mu_X`, shape `(D,)`
- `mu_Y`, shape `(D,)`

### Primary path
Use all embeddings from the same split already used for FewDeep analysis.

---

## Step 3.2 Compute per-dimension gap

For each dimension `j`:

\[
g_j = |\mu_X^{(j)} - \mu_Y^{(j)}|
\]

### Output
- `g`, shape `(D,)`

---

## Step 3.3 Normalize the gap scores

Use min-max normalization:

\[
\tilde g_j = \frac{g_j - \min(g)}{\max(g)-\min(g)+\varepsilon}
\]

Use:

- `eps = 1e-8`

### Output
- `g_tilde`, shape `(D,)`, values in `[0, 1]`

### Primary path
Use min-max normalization exactly.

### Fallback
None. Do not replace this unless the whole FewDeep pipeline already uses a different normalization consistently.

---

## Step 3.4 Compute semantic importance per dimension

For each dimension `j`, compute image-side and text-side subspace contributions:

\[
s^X_j = \sum_{k=1}^{d_{sub}} (W_X)_{j,k}^2
\]

\[
s^Y_j = \sum_{k=1}^{d_{sub}} (W_Y)_{j,k}^2
\]

Then average them:

\[
s_j = \frac{1}{2}(s^X_j + s^Y_j)
\]

### Output
- `s_X`, shape `(D,)`
- `s_Y`, shape `(D,)`
- `s`, shape `(D,)`

---

## Step 3.5 Normalize the semantic importance

\[
\tilde s_j = \frac{s_j - \min(s)}{\max(s)-\min(s)+\varepsilon}
\]

### Output
- `s_tilde`, shape `(D,)`, values in `[0, 1]`

### Primary path
Use min-max normalization exactly.

---

# 4. Define the good and bad dimension sets

This stage determines which coordinates are semantically useful and which are candidates for gap localization.

---

## Step 4.1 Define the retained set

Define the retained dimensions exactly as in the FewDeep logic:

\[
I = \{ j \mid \tilde s_j \ge \tau_{imp} \; \land \; \tilde g_j \le \tau_{gap} \}
\]

### Output
- `good_set` or `I`

---

## Step 4.2 Define the strong bad set

For heatmaps, use only the **strong bad dimensions**:

\[
B_{strong} = \{ j \mid \tilde s_j < \tau_{imp} \; \land \; \tilde g_j > \tau_{gap} \}
\]

### Output
- `bad_set`

### Primary path
Use `B_strong` only.

### Fallback
If `B_strong` is too small, use the removed set:
\[
B = \{1,\dots,D\} \setminus I
\]

---

## Step 4.3 Optional top-K filtering (ignore now)

If the bad set is too large, keep only the top `K` dimensions ranked by:

\[
b_j = \tilde g_j (1 - \tilde s_j)
\]

### Primary path
Use `K = 32`.

### Fallback
Try `K = 16` or `K = 64`.

### Output
- `bad_set_topk`

---

# 5. Compute weights for bad and good dimensions

This stage gives stronger influence to the most meaningful dimensions.

---

## Step 5.1 Compute badness weights

For each bad dimension:

\[
b_j = \tilde g_j (1 - \tilde s_j)
\]

Normalize over the bad set:

\[
w_j = \frac{b_j}{\sum_{l \in B_{strong}} b_l + \varepsilon}
\]

### Output
- `w_bad`, shape `(len(bad_set),)`

### Primary path
Use these weights exactly.

---

## Step 5.2 Compute goodness weights

For each good dimension:

\[
a_j = (1 - \tilde g_j)\tilde s_j
\]

Normalize over the good set:

\[
u_j = \frac{a_j}{\sum_{l \in I} a_l + \varepsilon}
\]

### Output
- `w_good`, shape `(len(good_set),)`

### Primary path
Use these weights exactly.

---

# 6. Hook the image encoder

This is the stage where the implementation becomes model-specific.

---

## Step 6.1 Detect encoder family

The agent must determine whether the image encoder is:

1. **ViT-based**
2. **CNN-based**

### Primary path
Assume **ViT-based CLIP/OpenCLIP** unless the code clearly uses a CNN.

---

## Step 6.2 Primary path for ViT

Hook the **final patch-token tensor before pooling and before the final image embedding projection if possible**, this means that you have to encode the pairs, hence construct first the dataset.

You need a tensor:

- `T`, shape `(B, P, C)`
- where `P` is number of patch tokens
- and CLS token is excluded

### Required
Store both:
- forward activations `T`
- backward gradients `dL/dT`

### Primary path
Hook the **last transformer block output before token pooling**.

### Important
Do **not** hook after CLS pooling or after the final global embedding only, because spatial information is already gone there.

---

## Step 6.3 Fallback path for CNN

Hook the final convolutional feature map:

- `A`, shape `(B, K, H, W)`

Store:
- forward activations `A`
- backward gradients `dL/dA`

### Fallback only
Use this path only if ViT patch tokens are not available.

---

# 7. Define the scalar target to backpropagate

This is the most important stage.

The target must be a scalar that expresses the contribution of the image to the bad dimensions.

---

## Step 7.1 Compute sample embeddings

For each image-text pair `(x, t)`, compute normalized embeddings:

- `z_I(x)`, shape `(D,)`
- `z_T(t)`, shape `(D,)`

### Primary path
Always use normalized embeddings:
\[
z = \frac{z}{\|z\|_2 + \varepsilon}
\]

---

## Step 7.2 Compute the global gap direction per dimension

For every dimension:

\[
\delta_j = \mu_X^{(j)} - \mu_Y^{(j)}
\]

### Output
- `delta`, shape `(D,)`

---

## Step 7.3 Define the primary bad target

For each bad dimension `j`, define:

\[
\phi_j(x,t) =
\operatorname{ReLU}
\Big(
\operatorname{sign}(\delta_j)(z^I_j(x)-z^T_j(t))
\Big)
\]

Then define the scalar target:

\[
L_{bad}(x,t) =
\sum_{j \in B_{strong}} w_j \, \phi_j(x,t)
\]

### Primary path
Use this target exactly.

### Meaning
This target is high when the sample pushes image and text apart along the globally bad dimensions in the same direction as the global modality gap.

---

## Step 7.4 Define the fallback bad target

If the signed target is unstable or returns zero too often, use:

\[
L_{bad,abs}(x,t) =
\sum_{j \in B_{strong}} w_j |z^I_j(x)-z^T_j(t)|
\]

### Fallback
Use this only for debugging or if the primary target produces degenerate maps.

---

## Step 7.5 Define the good target

For comparison, compute a good-dimension target:

\[
L_{good}(x,t) =
\sum_{j \in I} u_j \, z^I_j(x) z^T_j(t)
\]

### Primary path
Always implement this target too.

### Why
This gives a control heatmap showing where semantically aligned dimensions are supported.

---

# 8. Compute the heatmap for ViT (primary path)

This is the main implementation route.

---

## Step 8.1 Forward pass

For one image-text pair:
1. run the image through the image encoder
2. capture patch activations `T`
3. run the text through the text encoder
4. compute normalized embeddings `z_I`, `z_T`
5. compute `L_bad`

### Required
Call `backward()` on `L_bad`.

---

## Step 8.2 Get gradients w.r.t. patch tokens

After backpropagation, get:

- `T`, shape `(P, C)`
- `G = dL_bad/dT`, shape `(P, C)`

---

## Step 8.3 Compute patch saliency

### Primary path
Use gradient-times-activation at patch level:

\[
M_{bad}(p) =
\operatorname{ReLU}
\left(
\sum_{c=1}^{C} G_{p,c} T_{p,c}
\right)
\]

This gives one score per patch.

### Output
- `M_bad_patch`, shape `(P,)`

### Why this is primary
It is sharper and usually more stable than mean-gradient-only weighting for ViTs.

---

## Step 8.4 Reshape patch map to 2D grid

If the image has `P = h * w` patches:

\[
M_{bad}^{grid} \in \mathbb{R}^{h \times w}
\]

Example:
- ViT-B/32 with 224x224 input often gives `7 x 7`

### Output
- `M_bad_grid`

---

## Step 8.5 Normalize patch heatmap

Use min-max normalization:

\[
M \leftarrow \frac{M - \min(M)}{\max(M)-\min(M)+\varepsilon}
\]

### Output
- `M_bad_norm`, values in `[0,1]`

---

## Step 8.6 Upsample to image resolution

Resize the patch grid to the original image resolution using bilinear interpolation.

### Output
- `M_bad_img`, shape `(H_img, W_img)`

---

## Step 8.7 Repeat for good target

Run the same pipeline using `L_good`:

1. zero gradients
2. recompute forward if needed
3. compute `L_good`
4. backward
5. compute patch saliency
6. normalize
7. upsample

### Output
- `M_good_img`

---

## Step 8.8 Compute difference map

\[
M_{\Delta} = M_{bad} - M_{good}
\]

Optional display normalization may be applied afterward.

### Output
- `M_diff_img`

---

# 9. Compute the heatmap for CNN (fallback path)

Use this only when the image encoder does not expose patch tokens.

---

## Step 9.1 Forward pass

Get:
- final conv activations `A`, shape `(K, H, W)`
- scalar target `L_bad`

Call `backward()` on `L_bad`.

---

## Step 9.2 Compute channel weights

\[
\alpha_k = \frac{1}{HW} \sum_{u=1}^{H}\sum_{v=1}^{W}
\frac{\partial L_{bad}}{\partial A^k_{u,v}}
\]

### Output
- `alpha`, shape `(K,)`

---

## Step 9.3 Compute Grad-CAM map

\[
M_{bad}(u,v) =
\operatorname{ReLU}
\left(
\sum_{k=1}^{K} \alpha_k A^k_{u,v}
\right)
\]

### Output
- `M_bad`, shape `(H, W)`

---

## Step 9.4 Normalize and upsample

Normalize with min-max and resize to input image size.

Repeat for `L_good`.

---

# 10. Save the visual outputs

For every selected sample, save the following outputs.

---

## Step 10.1 Required outputs per sample

1. original image
2. bad heatmap
3. good heatmap
4. difference heatmap
5. image with bad overlay
6. image with good overlay
7. image with difference overlay

### Primary path
Always save all seven outputs.

---

## Step 10.2 Metadata to save

For each sample also save:
- sample id
- caption text
- `L_bad`
- `L_good`
- top contributing bad dimensions
- top contributing good dimensions

### Primary path
Store these in a CSV or JSON file.

---

# 11. Rank samples for qualitative visualization

Do not choose examples randomly.

---

## Step 11.1 Rank by bad score

For each sample compute:

\[
q(x,t) = L_{bad}(x,t)
\]

Sort descending.

### Primary path
Use the top-ranked samples for visualization.

---

## Step 11.2 Diversity filter

If possible, avoid choosing many near-duplicate examples from the same pseudo-class.

### Primary path
Select top examples with category diversity.

### Fallback
If category labels are unavailable, use only score ranking.

---

# 12. Aggregate heatmaps by category

This is important to avoid cherry-picking.

---

## Step 12.1 Normalize each heatmap to sum 1

For each sample heatmap `M_i`:

\[
\hat M_i = \frac{M_i}{\sum_{u,v} M_i(u,v)+\varepsilon}
\]

---

## Step 12.2 Average within category

For category `c`:

\[
\bar M^{(c)}_{bad} = \frac{1}{|D_c|}\sum_{i \in D_c}\hat M_i
\]

Do the same for good heatmaps.

### Output
- average bad heatmap per category
- average good heatmap per category

### Primary path
Use the pseudo-labels already used in clustering analysis.

### Fallback
If labels are not available, aggregate over the whole dataset only.

---

# 13. Quantitative evaluation of the heatmaps

At least one scalar evaluation must be implemented.

---

## Step 13.1 Primary quantitative metric: entropy

Normalize heatmap to sum 1:

\[
\hat M(u,v) = \frac{M(u,v)}{\sum_{u,v} M(u,v)+\varepsilon}
\]

Then compute entropy:

\[
H(M) = -\sum_{u,v} \hat M(u,v)\log(\hat M(u,v)+\varepsilon)
\]

### Interpretation
- lower entropy = more concentrated
- higher entropy = more diffuse

### Primary path
Compare entropy of bad and good heatmaps.

Expected:
- bad heatmaps often more diffuse or background-like
- good heatmaps often more concentrated

---

## Step 13.2 Optional quantitative metric: object focus ratio

If bounding boxes or masks are available, define object region `Omega` and compute:

\[
FocusRatio(M, \Omega) =
\frac{\sum_{(u,v)\in\Omega} M(u,v)}
{\sum_{u,v} M(u,v)+\varepsilon}
\]

### Primary path
Use this only if object boxes or masks are easily available.

Expected:
- `FocusRatio(good) > FocusRatio(bad)`

---

# 14. Required code modules

The agent should create these modules.

---

## 14.1 `compute_gap_semantic_scores.py`

### Responsibilities
- load normalized embeddings
- compute `mu_X`, `mu_Y`
- compute `g`, `g_tilde`
- load or compute `W_X`, `W_Y`
- compute `s`, `s_tilde`
- compute good / bad sets
- compute weights
- save all outputs

### Save
Use `.npz`, `.pt`, or `.json` depending on the existing codebase.

---

## 14.2 `gap_cam_core.py`

### Responsibilities
- model hooks
- ViT patch-token extraction
- CNN feature extraction fallback
- target functions `L_bad`, `L_bad_abs`, `L_good`
- heatmap generation functions
- normalization and overlay utilities

---

## 14.3 `run_gap_cam.py`

### Responsibilities
- load model
- load precomputed dimension statistics
- iterate through selected image-text pairs
- compute bad / good / difference heatmaps
- save visual outputs
- save metadata

---

## 14.4 `aggregate_gap_cam.py`

### Responsibilities
- rank samples by `L_bad`
- aggregate heatmaps by category
- compute entropy and optional focus ratio
- export CSV summaries
- save category-level figures

---

# 15. Function-level specification

Below is the recommended function structure.

---

## 15.1 Statistics functions

```python
def compute_modality_means(X: torch.Tensor, Y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    pass

def compute_gap_scores(mu_X: torch.Tensor, mu_Y: torch.Tensor) -> torch.Tensor:
    pass

def compute_semantic_scores(W_X: torch.Tensor, W_Y: torch.Tensor) -> torch.Tensor:
    pass

def minmax_normalize(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    pass

def build_dimension_sets(
    g_tilde: torch.Tensor,
    s_tilde: torch.Tensor,
    tau_gap: float,
    tau_imp: float,
    topk_bad: int | None = 32,
):
    pass
```

---

## 15.2 Target functions

```python
def compute_bad_target(
    z_img: torch.Tensor,
    z_txt: torch.Tensor,
    mu_X: torch.Tensor,
    mu_Y: torch.Tensor,
    bad_indices: torch.Tensor,
    bad_weights: torch.Tensor,
) -> torch.Tensor:
    pass

def compute_bad_target_abs(
    z_img: torch.Tensor,
    z_txt: torch.Tensor,
    bad_indices: torch.Tensor,
    bad_weights: torch.Tensor,
) -> torch.Tensor:
    pass

def compute_good_target(
    z_img: torch.Tensor,
    z_txt: torch.Tensor,
    good_indices: torch.Tensor,
    good_weights: torch.Tensor,
) -> torch.Tensor:
    pass
```

---

## 15.3 ViT heatmap functions

```python
def compute_vit_heatmap_from_target(
    model,
    image_tensor: torch.Tensor,
    text_tensor,
    target_fn,
    hook_module_name: str,
) -> torch.Tensor:
    pass
```

### Required internal logic
- register forward hook
- register backward hook
- run forward
- compute scalar target
- backward
- get activations and gradients
- compute patch saliency
- reshape grid
- normalize
- upsample

---

## 15.4 CNN heatmap functions

```python
def compute_cnn_heatmap_from_target(
    model,
    image_tensor: torch.Tensor,
    text_tensor,
    target_fn,
    hook_module_name: str,
) -> torch.Tensor:
    pass
```

---

# 16. Step-by-step execution order for the agent

This is the exact order the agent should follow.

---

## Stage A. Precompute dimension statistics

1. Load normalized image embeddings `X`
2. Load normalized text embeddings `Y`
3. Compute `mu_X`, `mu_Y`
4. Compute `g`
5. Normalize to `g_tilde`
6. Load `W_X`, `W_Y`
7. Compute `s`
8. Normalize to `s_tilde`
9. Build `good_set`
10. Build `bad_set = B_strong`
11. Apply top-K filtering to bad set if needed
12. Compute `w_bad`
13. Compute `w_good`
14. Save all outputs

### Primary path
Use this stage exactly.

---

## Stage B. Prepare model hooks

15. Load the CLIP / OpenCLIP model
16. Determine whether the image encoder is ViT or CNN
17. If ViT, identify the final patch-token module
18. Register forward and backward hooks
19. Validate tensor shapes with one sample

### Primary path
Use ViT patch tokens.

---

## Stage C. Generate heatmaps for one sample

20. Load image and caption
21. Preprocess both
22. Forward pass
23. Compute normalized embeddings `z_I`, `z_T`
24. Compute `L_bad`
25. Backpropagate
26. Compute `M_bad`
27. Clear gradients
28. Forward again if necessary
29. Compute `L_good`
30. Backpropagate
31. Compute `M_good`
32. Compute `M_diff = M_bad - M_good`
33. Save raw maps
34. Save overlays
35. Save metadata

### Primary path
Use signed bad target, good target, and ViT patch heatmap.

---

## Stage D. Batch processing

36. Run Stage C over all selected samples
37. Compute `L_bad` score per sample
38. Rank samples by `L_bad`
39. Save top examples

---

## Stage E. Aggregate and evaluate

40. Normalize each heatmap to sum 1
41. Aggregate by category
42. Compute entropy
43. Compute optional focus ratio if masks or boxes exist
44. Save summary CSV
45. Save aggregate figures

---

# 17. Failure modes and fixes

The agent must explicitly check for the following problems.

---

## 17.1 Heatmap all zeros

### Possible cause
The signed target is suppressed by ReLU.

### Fix
Use:
\[
L_{bad,abs}(x,t)
=
\sum_{j \in B_{strong}} w_j |z^I_j-z^T_j|
\]

### Secondary checks
- verify embeddings are normalized
- verify `bad_set` is not empty
- verify gradients are not detached

---

## 17.2 Heatmap too diffuse

### Possible cause
Too many bad dimensions are included.

### Fix
Reduce `K` in top-K filtering:
- try `K = 16`

### Secondary fix
Hook one layer earlier in the ViT.

---

## 17.3 Identical heatmaps across samples

### Possible cause
The implementation is using only dataset-level weights and not the sample-specific discrepancy.

### Fix
Verify the target contains:
- `z_I(x)`
- `z_T(t)`

and not only constant weights.

---

## 17.4 No spatial tokens in ViT hook

### Possible cause
The hook is registered after CLS pooling or after the final image embedding projection.

### Fix
Move the hook to the last transformer block output before pooling.

---

# 18. Minimal equations block for the agent

These are the core equations that must be implemented exactly.

## Dataset-level scores

\[
g_j = |\mu_X^{(j)} - \mu_Y^{(j)}|
\]

\[
\tilde g_j = \operatorname{minmax}(g_j)
\]

\[
s_j = \frac{1}{2}\left(\sum_k (W_X)_{j,k}^2 + \sum_k (W_Y)_{j,k}^2\right)
\]

\[
\tilde s_j = \operatorname{minmax}(s_j)
\]

## Dimension sets

\[
I = \{ j \mid \tilde s_j \ge \tau_{imp} \land \tilde g_j \le \tau_{gap} \}
\]

\[
B_{strong} = \{ j \mid \tilde s_j < \tau_{imp} \land \tilde g_j > \tau_{gap} \}
\]

## Weights

\[
w_j \propto \tilde g_j (1-\tilde s_j)
\]

\[
u_j \propto (1-\tilde g_j)\tilde s_j
\]

## Primary bad target

\[
\delta_j = \mu_X^{(j)} - \mu_Y^{(j)}
\]

\[
L_{bad}(x,t) =
\sum_{j \in B_{strong}} w_j
\operatorname{ReLU}
\Big(
\operatorname{sign}(\delta_j)(z^I_j(x)-z^T_j(t))
\Big)
\]

## Good target

\[
L_{good}(x,t) =
\sum_{j \in I} u_j \, z^I_j(x) z^T_j(t)
\]

## ViT patch heatmap

\[
M_{bad}(p) =
\operatorname{ReLU}
\left(
\sum_{c=1}^{C}
\frac{\partial L_{bad}}{\partial T_{p,c}} T_{p,c}
\right)
\]

## CNN fallback heatmap

\[
\alpha_k = \frac{1}{HW}\sum_{u,v}\frac{\partial L_{bad}}{\partial A^k_{u,v}}
\]

\[
M_{bad}(u,v) =
\operatorname{ReLU}
\left(
\sum_k \alpha_k A^k_{u,v}
\right)
\]

---

# 19. Final default choices to implement first

If the agent must choose one configuration without asking questions, use exactly this setup:

- use normalized CLIP embeddings
- use existing FewDeep `W_X`, `W_Y`, `tau_gap`, `tau_imp`
- define bad dimensions with `B_strong`
- keep top `K = 32` bad dimensions by `g_tilde * (1 - s_tilde)`
- use the signed pair-conditioned bad target
- use ViT patch tokens before pooling
- use patch-level gradient-times-activation
- compute bad, good, and difference heatmaps
- rank samples by `L_bad`
- compute entropy as the default quantitative metric
- save per-sample and category-level outputs

---

# 20. Deliverables expected from the agent

The completed implementation should produce:

1. a file containing precomputed dimension statistics
2. a reusable Gap-CAM core module
3. scripts to generate per-sample visualizations
4. aggregate class-level heatmaps
5. CSV summaries with scores and metadata
6. saved overlays for bad, good, and difference maps

---

End of implementation brief.
