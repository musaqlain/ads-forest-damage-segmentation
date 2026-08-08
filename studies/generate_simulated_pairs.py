"""
generate_simulated_pairs.py
===========================
Steps 2 & 3 of the ADS polygon realignment roadmap:
  - Step 2: Generate simulated dead-tree annotations and perturb them
  - Step 3: Train AffineRegistrationNet to recover those perturbations

Self-supervised pipeline:
  1. Generate synthetic irregular blob masks  ->  "fixed" (ground-truth annotation)
  2. Apply a known random affine perturbation ->  "moving" (misaligned survey polygon)
  3. Stack [fixed, moving] as a 2-channel input
  4. Train a network to predict the affine that was applied
  5. Visualise + QUANTIFY: fixed | moving | recovered

Perturbation scope (baby-step, per Ben's guidance):
  - Translation up to 30 px  (~13% of the 224px canvas)
  - Rotation    up to  5 deg
  - No scale, no shear

-------------------------------------------------------------------------------
WHAT CHANGED vs the first version (and why it matters)
-------------------------------------------------------------------------------
The first version reported a single MSE on the 6 raw affine numbers and showed
it going down ~84%. That number is *misleading*: it is dominated by translation
and hides the fact that rotation is barely recovered at all. The fixes below
make the experiment honest and interpretable, which is exactly what Step 4
("which perturbations can be recovered?") needs.

  1. Geometrically meaningful metrics: corner / reprojection error in PIXELS,
     translation error in PIXELS, rotation error in DEGREES, and mask IoU/Dice
     between the recovered and the fixed mask. (Raw-matrix MSE is not a unit you
     can reason about; "the polygon is off by 4 px" is.)
  2. Reprojection ("corner") loss option that weights every degree of freedom by
     its real pixel effect, instead of letting the large-magnitude translation
     entries dominate the gradient.
  3. Held-out quantitative evaluation over many pairs (not just 4 pictures),
     plus an "identity baseline" (= do nothing) so we can see how much the
     network actually improves over leaving the polygon where it is.
  4. Reproducibility fix: the perturbation RNG is now tied to each dataset's
     seed. Previously a single module-global displacer with an unseeded RNG was
     shared by every dataset, so `seed=` only controlled blob shapes and runs
     were not reproducible.
"""

from __future__ import annotations

import math
import random

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset

from augmentation import SyntheticDisplacer

# -- reproducibility ----------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# -- global canvas size -------------------------------------------------------
H, W = 224, 224


# ===============================================================================
# A  SYNTHETIC MASK GENERATOR
# ===============================================================================

def _random_blob(
    canvas_h: int,
    canvas_w: int,
    rng: np.random.Generator,
    min_radius: float = 20.0,
    max_radius: float = 60.0,
    n_vertices: int = 12,
) -> list[tuple[float, float]]:
    margin = max_radius + 5
    cx = rng.uniform(margin, canvas_w - margin)
    cy = rng.uniform(margin, canvas_h - margin)

    # Sorted random angles (ensures a non-self-intersecting polygon)
    angles = np.sort(rng.uniform(0, 2 * math.pi, n_vertices))
    # Independent random radii per vertex -> irregular shape
    radii = rng.uniform(min_radius, max_radius, n_vertices)

    vertices = [
        (cx + r * math.cos(a), cy + r * math.sin(a))
        for r, a in zip(radii, angles)
    ]
    return vertices


def generate_random_blob_mask(
    H: int = 224,
    W: int = 224,
    n_blobs: int | None = None,
    rng: np.random.Generator | None = None,
    soft_sigma: float = 0.0,
) -> np.ndarray:
    """Draw 2-4 irregular polygons onto a black canvas -> float32 mask in [0,1].

    soft_sigma > 0 applies a Gaussian blur so the boundary becomes a gradient
    a few pixels wide. A hard binary mask has zero gradient everywhere except
    exactly on the boundary; a soft edge gives the network a richer, smoother
    signal (this is the same trick used by transforms.rasterize_polygon_soft).
    For the baby-step experiment we keep it hard by default.
    """
    if rng is None:
        rng = np.random.default_rng()
    if n_blobs is None:
        n_blobs = int(rng.integers(2, 5))   # 2, 3, or 4

    img = Image.new("L", (W, H), color=0)   # 8-bit greyscale, start black
    draw = ImageDraw.Draw(img)
    for _ in range(n_blobs):
        draw.polygon(_random_blob(H, W, rng), fill=255)

    mask = np.array(img, dtype=np.float32) / 255.0   # -> {0.0, 1.0}

    if soft_sigma > 0.0:
        from scipy.ndimage import gaussian_filter
        mask = np.clip(gaussian_filter(mask, sigma=soft_sigma), 0.0, 1.0)
    return mask


# ===============================================================================
# B  AFFINE METRICS  (the part that makes the experiment honest)
# ===============================================================================
#
# The network outputs a flat (6,) affine in STN-normalised coordinates, laid out
# as [a, b, tx, c, d, ty] meaning:
#         x' = a*x + b*y + tx
#         y' = c*x + d*y + ty
# where (x, y) live in [-1, 1] (the affine_grid coordinate system). One
# normalised unit equals W/2 pixels in x and H/2 pixels in y.

def reference_grid_points(n: int = 3, device: str = "cpu") -> torch.Tensor:
    """An (3, n*n) homogeneous grid of points spanning normalised [-1, 1]^2.

    These act as "virtual corners". Spreading points to the edges is what makes
    rotation/scale/shear produce a large, learnable signal: a 5 deg rotation
    barely moves the centre but moves a corner by ~r*theta.
    """
    xs = torch.linspace(-1.0, 1.0, n)
    ys = torch.linspace(-1.0, 1.0, n)
    gx, gy = torch.meshgrid(xs, ys, indexing="xy")
    pts = torch.stack([gx.reshape(-1), gy.reshape(-1), torch.ones(n * n)], dim=0)
    return pts.to(device)   # (3, n*n)


def _apply_affine_to_points(flat: torch.Tensor, pts: torch.Tensor) -> torch.Tensor:
    """flat: (B, 6) normalised affine, pts: (3, P) homogeneous -> (B, 2, P)."""
    theta = flat.view(-1, 2, 3)        # (B, 2, 3)
    return theta @ pts                  # (B, 2, P)


def reprojection_loss(pred_flat: torch.Tensor, gt_flat: torch.Tensor,
                      pts: torch.Tensor) -> torch.Tensor:
    """Mean squared point-displacement after applying pred vs GT affine.

    This is the "corner loss" from deep homography estimation (DeTone et al.,
    2016). Because it is measured in coordinate units, every degree of freedom
    is weighted by how much it actually moves pixels -- so rotation is no longer
    drowned out by translation the way raw-parameter MSE drowns it.
    """
    pp = _apply_affine_to_points(pred_flat, pts)
    gp = _apply_affine_to_points(gt_flat, pts)
    return ((pp - gp) ** 2).mean()


@torch.no_grad()
def corner_error_px(pred_flat: torch.Tensor, gt_flat: torch.Tensor,
                    pts: torch.Tensor, W: int = W) -> float:
    """Mean Euclidean point displacement after pred vs GT affine, in PIXELS.

    This is the single most interpretable accuracy number: "on average the
    transformed boundary lands X pixels away from where it should."
    """
    pp = _apply_affine_to_points(pred_flat, pts)          # (B, 2, P)
    gp = _apply_affine_to_points(gt_flat, pts)
    dist_norm = torch.linalg.norm(pp - gp, dim=1)         # (B, P) in norm units
    return float(dist_norm.mean().item() * (W / 2.0))     # -> pixels


@torch.no_grad()
def translation_error_px(pred_flat: torch.Tensor, gt_flat: torch.Tensor,
                         W: int = W, H: int = H) -> float:
    """|pred translation - GT translation| in pixels (tx, ty entries only)."""
    dtx = (pred_flat[:, 2] - gt_flat[:, 2]).abs() * (W / 2.0)
    dty = (pred_flat[:, 5] - gt_flat[:, 5]).abs() * (H / 2.0)
    return float(((dtx + dty) / 2.0).mean().item())


@torch.no_grad()
def rotation_error_deg(pred_flat: torch.Tensor, gt_flat: torch.Tensor) -> float:
    """Rotation error in degrees.

    Valid when the transform is a similarity (scale~1, shear~0), which is the
    baby-step regime here: angle = atan2(c, a). For scale/shear regimes prefer
    corner_error_px, which is convention-free.
    """
    pred_ang = torch.atan2(pred_flat[:, 3], pred_flat[:, 0])   # atan2(c, a)
    gt_ang = torch.atan2(gt_flat[:, 3], gt_flat[:, 0])
    diff = (pred_ang - gt_ang + math.pi) % (2 * math.pi) - math.pi
    return float(diff.abs().mean().item() * 180.0 / math.pi)


@torch.no_grad()
def scale_error(pred_flat: torch.Tensor, gt_flat: torch.Tensor) -> float:
    """Mean abs error on the diagonal linear entries (a=sx, d=sy). Interpretable in the ISOLATED-scale
    regime, where the GT linear part is diag(sx, sy): '0.03' means the scale factor is off by 3%."""
    da = (pred_flat[:, 0] - gt_flat[:, 0]).abs()
    dd = (pred_flat[:, 4] - gt_flat[:, 4]).abs()
    return float(((da + dd) / 2.0).mean().item())


@torch.no_grad()
def shear_error_deg(pred_flat: torch.Tensor, gt_flat: torch.Tensor) -> float:
    """Shear-angle error in degrees from the b entry (A[0,1]=tan(phi)). Interpretable in the
    ISOLATED-shear regime, where the GT linear part is [[1, tan(phi)], [0, 1]]."""
    dp = torch.atan(pred_flat[:, 1]) - torch.atan(gt_flat[:, 1])
    return float(dp.abs().mean().item() * 180.0 / math.pi)


@torch.no_grad()
def mask_iou(a: torch.Tensor, b: torch.Tensor, thr: float = 0.5) -> float:
    """IoU between two soft masks after thresholding."""
    ab = (a > thr)
    bb = (b > thr)
    inter = (ab & bb).sum().item()
    union = (ab | bb).sum().item()
    return inter / union if union > 0 else 1.0


# ===============================================================================
# C  PERTURBATION PIPELINE  ->  SimulatedPairDataset
# ===============================================================================

class SimulatedPairDataset(Dataset):
    """Generates (stacked[fixed, moving], gt_affine) pairs on the fly.

    Reproducibility: ONE seeded RNG drives both the blob shapes and the
    perturbations, so a given `seed` fully determines the dataset. (The previous
    version shared a single unseeded module-global displacer across all
    datasets, so only the blob shapes were reproducible.)
    """

    def __init__(
        self,
        length: int = 2000,
        H: int = H,
        W: int = W,
        seed: int = 0,
        max_translation_px: float = 30.0,
        max_rotation_deg: float = 5.0,
        scale_range: tuple[float, float] = (1.0, 1.0),
        max_shear_deg: float = 0.0,
        soft_sigma: float = 0.0,
    ) -> None:
        self.length = length
        self.H = H
        self.W = W
        self.soft_sigma = soft_sigma
        self.rng = np.random.default_rng(seed)
        # The displacer shares this dataset's RNG -> perturbations are reproducible
        self.displacer = SyntheticDisplacer(
            max_translation_px=max_translation_px,
            max_rotation_deg=max_rotation_deg,
            scale_range=scale_range,
            max_shear_deg=max_shear_deg,
            rng=self.rng,
        )

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # -- fixed mask (the "ground-truth annotation") ------------------------
        mask_np = generate_random_blob_mask(
            self.H, self.W, rng=self.rng, soft_sigma=self.soft_sigma
        )
        fixed = torch.from_numpy(mask_np).unsqueeze(0)            # (1, H, W)

        # -- known random affine in pixel space --------------------------------
        affine_px = self.displacer.generate_random_affine()      # (2, 3)

        # -- moving mask = fixed warped by that affine -------------------------
        moving = self.displacer.displace_mask_tensor(fixed, affine_px)  # (1,H,W)

        # -- 2-channel input ---------------------------------------------------
        stacked = torch.cat([fixed, moving], dim=0)              # (2, H, W)

        # -- GT affine in normalised space, flat (6,) --------------------------
        gt_norm = self.displacer.to_normalized_affine(affine_px, self.H, self.W)
        gt_flat = gt_norm.flatten().float()                      # (6,)

        return stacked, gt_flat


# ===============================================================================
# D  REGISTRATION NETWORK  (Ben's AffineRegistrationNet)
# ===============================================================================

class AffineRegistrationNet(nn.Module):
    """2-channel (fixed, moving) -> 6 affine parameters.

    NOTE (concept worth understanding): the AdaptiveAvgPool2d((4,4)) is a strong
    spatial averaging step, and average pooling is *approximately translation
    invariant* -- it deliberately discards "where" things are. That is in slight
    tension with regressing translation. It works here because the two channels
    are stacked, so the conv layers can compute *relative* offset features before
    the pool; but it does cap how precisely position can be recovered. Keep this
    in mind for Step 4 (see "translation equivariance" in the concepts list).
    """

    def __init__(self) -> None:
        super().__init__()
        self.localization = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )
        self.fc_loc = nn.Linear(64 * 4 * 4, 6)

        # Identity initialisation -> network starts by predicting "no transform"
        self.fc_loc.weight.data.zero_()
        self.fc_loc.bias.data.copy_(
            torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=torch.float32)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc_loc(self.localization(x))


# ===============================================================================
# E  TRAINING LOOP
# ===============================================================================

def train(
    n_steps: int = 400,
    batch_size: int = 32,
    lr: float = 1e-3,
    loss_kind: str = "reproj",      # "reproj" (recommended) or "params"
    log_every: int = 25,
    device: str = "cpu",
    dataset_kwargs: dict | None = None,
) -> tuple[AffineRegistrationNet, dict]:
    print(f"\n{'-'*64}")
    print(f"  Training AffineRegistrationNet")
    print(f"  steps={n_steps}  batch={batch_size}  lr={lr}  loss={loss_kind}  device={device}")
    print(f"{'-'*64}")

    dataset_kwargs = dataset_kwargs or {}
    dataset = SimulatedPairDataset(length=n_steps * batch_size, seed=SEED, **dataset_kwargs)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=(device == "cuda"),
    )

    model = AffineRegistrationNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    pts = reference_grid_points(n=3, device=device)
    param_mse = nn.MSELoss()

    history = {"loss": [], "corner_px": [], "trans_px": [], "rot_deg": []}
    loader_iter = iter(loader)

    for step in range(1, n_steps + 1):
        stacked, gt_flat = next(loader_iter)
        stacked, gt_flat = stacked.to(device), gt_flat.to(device)

        optimizer.zero_grad()
        pred_flat = model(stacked)                       # (B, 6)
        if loss_kind == "reproj":
            loss = reprojection_loss(pred_flat, gt_flat, pts)
        else:
            loss = param_mse(pred_flat, gt_flat)
        loss.backward()
        optimizer.step()

        history["loss"].append(loss.item())
        history["corner_px"].append(corner_error_px(pred_flat, gt_flat, pts))
        history["trans_px"].append(translation_error_px(pred_flat, gt_flat))
        history["rot_deg"].append(rotation_error_deg(pred_flat, gt_flat))

        if step % log_every == 0 or step == 1:
            print(f"  step {step:>4}/{n_steps}  loss={loss.item():.5f}  |  "
                  f"corner={history['corner_px'][-1]:5.2f}px   "
                  f"trans={history['trans_px'][-1]:5.2f}px   "
                  f"rot={history['rot_deg'][-1]:4.2f}deg")

    print(f"{'-'*64}")
    print(f"  Final:  corner={history['corner_px'][-1]:.2f}px  "
          f"trans={history['trans_px'][-1]:.2f}px  rot={history['rot_deg'][-1]:.2f}deg")
    print(f"{'-'*64}\n")
    return model, history


# ===============================================================================
# F  HELD-OUT QUANTITATIVE EVALUATION  (numbers, not just pictures)
# ===============================================================================

def _invert_affine_flat(pred_flat: torch.Tensor) -> torch.Tensor:
    """Invert a (6,) normalised affine -> (1, 2, 3) theta for grid_sample.

    The moving mask was produced as moving = warp(fixed, theta). To recover the
    fixed mask we warp the moving mask by the INVERSE theta.
    """
    theta = pred_flat.view(2, 3)
    A, t = theta[:, :2], theta[:, 2:3]
    det = (A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]).clamp(min=1e-8)
    A_inv = torch.stack([
        torch.stack([A[1, 1], -A[0, 1]]),
        torch.stack([-A[1, 0], A[0, 0]]),
    ]) / det
    t_inv = -(A_inv @ t)
    return torch.cat([A_inv, t_inv], dim=1).unsqueeze(0)       # (1, 2, 3)


@torch.no_grad()
def recover_mask(moving: torch.Tensor, pred_flat: torch.Tensor) -> torch.Tensor:
    """Warp a (1,H,W) moving mask back by the inverse of the predicted affine."""
    theta_inv = _invert_affine_flat(pred_flat).to(moving.dtype)
    moving_4d = moving.unsqueeze(0)                            # (1,1,H,W)
    grid = F.affine_grid(theta_inv, moving_4d.size(), align_corners=False)
    rec = F.grid_sample(moving_4d, grid, align_corners=False, padding_mode="zeros")
    return rec.squeeze(0)                                      # (1,H,W)


@torch.no_grad()
def evaluate(model: AffineRegistrationNet, n_pairs: int = 200,
             seed: int = 999, device: str = "cpu",
             dataset_kwargs: dict | None = None) -> dict:
    """Mean recovery metrics over many unseen pairs, vs an identity baseline.

    The identity baseline = "predict no transform / leave the polygon where it
    is". Beating it is the bar: if the network cannot beat 'do nothing' for a
    perturbation type, that type is effectively not being recovered.
    """
    model.eval()
    ds = SimulatedPairDataset(length=n_pairs, seed=seed, **(dataset_kwargs or {}))
    pts = reference_grid_points(n=3, device=device)
    identity = torch.tensor([1., 0., 0., 0., 1., 0.], device=device).unsqueeze(0)

    agg = {k: 0.0 for k in
           ["corner_px", "trans_px", "rot_deg", "iou_recovered",
            "corner_px_baseline", "iou_moving", "scale_err", "shear_err"]}

    for i in range(n_pairs):
        stacked, gt_flat = ds[i]
        stacked = stacked.unsqueeze(0).to(device)
        gt = gt_flat.unsqueeze(0).to(device)
        pred = model(stacked)

        agg["corner_px"] += corner_error_px(pred, gt, pts)
        agg["trans_px"] += translation_error_px(pred, gt)
        agg["rot_deg"] += rotation_error_deg(pred, gt)
        agg["scale_err"] += scale_error(pred, gt)             # meaningful in the scale regime
        agg["shear_err"] += shear_error_deg(pred, gt)         # meaningful in the shear regime
        agg["corner_px_baseline"] += corner_error_px(identity, gt, pts)

        fixed = stacked[0, 0:1]
        moving = stacked[0, 1:2]
        recovered = recover_mask(moving, pred.squeeze(0))
        agg["iou_recovered"] += mask_iou(recovered, fixed)
        agg["iou_moving"] += mask_iou(moving, fixed)

    for k in agg:
        agg[k] /= n_pairs
    return agg


def print_eval(tag: str, m: dict) -> None:
    print(f"\n  === Held-out evaluation ({tag}) ===")
    print(f"    corner error     : {m['corner_px']:6.2f} px   "
          f"(identity baseline {m['corner_px_baseline']:6.2f} px)")
    print(f"    translation error: {m['trans_px']:6.2f} px")
    print(f"    rotation error   : {m['rot_deg']:6.2f} deg")
    print(f"    IoU(recovered,fixed): {m['iou_recovered']:.3f}   "
          f"(moving vs fixed {m['iou_moving']:.3f})")
    frac = 1.0 - m['corner_px'] / max(m['corner_px_baseline'], 1e-9)
    print(f"    -> closes {frac*100:4.1f}% of the gap vs leaving the polygon in place")


# ===============================================================================
# G  VISUALISATION
# ===============================================================================

def save_loss_curve(history: dict, out_path: str = "sim_loss_curve.png") -> None:
    steps = list(range(1, len(history["loss"]) + 1))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(steps, history["loss"], color="#1f77b4")
    axes[0].set_title("Training loss")
    axes[0].set_xlabel("step"); axes[0].set_ylabel("loss"); axes[0].grid(alpha=0.3)

    axes[1].plot(steps, history["trans_px"], label="translation err (px)", color="#2ca02c")
    axes[1].plot(steps, history["corner_px"], label="corner err (px)", color="#1f77b4")
    ax2 = axes[1].twinx()
    ax2.plot(steps, history["rot_deg"], label="rotation err (deg)", color="#d62728", alpha=0.7)
    ax2.set_ylabel("rotation error (deg)", color="#d62728")
    axes[1].set_title("Interpretable errors\n(translation drops fast; rotation is the hard part)")
    axes[1].set_xlabel("step"); axes[1].set_ylabel("pixel error"); axes[1].grid(alpha=0.3)
    axes[1].legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  >  Loss/metric curves saved to {out_path}")


def save_recovery_results(model: AffineRegistrationNet, n_samples: int = 4,
                          out_path: str = "sim_recovery_results.png",
                          device: str = "cpu",
                          dataset_kwargs: dict | None = None) -> None:
    model.eval()
    test_ds = SimulatedPairDataset(length=n_samples, seed=777, **(dataset_kwargs or {}))
    pts = reference_grid_points(n=3, device=device)

    fig = plt.figure(figsize=(11, 3.2 * n_samples))
    gs = gridspec.GridSpec(n_samples, 3, figure=fig, hspace=0.35, wspace=0.08)
    titles = ["Fixed (ground truth)", "Moving (perturbed)", "Recovered (network)"]
    cmaps = ["Greens", "Reds", "Blues"]

    with torch.no_grad():
        for row in range(n_samples):
            stacked, gt_flat = test_ds[row]
            sb = stacked.unsqueeze(0).to(device)
            pred = model(sb).squeeze(0)

            fixed, moving = stacked[0:1], stacked[1:2]
            recovered = recover_mask(moving.to(device), pred).cpu()

            iou_before = mask_iou(moving, fixed)
            iou_after = mask_iou(recovered, fixed)
            ce = corner_error_px(pred.unsqueeze(0), gt_flat.unsqueeze(0).to(device), pts)

            masks = [fixed.squeeze().numpy(), moving.squeeze().numpy(),
                     recovered.squeeze().numpy()]
            for col, (mask, title, cmap) in enumerate(zip(masks, titles, cmaps)):
                ax = fig.add_subplot(gs[row, col])
                ax.imshow(mask, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
                if col > 0:
                    ax.contour(masks[0], levels=[0.5], colors=["lime"], linewidths=0.8)
                if row == 0:
                    ax.set_title(title, fontsize=10, fontweight="bold")
                # Annotate the IoU change on the moving/recovered panels
                if col == 1:
                    ax.text(4, 16, f"IoU {iou_before:.2f}", color="white", fontsize=9,
                            bbox=dict(facecolor="black", alpha=0.6, pad=1))
                if col == 2:
                    ax.text(4, 16, f"IoU {iou_after:.2f}", color="white", fontsize=9,
                            bbox=dict(facecolor="black", alpha=0.6, pad=1))
                ax.axis("off")

            print(f"  sample {row}: IoU {iou_before:.2f} -> {iou_after:.2f}   "
                  f"corner err {ce:.2f}px")

    fig.suptitle("Simulated ADS Polygon Recovery   "
                 "(green contour = ground-truth boundary)", fontsize=11, y=1.01)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  >  Recovery figure saved to {out_path}")


# ===============================================================================
# MAIN
# ===============================================================================

def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*64}")
    print(f"  Simulated Dead-Tree Annotation Recovery   |   Steps 2 & 3")
    print(f"  device={device}")
    print(f"{'='*64}")

    print("\n[A] Synthetic mask generator ...")
    rng = np.random.default_rng(0)
    m = generate_random_blob_mask(H, W, rng=rng)
    assert m.shape == (H, W) and m.dtype == np.float32
    print(f"    mask shape={m.shape}  coverage={m.mean()*100:.1f}%  OK")

    print("\n[B] Affine-metric self-checks (catch convention bugs) ...")
    pts = reference_grid_points(n=3)
    ident = torch.tensor([[1., 0., 0., 0., 1., 0.]])
    assert corner_error_px(ident, ident, pts) == 0.0
    # A known +30px x-translation in normalised units (30 / (W/2)):
    shift = torch.tensor([[1., 0., 30. / (W / 2), 0., 1., 0.]])
    assert abs(translation_error_px(ident, shift) - 15.0) < 1e-3, "trans metric off"
    # A known +5 deg rotation must read back as ~5 deg:
    th = math.radians(5.0)
    rot = torch.tensor([[math.cos(th), -math.sin(th), 0.,
                         math.sin(th),  math.cos(th), 0.]])
    assert abs(rotation_error_deg(ident, rot) - 5.0) < 1e-2, "rotation metric off"
    print("    corner/translation/rotation metrics verified  OK")

    print("\n[C] SimulatedPairDataset (reproducibility) ...")
    a = SimulatedPairDataset(length=4, seed=1)[0][1]
    b = SimulatedPairDataset(length=4, seed=1)[0][1]
    c = SimulatedPairDataset(length=4, seed=2)[0][1]
    assert torch.allclose(a, b) and not torch.allclose(a, c), "seed not controlling perturbations"
    print("    same seed -> identical perturbations, different seed -> different  OK")

    print("\n[D] AffineRegistrationNet ...")
    out = AffineRegistrationNet()(torch.zeros(2, 2, H, W))
    assert out.shape == (2, 6)
    print(f"    output shape={tuple(out.shape)}  identity-init  OK")

    print("\n[E] Training (reprojection loss) ...")
    model, history = train(n_steps=400, batch_size=32, lr=1e-3,
                           loss_kind="reproj", device=device)
    save_loss_curve(history, "sim_loss_curve.png")

    print("\n[F] Held-out evaluation ...")
    metrics = evaluate(model, n_pairs=200, device=device)
    print_eval("translation<=30px, rotation<=5deg", metrics)

    print("\n[G] Recovery visualisation ...")
    save_recovery_results(model, n_samples=4, device=device)

    print(f"\n{'='*64}")
    print("  Done. Read the numbers in [F], not just the loss curve:")
    print("    - translation error should be small (a few px)")
    print("    - rotation error tells you how recoverable 5 deg rotation is")
    print("    - IoU(recovered,fixed) is the headline 'did we realign it' score")
    print(f"{'='*64}\n")


if __name__ == "__main__":
    main()
