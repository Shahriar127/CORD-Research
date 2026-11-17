"""Validated experiment configuration.

A run is fully described by one YAML file plus optional dotted overrides
(``--set attack.rate=0.01 seed=2``). The resolved config is written next to the artifacts, so
every result can be traced back to the exact settings that produced it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from cord.utils.seeding import derive_seed


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DataConfig(_Strict):
    name: str = "cifar10"
    root: Path = Path("data")
    verify_checksums: bool = True
    subset_train: int | None = Field(None, description="random sub-sample of the pool (smoke tests)")
    subset_test: int | None = None


class AttackConfig(_Strict):
    name: str = "badnets"
    rate: float = Field(0.05, ge=0.0, le=0.5)
    target: int = 0
    mapping: Literal["all_to_one", "all_to_all"] = "all_to_one"
    params: dict[str, Any] = Field(default_factory=dict)
    seed: int | None = None


class SplitConfig(_Strict):
    n_folds: int = Field(2, ge=2)
    m_tr: int = Field(2000, ge=10, description="|C_tr|, trains the PU fusion")
    m_cal: int = Field(2000, ge=10, description="|C_cal|, calibrates p-values")
    seed: int | None = None


class ModelConfig(_Strict):
    name: str = "preact_resnet18"
    kwargs: dict[str, Any] = Field(default_factory=dict)


class TrainConfig(_Strict):
    epochs: int = Field(100, ge=1)
    batch_size: int = 128
    optimizer: Literal["sgd"] = "sgd"
    lr: float = 0.01
    momentum: float = 0.9
    nesterov: bool = False
    weight_decay: float = 5e-4
    schedule: Literal["cosine", "constant"] = "cosine"
    warmup_epochs: int = 0
    crop_padding: int = 4
    hflip: bool = True
    amp: bool = True
    channels_last: bool = True
    compile: bool = False
    deterministic: bool = False
    trajectory_every: int = Field(5, ge=1, description="record held-out confidence every k epochs")
    checkpoint_every: int = Field(5, ge=1, description="resumable state every k epochs")
    eval_batch_size: int = 1024
    seed: int | None = None


class ExtractConfig(_Strict):
    batch_size: int = 1024
    layers: tuple[str, ...] = ("layer3", "penultimate")
    scaleup_factors: tuple[float, ...] = (3.0, 5.0, 7.0, 9.0, 11.0)
    strip_blends: int = 8
    strip_weight: float = 0.5


class SignalsConfig(_Strict):
    enabled: tuple[str, ...] = ("latent", "confidence", "perturbation", "activation", "label_content")
    latent_pca_dim: int = 64
    latent_trim: float = 0.1
    latent_iters: int = 3
    knn_k: int = 10
    ssl_encoder: str = "dinov2_vits14"
    ssl_batch_size: int = 256
    activation_topk: float = 0.02


class FusionConfig(_Strict):
    method: Literal["hgb", "logistic"] = "hgb"
    hgb_max_iter: int = 200
    hgb_learning_rate: float = 0.05
    hgb_max_leaf_nodes: int = 15
    hgb_min_samples_leaf: int = 50
    hgb_l2: float = 1.0
    baselines: tuple[str, ...] = ("oriented_mean_rank", "mean_rank", "max_rank", "single")
    exchangeability_permutations: int = 100
    reference: Literal["audited", "random", "oracle_clean"] = Field(
        "audited",
        description=(
            "C_tr used as the PU negative class. 'audited' (default): per fold, the top "
            "audit.budget_tr C_tr points under audit.tr_ranking are audited and confirmed poison is "
            "dropped from the fit. 'random': the raw random hold-out (mode T1 as in proposal v3.1; it "
            "contains poison at the pool rate, which removes most of the PU signal). 'oracle_clean': "
            "all poisoned C_tr points are dropped, an upper bound (full audit or mode T2). Validity "
            "holds in all three, since C_tr is independent of the scored points."
        ),
    )


class AuditConfig(_Strict):
    budget: int = Field(100, ge=0, description="top-b calibration points inspected per fold; 0 = no audit")
    miss_rate: float = Field(0.0, ge=0.0, le=1.0, description="share of poison shown to the auditor it misses")
    wrong_removals: int = Field(0, ge=0, description="clean points the auditor removes by mistake")
    wrong_mode: Literal["top", "random"] = "random"
    budget_tr: int = Field(100, ge=0, description="top-b C_tr points audited per fold (fusion.reference=audited)")
    tr_ranking: str = Field(
        "mean_rank",
        description=(
            "fusion-free score that ranks C_tr for its audit: mean_rank (designed orientation; best "
            "when the backdoor is learned), oriented_mean_rank (signals oriented by tail skewness; "
            "helped only when it was not), max_rank or single:<signal>"
        ),
    )


class CalibrationConfig(_Strict):
    alpha: float = Field(0.05, gt=0, lt=1)
    alpha_bh: float = Field(0.045, gt=0, lt=1, description="BH level that builds the e-values")
    removal_level: float = Field(0.10, gt=0, lt=1, description="t of the removal set R = {p <= t}")
    presence_k: int = Field(1, ge=1)
    presence_baselines: bool = True
    presence_permutations: int = 200
    n_resamples: int = Field(20, ge=1, description="B redraws of the C_tr / C_cal partition")
    storey_lambda: float = 0.5


class ExperimentConfig(_Strict):
    name: str
    seed: int = 0
    output_dir: Path = Path("runs")
    device: str = "auto"
    results_tag: str = Field(
        "", description="score into results-<tag>/ so alternative scorings reuse the same trained models"
    )
    data: DataConfig = DataConfig()
    attack: AttackConfig = AttackConfig()
    split: SplitConfig = SplitConfig()
    model: ModelConfig = ModelConfig()
    train: TrainConfig = TrainConfig()
    extract: ExtractConfig = ExtractConfig()
    signals: SignalsConfig = SignalsConfig()
    fusion: FusionConfig = FusionConfig()
    audit: AuditConfig = AuditConfig()
    calibration: CalibrationConfig = CalibrationConfig()

    @model_validator(mode="after")
    def _check(self) -> ExperimentConfig:
        if self.calibration.presence_k > self.split.m_cal:
            raise ValueError("presence_k cannot exceed m_cal")
        if self.attack.name == "none" and self.attack.rate != 0:
            raise ValueError("attack 'none' requires rate 0")
        return self

    # Seeds: each component is independent of the others and of evaluation order.
    def component_seed(self, component: Literal["attack", "split", "train", "score"]) -> int:
        explicit = {"attack": self.attack.seed, "split": self.split.seed, "train": self.train.seed}.get(component)
        return explicit if explicit is not None else derive_seed(self.seed, component)

    @property
    def run_dir(self) -> Path:
        return self.output_dir / self.name


# ---------------------------------------------------------------------------------------------


def _set_dotted(tree: dict[str, Any], dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    node = tree
    for k in keys[:-1]:
        node = node.setdefault(k, {})
        if not isinstance(node, dict):
            raise ValueError(f"cannot set {dotted}: {k} is not a mapping")
    node[keys[-1]] = value


def parse_overrides(pairs: list[str]) -> dict[str, Any]:
    """``["attack.rate=0.01", "train.amp=false"]`` -> nested dict, values parsed as YAML."""
    tree: dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"override must look like key=value, got {pair!r}")
        key, raw = pair.split("=", 1)
        _set_dotted(tree, key.strip(), yaml.safe_load(raw))
    return tree


def deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in update.items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path: Path | None, overrides: list[str] | None = None) -> ExperimentConfig:
    """Load YAML (supports ``defaults: [relative/base.yaml]`` inheritance) and apply overrides."""
    raw: dict[str, Any] = {}
    if path is not None:
        raw = _load_with_defaults(Path(path))
    raw = deep_merge(raw, parse_overrides(overrides or []))
    return ExperimentConfig.model_validate(raw)


def _load_with_defaults(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    merged: dict[str, Any] = {}
    for parent in raw.pop("defaults", []) or []:
        merged = deep_merge(merged, _load_with_defaults((path.parent / parent).resolve()))
    return deep_merge(merged, raw)


def dump_config(cfg: ExperimentConfig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg.model_dump(mode="json"), f, sort_keys=False)
