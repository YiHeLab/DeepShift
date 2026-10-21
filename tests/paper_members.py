"""A run root carrying the five training records a paper table's arm is measured from.

``si_common.final_members`` reads ``final/fold<N>.json``: the checkpoint each member
selected and the manifest it was trained under. A table that
declares the final model is only believed when those five records certify it, so a test
that emits one builds them here rather than stubbing the check out.
"""
import json
from pathlib import Path

from deepshift import training

MEMBER_FOLDS = (1, 2, 3, 4, 5)


def final_recipe_manifest() -> dict:
    """The protocol, model and loss blocks a paper member's training record carries."""
    recipe = training.FINAL_TRAINING_RECIPE
    return {
        "protocol": {"epochs": recipe.epochs, "batch_size": recipe.batch_size,
                     "learning_rate": recipe.learning_rate,
                     "eta_min": recipe.learning_rate * recipe.eta_min_fraction,
                     "weight_decay": recipe.weight_decay, "dropout": recipe.dropout,
                     "heads": recipe.heads, "gradient_clip": recipe.gradient_clip,
                     "seed": recipe.seed,
                     "ckpt_every": recipe.checkpoint_every, "data_frac": 0.0,
                     "checkpoint_selection": recipe.checkpoint_selection,
                     "determinism": {"deterministic_algorithms": recipe.deterministic}},
        "model": {"tag": training.MODEL_TAG,
                  "dim": recipe.dimension, "layers": recipe.layers},
        "loss": {"huber_delta": recipe.huber_delta},
        "inputs": {"train_materializations": None},
    }


def build(root: Path) -> Path:
    """A build carrying both the unfiltered calibration members and the final members."""
    (root / "final").mkdir(parents=True)
    for fold in MEMBER_FOLDS:
        calibration = root / f"refold_fold{fold}.final.pt"
        calibration.write_text(f"calibration member {fold}\n")
        checkpoint = root / "final" / f"fold{fold}.final.pt"
        checkpoint.write_text(f"final member {fold}\n")
        (root / "final" / f"fold{fold}.json").write_text(json.dumps({
            "ckpt": str(checkpoint),
            "manifest": final_recipe_manifest(),
        }))
    return root
