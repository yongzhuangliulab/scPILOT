import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys
import random
from collections import namedtuple
from typing import List, Tuple

import yaml
import numpy as np
import torch
from absl import app, flags

import cellot.train
from cellot.train.experiment import prepare


Pair = namedtuple("Pair", "source target")

FLAGS = flags.FLAGS

flags.DEFINE_multi_string("config", "", "Path to config")
flags.DEFINE_string("exp_group", "cellot_exps", "Name of experiment.")
flags.DEFINE_string("online", "offline", "Run experiment online or offline.")
flags.DEFINE_boolean("restart", False, "Delete cache.")
flags.DEFINE_boolean("debug", False, "Debug mode.")
flags.DEFINE_boolean("dry", False, "Dry mode.")
flags.DEFINE_boolean("verbose", False, "Run in verbose mode.")
flags.DEFINE_integer("seed", 1327, "Random seed for training stochasticity.")


def _strip_seed_arg(argv: List[str], default: int = 1327) -> Tuple[List[str], int]:
    """Extract --seed/--seed= from argv and remove it before CellOT prepare()."""
    clean_argv = []
    seed = int(default)
    skip_next = False

    for i, arg in enumerate(argv):
        if skip_next:
            skip_next = False
            continue

        if arg == "--seed":
            if i + 1 >= len(argv):
                raise ValueError("--seed requires an integer value")
            seed = int(argv[i + 1])
            skip_next = True
            continue

        if arg.startswith("--seed="):
            seed = int(arg.split("=", 1)[1])
            continue

        clean_argv.append(arg)

    return clean_argv, seed


def set_seed(seed: int, deterministic: bool = True):
    """Set RNG states used by Python, NumPy and PyTorch."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    print(f"Global seed set to {seed}", flush=True)


def _safe_record_seed(config, seed: int):
    """Record training seed in config.yaml for traceability.

    This does not change config.datasplit.random_state, which should remain fixed
    at 0 in these benchmark runs.
    """
    try:
        setattr(config, "seed", int(seed))
    except Exception:
        pass

    try:
        config.training.seed = int(seed)
    except Exception:
        try:
            config.training["seed"] = int(seed)
        except Exception:
            pass


def main(argv):
    argv_for_prepare, seed = _strip_seed_arg(list(argv), default=1327)
    set_seed(seed)

    config, outdir = prepare(argv_for_prepare)
    _safe_record_seed(config, seed)
    set_seed(seed)

    if FLAGS.dry:
        print(outdir)
        print(config)
        return

    outdir = outdir.resolve()
    outdir.mkdir(exist_ok=True, parents=True)

    yaml.dump(
        config.to_dict(), open(outdir / "config.yaml", "w"), default_flow_style=False
    )

    cachedir = outdir / "cache"
    cachedir.mkdir(exist_ok=True)

    if FLAGS.restart:
        (cachedir / "model.pt").unlink(missing_ok=True)
        (cachedir / "scalars").unlink(missing_ok=True)
        (cachedir / "last.pt").unlink(missing_ok=True)
        (cachedir / "status").unlink(missing_ok=True)

    if config.model.name == "cellot":
        train = cellot.train.train_cellot
    elif config.model.name == "scgen" or config.model.name == "cae":
        train = cellot.train.train_auto_encoder
    elif config.model.name in {"identity", "random"}:
        return
    else:
        raise ValueError(f"Unknown model name: {config.model.name}")

    status = cachedir / "status"
    status.write_text("running")

    try:
        set_seed(seed)
        train(outdir, config)
    except ValueError as error:
        status.write_text("bugged")
        print("Training bugged", flush=True)
        raise error
    else:
        status.write_text("done")
        print("Training finished", flush=True)


if __name__ == "__main__":
    if "--help" in sys.argv:
        app.run(main)
    else:
        main(sys.argv)
