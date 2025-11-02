"""TEA transpiration partitioning batch utilities."""

from .batch import main as run_batch
from .TEA.TEA import simplePartition, partition
from .TEA.PreProc import build_dataset, preprocess

__all__ = ["run_batch", "simplePartition", "partition", "build_dataset", "preprocess"]
