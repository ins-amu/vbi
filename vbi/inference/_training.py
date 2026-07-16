"""Structured training options for InferencePipeline / SNPE.train()."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrainingOptions:
    """
    Discoverable subset of ``SNPE.train()`` kwargs.

    ``to_kwargs()`` translates field names to the exact sbi-mirroring kwarg
    names ``SNPE.train()`` (and ``sbi.inference.SNPE.train()``) expect.  Any
    kwarg not covered here (e.g. vbi-only extensions like ``lr_schedule``,
    ``monitor_collapse``) can still be passed directly to ``train(**kwargs)``.
    """

    batch_size: int = 200
    learning_rate: float = 2e-4
    max_epochs: int = 2000
    stop_after_epochs: int = 20
    validation_fraction: float = 0.1
    clip_max_norm: float | None = 5.0
    num_atoms: int = 10
    resume_training: bool = False
    seed: int | None = None

    def to_kwargs(self) -> dict:
        kwargs = dict(
            training_batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            max_num_epochs=self.max_epochs,
            stop_after_epochs=self.stop_after_epochs,
            validation_fraction=self.validation_fraction,
            clip_max_norm=self.clip_max_norm,
            num_atoms=self.num_atoms,
            resume_training=self.resume_training,
        )
        if self.seed is not None:
            kwargs["seed"] = self.seed
        return kwargs
