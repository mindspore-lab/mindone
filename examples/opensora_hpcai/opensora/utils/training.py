from dataclasses import dataclass


@dataclass
class TrainingOptions:
    steps: int = 100
