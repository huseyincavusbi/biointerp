"""Core ProteinGym assays used by the project.

Assays are part of the scientific design, so they are explicit here. Models and
datasets are resolved from configuration instead of being hardcoded.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Assay:
    name: str
    shard: int
    protein: str


CORE_ASSAYS: tuple[Assay, ...] = (
    Assay("SDA_BACSU_Tsuboyama_2023_1PV0", 0, "SDA"),
    Assay("MTHR_HUMAN_Weile_2021", 1, "MTHR"),
    Assay("S22A1_HUMAN_Yee_2023_activity", 0, "S22A1"),
    Assay("MSH2_HUMAN_Jia_2020", 1, "MSH2"),
    Assay("PPARG_HUMAN_Majithia_2016", 1, "PPARG"),
    Assay("PAI1_HUMAN_Huttinger_2021", 0, "PAI1"),
)


def core_names() -> list[str]:
    return [assay.name for assay in CORE_ASSAYS]


def get_assay(name: str) -> Assay | None:
    return next((assay for assay in CORE_ASSAYS if assay.name == name), None)


def shard_for(name: str) -> int | None:
    assay = get_assay(name)
    return None if assay is None else assay.shard
