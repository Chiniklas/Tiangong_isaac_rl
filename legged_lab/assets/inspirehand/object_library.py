"""Utilities for loading InspireHand grasp objects from the GraspXL dataset.

This module keeps the existing project untouched while adding helpers that
scan the externally-provided ``dataset/mixed_train`` folder and expose a set
of lightweight metadata objects that downstream code can use to build grasp
scenes.

Nothing here mutates environment behaviour yet—the loader is standalone so it
can be integrated incrementally.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence


DEFAULT_DATASET_ROOT = Path("dataset") / "mixed_train"
DEFAULT_CONVERTED_ROOT = Path("dataset") / "grasp_usd"


@dataclass(slots=True)
class GraspObjectInfo:
    """Metadata bundle for a single grasp object.

    Attributes:
        object_id: Full directory name (e.g. ``Mug_123``) used as a stable id.
        category: High-level category parsed from the prefix of ``object_id``.
        root_dir: Directory containing the object's assets.
        urdf: Path to the floating-base URDF if present.
        fixed_base_urdf: Path to the fixed-base URDF if present.
        affordance_mesh: Mesh describing affordance regions (typically ``top``).
        non_affordance_mesh: Mesh describing discouraged contact regions.
        lowest_point: Optional scalar read from ``lowest_point_new.txt``.
        static_usd: Optional path to a pre-converted single-body USD (generated via
            ``convert_dataset_to_usd.py``).
        affordance_usd: Optional USD referencing the affordance mesh (if generated).
        non_affordance_usd: Optional USD referencing the non-affordance mesh.
    """

    object_id: str
    category: str
    root_dir: Path
    urdf: Optional[Path]
    fixed_base_urdf: Optional[Path]
    affordance_mesh: Optional[Path]
    non_affordance_mesh: Optional[Path]
    lowest_point: Optional[float]
    static_usd: Optional[Path]
    affordance_usd: Optional[Path] = None
    non_affordance_usd: Optional[Path] = None
    affordance_sdf: Optional[Path] = None
    non_affordance_sdf: Optional[Path] = None
    affordance_sdf_data: Optional[dict[str, Any]] = field(default=None, repr=False, compare=False)
    non_affordance_sdf_data: Optional[dict[str, Any]] = field(default=None, repr=False, compare=False)

    def to_dict(self) -> Dict[str, object]:
        """Convert to a serialisable dictionary with POSIX-style paths."""

        def _maybe_path(path: Optional[Path]) -> Optional[str]:
            return str(path.as_posix()) if path is not None else None

        data = asdict(self)
        data["root_dir"] = _maybe_path(self.root_dir)
        data["urdf"] = _maybe_path(self.urdf)
        data["fixed_base_urdf"] = _maybe_path(self.fixed_base_urdf)
        data["affordance_mesh"] = _maybe_path(self.affordance_mesh)
        data["non_affordance_mesh"] = _maybe_path(self.non_affordance_mesh)
        data["static_usd"] = _maybe_path(self.static_usd)
        data["affordance_usd"] = _maybe_path(self.affordance_usd)
        data["non_affordance_usd"] = _maybe_path(self.non_affordance_usd)
        data.pop("affordance_sdf_data", None)
        data.pop("non_affordance_sdf_data", None)
        return data


class GraspObjectLibrary:
    """Thin wrapper around the GraspXL PartNet-derived object set."""

    def __init__(
        self,
        dataset_root: Path | str = DEFAULT_DATASET_ROOT,
        converted_root: Path | str | None = DEFAULT_CONVERTED_ROOT,
    ) -> None:
        self.dataset_root = Path(dataset_root).expanduser().resolve()
        if not self.dataset_root.exists():
            raise FileNotFoundError(
                f"Grasp object dataset not found: {self.dataset_root}. "
                "Ensure `dataset/mixed_train` is populated before using the library."
            )

        if converted_root:
            candidate = Path(converted_root).expanduser().resolve()
            self.converted_root = candidate if candidate.exists() else None
        else:
            self.converted_root = None

        self._objects: Dict[str, GraspObjectInfo] = {}
        self._scan()

    # --------------------------------------------------------------------- #
    # Public API                                                            #
    # --------------------------------------------------------------------- #
    def all_objects(self) -> Sequence[GraspObjectInfo]:
        """Return all discovered objects sorted by id."""

        return tuple(self._objects[obj_id] for obj_id in sorted(self._objects))

    def by_category(self, category: str) -> Sequence[GraspObjectInfo]:
        """Return objects filtered by ``category`` (case-insensitive)."""

        cat_lower = category.lower()
        return tuple(
            info
            for info in self._objects.values()
            if info.category.lower() == cat_lower
        )

    def get(self, object_id: str) -> GraspObjectInfo:
        """Lookup a specific object by directory name."""

        try:
            return self._objects[object_id]
        except KeyError:
            raise KeyError(
                f"Unknown object id '{object_id}'. Available ids: {sorted(self._objects)}"
            ) from None

    # ------------------------------------------------------------------ #
    # Internal helpers                                                   #
    # ------------------------------------------------------------------ #
    def _scan(self) -> None:
        """Populate ``self._objects`` by walking the dataset tree."""

        for child in sorted(self.dataset_root.iterdir()):
            if not child.is_dir():
                continue
            object_id = child.name
            category = object_id.split("_", 1)[0]
            urdf = self._find_first(child, [object_id], suffixes=(".urdf",))
            fixed_urdf = self._find_first(
                child,
                [f"{object_id}_fixed_base", f"{object_id}_fixed"],
                suffixes=(".urdf",),
            )

            affordance_mesh = self._find_first(
                child,
                ["top_watertight_tiny", "top_watertight", "affordance"],
                suffixes=(".obj", ".stl"),
            )
            non_aff_mesh = self._find_first(
                child,
                ["bottom_watertight_tiny", "bottom_watertight", "non_affordance"],
                suffixes=(".obj", ".stl"),
            )

            lowest_point = self._read_lowest_point(child / "lowest_point_new.txt")
            static_usd, affordance_usd, non_affordance_usd, affordance_sdf, non_affordance_sdf = self._find_converted_usds(child, object_id)

            info = GraspObjectInfo(
                object_id=object_id,
                category=category,
                root_dir=child,
                urdf=urdf,
                fixed_base_urdf=fixed_urdf,
                affordance_mesh=affordance_mesh,
                non_affordance_mesh=non_aff_mesh,
                lowest_point=lowest_point,
                static_usd=static_usd,
                affordance_usd=affordance_usd,
                non_affordance_usd=non_affordance_usd,
                affordance_sdf=affordance_sdf,
                non_affordance_sdf=non_affordance_sdf,
            )
            self._objects[object_id] = info

    @staticmethod
    def _find_first(
        directory: Path, stems: Iterable[str], suffixes: Iterable[str]
    ) -> Optional[Path]:
        """Return the first existing path matching ``stem`` + ``suffix``."""

        for stem in stems:
            for suffix in suffixes:
                candidate = directory / f"{stem}{suffix}"
                if candidate.exists():
                    return candidate
        return None

    @staticmethod
    def _read_lowest_point(file_path: Path) -> Optional[float]:
        """Parse ``lowest_point_new.txt`` which stores a single float."""

        if not file_path.exists():
            return None
        try:
            text = file_path.read_text().strip()
            if not text:
                return None
            return float(text.split()[0])
        except (OSError, ValueError):
            return None

    def _find_converted_usds(
        self, directory: Path, object_id: str
    ) -> tuple[Optional[Path], Optional[Path], Optional[Path], Optional[Path], Optional[Path]]:
        local_root = directory / "usd"
        metadata_path = local_root / "metadata.json"
        candidates: list[Path] = []
        if metadata_path.exists():
            candidates.append(metadata_path)
        if self.converted_root is not None:
            converted_meta = self.converted_root / object_id / "metadata.json"
            if converted_meta.exists():
                candidates.append(converted_meta)

        static_usd: Optional[Path] = None
        affordance_usd: Optional[Path] = None
        non_affordance_usd: Optional[Path] = None
        affordance_sdf: Optional[Path] = None
        non_affordance_sdf: Optional[Path] = None

        for meta in candidates:
            try:
                data = json.loads(meta.read_text())
            except (OSError, json.JSONDecodeError):
                continue

            static_usd = Path(data["static_usd"]).expanduser() if data.get("static_usd") else static_usd
            affordance_usd = (
                Path(data["affordance_usd"]).expanduser() if data.get("affordance_usd") else affordance_usd
            )
            non_affordance_usd = (
                Path(data["non_affordance_usd"]).expanduser()
                if data.get("non_affordance_usd")
                else non_affordance_usd
            )
            affordance_sdf = (
                Path(data["affordance_sdf"]).expanduser() if data.get("affordance_sdf") else affordance_sdf
            )
            non_affordance_sdf = (
                Path(data["non_affordance_sdf"]).expanduser()
                if data.get("non_affordance_sdf")
                else non_affordance_sdf
            )

        if static_usd is None:
            fallback = local_root / f"{object_id}_static.usd"
            if fallback.exists():
                static_usd = fallback
        if static_usd is None and self.converted_root is not None:
            fallback = self.converted_root / object_id / f"{object_id}_static.usd"
            if fallback.exists():
                static_usd = fallback

        return static_usd, affordance_usd, non_affordance_usd, affordance_sdf, non_affordance_sdf


__all__ = ["GraspObjectInfo", "GraspObjectLibrary", "DEFAULT_DATASET_ROOT"]
