from __future__ import annotations

import gzip
import importlib.util
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest import mock

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "step3-simplify_metrics.py"
SPEC = importlib.util.spec_from_file_location("step3_simplify_metrics", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class PdbStructureExportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.pred_dir = self.root / "predictions"
        self.ref_dir = self.root / "references"
        self.output_dir = self.root / "output"

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def write_prediction(
        self, entry_id: str, seed: int, sample: int, content: str
    ) -> Path:
        path = (
            self.pred_dir
            / entry_id
            / f"seed_{seed}"
            / "predictions"
            / f"{entry_id}_sample_{sample}.cif"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def write_reference(
        self, entry_id: str, content: str, *, compressed: bool = False
    ) -> Path:
        suffix = ".cif.gz" if compressed else ".cif"
        path = self.ref_dir / f"{entry_id}{suffix}"
        path.parent.mkdir(parents=True, exist_ok=True)
        if compressed:
            with gzip.open(path, "wt", encoding="utf-8") as handle:
                handle.write(content)
        else:
            path.write_text(content, encoding="utf-8")
        return path

    @staticmethod
    def pdb_row(
        entry_id: str, method: str, seed: int, sample: int
    ) -> dict[str, object]:
        return {
            "PDB编号": entry_id,
            "选择方式": method,
            "seed": seed,
            "sample": sample,
        }

    def test_builds_oracle_and_predicted_exports(self) -> None:
        oracle = self.write_prediction("1abc", 2, 3, "oracle")
        predicted = self.write_prediction("1abc", 4, 1, "predicted")
        reference = self.write_reference("1abc", "ground truth")
        rows = [
            self.pdb_row("1abc", MODULE.ORACLE, 2, 3),
            self.pdb_row("1abc", MODULE.PREDICTED_BEST, 4, 1),
        ]

        exports = MODULE.build_pdb_structure_exports(self.pred_dir, self.ref_dir, rows)

        self.assertEqual(
            [(item.source, item.filename) for item in exports],
            [
                (oracle.resolve(), "1abc_oracle.cif"),
                (predicted.resolve(), "1abc.pbest.cif"),
                (reference.resolve(), "1abc.gt.cif"),
            ],
        )

    def test_same_source_still_creates_two_named_files(self) -> None:
        source = self.write_prediction("2xyz", 1, 0, "same")
        self.write_reference("2xyz", "ground truth")
        rows = [
            self.pdb_row("2xyz", MODULE.ORACLE, 1, 0),
            self.pdb_row("2xyz", MODULE.PREDICTED_BEST, 1, 0),
        ]
        exports = MODULE.build_pdb_structure_exports(self.pred_dir, self.ref_dir, rows)

        MODULE.publish_csv_bundle(self.output_dir, {}, exports)

        self.assertTrue(source.is_file())
        self.assertEqual(
            (self.output_dir / "pdbs" / "2xyz_oracle.cif").read_text(), "same"
        )
        self.assertEqual(
            (self.output_dir / "pdbs" / "2xyz.pbest.cif").read_text(), "same"
        )
        self.assertEqual(
            (self.output_dir / "pdbs" / "2xyz.gt.cif").read_text(),
            "ground truth",
        )

    def test_compressed_reference_is_decompressed_to_gt_cif(self) -> None:
        self.write_prediction("5jkl", 1, 0, "oracle")
        self.write_prediction("5jkl", 1, 1, "predicted")
        self.write_reference("5jkl", "compressed ground truth", compressed=True)
        rows = [
            self.pdb_row("5jkl", MODULE.ORACLE, 1, 0),
            self.pdb_row("5jkl", MODULE.PREDICTED_BEST, 1, 1),
        ]
        exports = MODULE.build_pdb_structure_exports(self.pred_dir, self.ref_dir, rows)

        MODULE.publish_csv_bundle(self.output_dir, {}, exports)

        self.assertEqual(
            (self.output_dir / "pdbs" / "5jkl.gt.cif").read_text(),
            "compressed ground truth",
        )

    def test_missing_source_does_not_change_existing_outputs(self) -> None:
        old_pdb_dir = self.output_dir / "pdbs"
        old_pdb_dir.mkdir(parents=True)
        old_file = old_pdb_dir / "old.cif"
        old_file.write_text("old", encoding="utf-8")
        self.pred_dir.mkdir()
        self.ref_dir.mkdir()
        rows = [self.pdb_row("missing", MODULE.ORACLE, 1, 0)]

        with self.assertRaisesRegex(MODULE.SimplifyError, "Missing selected"):
            MODULE.build_pdb_structure_exports(self.pred_dir, self.ref_dir, rows)

        self.assertEqual(old_file.read_text(encoding="utf-8"), "old")

    def test_republish_replaces_pdb_directory_and_removes_stale_files(self) -> None:
        source = self.write_prediction("3def", 5, 2, "new")
        stale_dir = self.output_dir / "pdbs"
        stale_dir.mkdir(parents=True)
        (stale_dir / "stale.cif").write_text("stale", encoding="utf-8")
        exports = [MODULE.PdbStructureExport(source, "3def_oracle.cif")]

        MODULE.publish_csv_bundle(self.output_dir, {}, exports)

        self.assertEqual(
            sorted(path.name for path in stale_dir.iterdir()), ["3def_oracle.cif"]
        )
        self.assertEqual(
            (stale_dir / "3def_oracle.cif").read_text(encoding="utf-8"), "new"
        )

    def test_publish_failure_restores_old_csv_and_pdb_directory(self) -> None:
        source = self.write_prediction("4ghi", 1, 0, "new")
        self.output_dir.mkdir()
        csv_path = self.output_dir / "DockQ_pdb_sum.csv"
        csv_path.write_text("old csv", encoding="utf-8")
        pdb_dir = self.output_dir / "pdbs"
        pdb_dir.mkdir()
        old_pdb = pdb_dir / "old.cif"
        old_pdb.write_text("old pdb", encoding="utf-8")
        exports = [MODULE.PdbStructureExport(source, "4ghi_oracle.cif")]
        tables = {"DockQ_pdb_sum.csv": (("column",), ({"column": "new"},))}
        real_replace = MODULE.os.replace

        def fail_pdb_publish(source_path: object, destination_path: object) -> None:
            source_candidate = Path(source_path)
            destination_candidate = Path(destination_path)
            if (
                source_candidate.name == "pdbs"
                and destination_candidate == pdb_dir
                and source_candidate.parent != self.output_dir
            ):
                raise OSError("simulated publish failure")
            real_replace(source_path, destination_path)

        with (
            mock.patch.object(MODULE.os, "replace", side_effect=fail_pdb_publish),
            self.assertRaisesRegex(OSError, "simulated publish failure"),
        ):
            MODULE.publish_csv_bundle(self.output_dir, tables, exports)

        self.assertEqual(csv_path.read_text(encoding="utf-8"), "old csv")
        self.assertEqual(old_pdb.read_text(encoding="utf-8"), "old pdb")

    def test_pred_dir_is_required(self) -> None:
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            MODULE.parse_args([])

    def test_ref_dir_is_required(self) -> None:
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            MODULE.parse_args(["--pred-dir", str(self.pred_dir)])

    def test_nonexistent_pred_dir_is_rejected(self) -> None:
        rows = [self.pdb_row("1abc", MODULE.ORACLE, 1, 0)]
        with self.assertRaisesRegex(MODULE.SimplifyError, "does not exist"):
            MODULE.build_pdb_structure_exports(self.pred_dir, self.ref_dir, rows)

    def test_missing_reference_is_rejected(self) -> None:
        self.write_prediction("6mno", 1, 0, "oracle")
        self.write_prediction("6mno", 1, 1, "predicted")
        self.ref_dir.mkdir()
        rows = [
            self.pdb_row("6mno", MODULE.ORACLE, 1, 0),
            self.pdb_row("6mno", MODULE.PREDICTED_BEST, 1, 1),
        ]
        with self.assertRaisesRegex(MODULE.SimplifyError, "Missing ground-truth"):
            MODULE.build_pdb_structure_exports(self.pred_dir, self.ref_dir, rows)


if __name__ == "__main__":
    unittest.main()
