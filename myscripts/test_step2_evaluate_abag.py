import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
MODULE_PATH = SCRIPT_DIR / "step2-evaluate_abag.py"


def load_evaluate_module():
    sys.path.insert(0, str(SCRIPT_DIR))
    try:
        spec = importlib.util.spec_from_file_location("step2_evaluate_abag", MODULE_PATH)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(SCRIPT_DIR))


evaluate_abag = load_evaluate_module()


class TestProtenixSeedPaths(unittest.TestCase):
    def test_antibody_postprocess_reads_pxmeter_numeric_seed_directory(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            prediction_dir = root / "predictions" / "10gh_assembly1" / "seed_1"
            prediction_dir.mkdir(parents=True)

            metrics_dir = root / "per_sample" / "10gh_assembly1" / "1"
            metrics_dir.mkdir(parents=True)
            metrics_path = metrics_dir / "sample_0_metrics.json"
            metrics_path.write_text(
                json.dumps(
                    {
                        "chain": {"H": {"lddt": 0.9}, "A": {"lddt": 0.8}},
                        "interface": {"A,H": {"dockq": 0.7}},
                    }
                ),
                encoding="utf-8",
            )

            target = evaluate_abag.AntibodyTarget(
                alias="10gh_assembly1",
                pdb_id="10gh_assembly1",
                pdb_code="10gh",
                reference_cif=root / "10gh_assembly1.cif",
                antibody_chains={"H": "heavy"},
                antigen_chains={
                    "A": {
                        "label_asym_id": "A",
                        "entity_type": "protein",
                        "sabdab_antigen_types": "protein",
                    }
                },
                candidate_interfaces={("A", "H")},
                ligand_label_asym_ids=set(),
                sabdab_instances=(),
                interface_metadata={},
                sabdab_metadata={},
            )
            batch_info = evaluate_abag.BatchInfo(
                seeds=(1,),
                samples=1,
                pdb_ids=("10gh_assembly1",),
                ref_assembly_id="1",
                indices_csv=root / "indices.csv",
            )

            subset, details = evaluate_abag.build_antibody_subset_and_details(
                {"10gh_assembly1": target},
                root / "per_sample",
                batch_info,
                [],
            )

            self.assertTrue(prediction_dir.is_dir())
            self.assertEqual(evaluate_abag.pxmeter_seed_dir(1), "1")
            self.assertEqual(len(subset), 2)
            self.assertEqual(details[0]["dockq"], 0.7)


if __name__ == "__main__":
    unittest.main()
