import json
import pathlib
import unittest

try:
    import numpy as np

    from qualiaform_part2.portfolio import bootstrap, choose_threshold, input_text, score

    HAVE_DEPS = True
except ImportError:
    HAVE_DEPS = False
ROOT = pathlib.Path(__file__).resolve().parents[1]


@unittest.skipUnless(HAVE_DEPS, "portfolio requires optional numpy")
class PortfolioTests(unittest.TestCase):
    def test_forbidden_fields_do_not_enter_features(self):
        r = {
            "noun_lemma": "书",
            "verb_lemma": "买",
            "relation": "obj",
            "masked_context": "他买了[TARGET_NP]。",
        }
        before = input_text(r)
        r.update(
            observed_form="YI_CL_N",
            classifier="本",
            np_text="一本书",
            sentence="UNSAFE_RAW",
            flags=["marker"],
            candidate_id="secret",
        )
        self.assertEqual(before, input_text(r))
        self.assertNotIn("UNSAFE_RAW", before)

    def test_dev_threshold_finds_useful_boundary(self):
        y = np.array([0, 0, 1, 1])
        p = np.array([0.1, 0.2, 0.3, 0.4])
        t, f = choose_threshold(y, p)
        self.assertTrue(0.2 < t <= 0.3)
        self.assertEqual(f, 1.0)

    def test_score_known_predictions(self):
        m = score(np.array([0, 1, 1]), np.array([0.1, 0.8, 0.1]), 0.5)
        self.assertAlmostEqual(m["accuracy"], 2 / 3)
        self.assertEqual(m["confusion_matrix"]["YI_CL_N"]["BN"], 1)

    def test_sentence_bootstrap_is_deterministic(self):
        rows = [{"sentence_hash": str(i), "observed_form": "BN" if i < 3 else "YI_CL_N"} for i in range(6)]
        p = np.array([0.1, 0.1, 0.8, 0.8, 0.8, 0.1])
        self.assertEqual(bootstrap(rows, p, 0.5, 30), bootstrap(rows, p, 0.5, 30))

    def test_protocol_excludes_test_selection(self):
        p = json.loads((ROOT / "configs/portfolio_protocol_v1.json").read_text())
        self.assertIn("dev", p["serving_selection"])
        self.assertTrue(p["encoder_frozen"])
