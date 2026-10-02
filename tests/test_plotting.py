import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import plotting


class PlottingTests(unittest.TestCase):
    def test_bucket_precision_aligns_labels_by_global_index(self):
        global_idx = np.array([4, 5, 6, 7, 0, 1, 2, 3])
        predicted_class = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        frame = pd.DataFrame(
            {
                "global_idx": global_idx,
                "class": predicted_class,
                "typicality": np.linspace(0.05, 0.95, 8),
                "selected_moo": True,
                "selected_naive": False,
            }
        )
        true_labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])

        with tempfile.TemporaryDirectory() as directory:
            with patch.object(plotting, "PLOTS_DIR", Path(directory)):
                path, result = plotting.plot_precision_by_typicality_bucket(
                    frame, true_labels
                )

            self.assertTrue(path.exists())

        moo_rows = result[result["method"] == "MOO"]
        np.testing.assert_allclose(moo_rows["precision"], 1.0)


if __name__ == "__main__":
    unittest.main()
