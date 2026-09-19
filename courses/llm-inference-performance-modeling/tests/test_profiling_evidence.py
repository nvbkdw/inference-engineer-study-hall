"""CPU trace fixtures check attribution; these are not GPU measurements."""

import importlib
import json
from pathlib import Path
import tempfile
import unittest

evidence = importlib.import_module("chapters.03_kernels.code.evidence")


class TraceAttribution(unittest.TestCase):
    def test_direct_launch_without_external_id_and_idle_intervals(self):
        events = [
            dict(cat="user_annotation", name="normalize", tid=1, ts=100, dur=15),
            dict(
                cat="cpu_op",
                name="aten::mul",
                tid=1,
                ts=101,
                dur=2,
                args={"External id": 7},
            ),
            dict(
                cat="kernel",
                name="kernel_cutlass_attention_kernel_tensor",
                ts=200,
                dur=20,
            ),
            dict(
                cat="kernel",
                name="torch_elementwise",
                ts=230,
                dur=30,
                args={"External id": 7},
            ),
            dict(cat="kernel", name="foreign_attention_kernel", ts=270, dur=5),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.json"
            path.write_text(json.dumps({"traceEvents": events}))
            result = evidence.attribute_trace(path)
        self.assertEqual(result["kernel_count"], 3)
        self.assertEqual(result["span_us"], 75)
        self.assertEqual(result["kernel_busy_us"], 55)
        self.assertEqual(
            result["categories_us"],
            {
                "attention": 20,
                "normalize": 30,
                "other": 5,
                "host_idle": 20,
            },
        )


if __name__ == "__main__":
    unittest.main()
