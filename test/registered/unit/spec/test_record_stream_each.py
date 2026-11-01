"""Unit tests for record_stream_each (spec-V2 verify tensor lifetimes)."""

import unittest
from unittest.mock import patch

import torch

from sglang.srt.speculative import spec_utils
from sglang.srt.speculative.spec_utils import record_stream_each
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestRecordStreamEach(CustomTestCase):
    def setUp(self):
        spec_utils._RECORD_STREAM_UNSUPPORTED.clear()

    def tearDown(self):
        spec_utils._RECORD_STREAM_UNSUPPORTED.clear()

    def test_records_device_tensors_and_skips_host_and_non_tensors(self):
        device_tensor = torch.empty(2, device="meta")  # stands in for a device tensor
        host_tensor = torch.empty(2)
        stream = torch.Stream("cpu")
        calls = []

        def fake_record_stream(tensor, s):
            calls.append((tensor, s))

        with patch.object(torch.Tensor, "record_stream", fake_record_stream):
            record_stream_each([device_tensor, host_tensor, None, 3], stream)
        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0][0], device_tensor)
        self.assertIs(calls[0][1], stream)

    def test_backend_without_record_stream_is_skipped_with_one_warning(self):
        # Meta tensors have no record_stream kernel, like an out-of-tree backend
        # that does not register one.
        tensors = [torch.empty(2, device="meta"), torch.empty(3, device="meta")]
        with self.assertLogs(spec_utils.logger, level="WARNING") as logs:
            record_stream_each(tensors, torch.Stream("cpu"))
            record_stream_each(tensors, torch.Stream("cpu"))
        self.assertEqual(len(logs.records), 1)
        self.assertIn("meta", spec_utils._RECORD_STREAM_UNSUPPORTED)


if __name__ == "__main__":
    unittest.main()
