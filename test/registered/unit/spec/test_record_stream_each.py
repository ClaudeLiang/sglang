"""record_stream_each keeps spec-v2 verify tensors alive for the forward stream."""

import unittest

import torch

from sglang.srt.speculative.spec_utils import record_stream_each
from sglang.test.ci.ci_register import register_cpu_ci, register_xpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
register_xpu_ci(est_time=10, suite="stage-a-test-1-gpu-xpu")


class TestRecordStreamEachHostEntries(CustomTestCase):
    def test_host_tensors_and_non_tensors_are_skipped(self):
        """Host tensors have no record_stream kernel; they must be skipped, not raise."""
        record_stream_each([torch.empty(2), None, 3], torch.Stream("cpu"))


@unittest.skipUnless(torch.xpu.is_available(), "XPU is required for this test.")
class TestRecordStreamEachXPU(CustomTestCase):
    def test_block_is_not_reused_while_recorded_stream_is_busy(self):
        """A recorded device tensor's block must not be handed out again on its
        allocating stream while the recorded stream still has queued work."""
        alloc_stream, read_stream = torch.xpu.Stream(), torch.xpu.Stream()
        # A stream sharing the default queue could find another free block there.
        self.assertNotEqual(alloc_stream, torch.xpu.current_stream())
        # Sizes from torch-xpu-ops' record_stream test (about 2 GiB); read_stream
        # must still be busy at the reallocation, or the reuse below is legitimate.
        x = torch.randn(256, 1024, 1024, device="xpu")
        y = torch.randn_like(x)
        torch.xpu.synchronize()

        with torch.xpu.stream(alloc_stream):
            tensor = torch.empty(4, device="xpu")
        alloc_stream.synchronize()
        with torch.xpu.stream(read_stream):
            for _ in range(30):
                x.add_(y)
        record_stream_each([tensor], read_stream)
        freed_ptr = tensor.data_ptr()
        del tensor

        with torch.xpu.stream(alloc_stream):
            realloc = torch.empty(4, device="xpu")
        self.assertNotEqual(realloc.data_ptr(), freed_ptr)
        torch.xpu.synchronize()


if __name__ == "__main__":
    unittest.main()
