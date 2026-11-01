"""record_stream_each keeps spec-v2 verify tensors alive for the forward stream.

A block freed while another stream still reads it must not be handed out again
on the allocating stream, and a device type without a record_stream kernel must
not fail the verify step.
"""

import unittest

import torch

from sglang.srt.speculative import spec_utils
from sglang.srt.speculative.spec_utils import record_stream_each
from sglang.srt.utils import get_device
from sglang.test.ci.ci_register import (
    register_cpu_ci,
    register_cuda_ci,
    register_xpu_ci,
)
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")
register_xpu_ci(est_time=10, suite="stage-a-test-1-gpu-xpu")


# Devices this file is registered for; others (e.g. mps) have no stream API.
STREAM_DEVICES = ("cuda", "xpu")


def _stream_device():
    # get_device() raises when the host has no accelerator.
    try:
        device = get_device()
    except RuntimeError:
        return None
    return device if device in STREAM_DEVICES else None


DEVICE = _stream_device()


class TestRecordStreamEach(CustomTestCase):
    def setUp(self):
        spec_utils._RECORD_STREAM_UNSUPPORTED.clear()

    def tearDown(self):
        spec_utils._RECORD_STREAM_UNSUPPORTED.clear()

    @unittest.skipUnless(DEVICE, "needs a CUDA or XPU device")
    def test_block_is_not_reused_while_another_stream_reads_it(self):
        """Regression: non-CUDA tensors were never recorded, so XPU reused the block at once."""
        module = torch.get_device_module(DEVICE)
        alloc_stream, read_stream = module.Stream(), module.Stream()
        # Sizes from torch-xpu-ops' record_stream test (about 2 GiB); read_stream
        # must still be busy at the reallocation, or the reuse below is legitimate.
        x = torch.randn(256, 1024, 1024, device=DEVICE)
        y = torch.randn_like(x)
        module.synchronize()

        with module.stream(alloc_stream):
            tensor = torch.empty(4, device=DEVICE)
        alloc_stream.synchronize()
        with module.stream(read_stream):
            for _ in range(30):
                x.add_(y)
        record_stream_each([tensor], read_stream)
        freed_ptr = tensor.data_ptr()
        del tensor

        with module.stream(alloc_stream):
            realloc = torch.empty(4, device=DEVICE)
        self.assertNotEqual(realloc.data_ptr(), freed_ptr)
        module.synchronize()

    def test_device_type_without_record_stream_warns_once(self):
        """An out-of-tree device with no record_stream kernel must not fail the step."""
        # Meta tensors stand in for such a device: record_stream raises
        # NotImplementedError for them too, though with a different message.
        tensors = [torch.empty(2, device="meta"), torch.empty(3, device="meta")]
        with self.assertLogs(spec_utils.logger, level="WARNING") as logs:
            record_stream_each(tensors, torch.Stream("cpu"))
            record_stream_each(tensors, torch.Stream("cpu"))
        self.assertEqual(len(logs.records), 1)


if __name__ == "__main__":
    unittest.main()
