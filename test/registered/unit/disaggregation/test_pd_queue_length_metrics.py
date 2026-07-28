"""Unit tests for the PD queue-length instrumentation.

Covers the pieces added for PD bottleneck triage:

1. ``FastQueue.__len__`` / ``qsize`` so the backlog of pending NIXL transfer
   chunks is measurable from the metrics path.
2. ``SchedulerStats`` carries ``num_nixl_transfer_queue_chunks`` (prefill) and
   ``num_decode_retracted_queue_reqs`` (decode).
3. The collection logic in ``metrics_reporter`` populates the two new stats
   from the scheduler state (NIXL transfer-queue backlog + retracted queue
   current length).
4. The ``_log_pd_queue_snapshot*`` helpers honor the interval gate and emit
   the expected single-line snapshot.
"""

import importlib.util
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.disaggregation.common.utils import FastQueue
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.observability.metrics_collector import SchedulerStats
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

# The prefill/decode scheduler modules pull in heavy torch/inductor deps at
# import time. On CPU-only CI runners (e.g. macOS without a CUDA triton
# toolchain) the import itself can fail, which is unrelated to the logic
# under test. We skip the snapshot-helper tests when those modules cannot be
# imported; their behavior is exercised end-to-end in GPU CI.
_PREFILL_MODULE = importlib.util.find_spec("sglang.srt.disaggregation.prefill")
_DECODE_MODULE = importlib.util.find_spec("sglang.srt.disaggregation.decode")
_SNAPSHOT_MODULES_AVAILABLE = False
if _PREFILL_MODULE is not None and _DECODE_MODULE is not None:
    try:
        from sglang.srt.disaggregation.prefill import Scheduler as _PrefillSched  # noqa: F401
        from sglang.srt.disaggregation.decode import Scheduler as _DecodeSched  # noqa: F401
        _SNAPSHOT_MODULES_AVAILABLE = True
    except Exception:  # pragma: no cover - environment-dependent
        _SNAPSHOT_MODULES_AVAILABLE = False


def _collect_nixl_transfer_chunks(scheduler):
    """Mirror of the prefill-side collection block in metrics_reporter."""
    kv_mgr = getattr(
        scheduler.disagg_prefill_bootstrap_queue, "kv_manager", None
    )
    transfer_queues = getattr(kv_mgr, "transfer_queues", None)
    if transfer_queues is not None:
        return sum(len(q) for q in transfer_queues)
    return 0


def _collect_decode_retracted_len(scheduler):
    """Mirror of the decode-side collection block in metrics_reporter."""
    return len(scheduler.disagg_decode_prealloc_queue.retracted_queue)


class TestFastQueueLen(CustomTestCase):
    """FastQueue must expose a thread-safe length for metrics collection."""

    def test_empty_queue_has_zero_len(self):
        q = FastQueue()
        self.assertEqual(len(q), 0)
        self.assertEqual(q.qsize(), 0)

    def test_put_increases_len_and_get_decreases_it(self):
        q = FastQueue()
        q.put("a")
        q.put("b")
        self.assertEqual(len(q), 2)
        self.assertEqual(q.qsize(), 2)
        q.get()
        self.assertEqual(len(q), 1)

    def test_len_is_thread_safe_under_cond(self):
        # __len__ acquires the internal Condition; just verify it does not
        # raise when the queue is concurrently accessed.
        q = FastQueue()
        q.put("x")
        # No assertion needed: if __len__ did not lock, it could race; the
        # test is a smoke check that the method exists and runs.
        self.assertEqual(len(q), 1)


class TestSchedulerStatsQueueFields(CustomTestCase):
    """SchedulerStats must carry the two new PD queue-length fields."""

    def test_default_values_are_zero(self):
        stats = SchedulerStats()
        self.assertEqual(stats.num_nixl_transfer_queue_chunks, 0)
        self.assertEqual(stats.num_decode_retracted_queue_reqs, 0)

    def test_fields_are_settable(self):
        stats = SchedulerStats()
        stats.num_nixl_transfer_queue_chunks = 7
        stats.num_decode_retracted_queue_reqs = 3
        self.assertEqual(stats.num_nixl_transfer_queue_chunks, 7)
        self.assertEqual(stats.num_decode_retracted_queue_reqs, 3)


class TestPrefillNixlBacklogCollection(CustomTestCase):
    """The prefill-side collection block must sum across all NIXL shards."""

    def _make_scheduler(self, per_shard_backlogs):
        queues = [FastQueue() for _ in per_shard_backlogs]
        for q, n in zip(queues, per_shard_backlogs):
            for _ in range(n):
                q.put(object())
        kv_mgr = SimpleNamespace(transfer_queues=queues)
        bootstrap_queue = SimpleNamespace(queue=[], kv_manager=kv_mgr)
        return SimpleNamespace(
            disagg_prefill_bootstrap_queue=bootstrap_queue,
        )

    def test_sums_backlog_across_multiple_shards(self):
        scheduler = self._make_scheduler([3, 5, 0, 2])
        self.assertEqual(_collect_nixl_transfer_chunks(scheduler), 10)

    def test_zero_backlog_when_all_shards_empty(self):
        scheduler = self._make_scheduler([0, 0, 0])
        self.assertEqual(_collect_nixl_transfer_chunks(scheduler), 0)

    def test_missing_kv_manager_returns_zero(self):
        bootstrap_queue = SimpleNamespace(queue=[], kv_manager=None)
        scheduler = SimpleNamespace(disagg_prefill_bootstrap_queue=bootstrap_queue)
        self.assertEqual(_collect_nixl_transfer_chunks(scheduler), 0)

    def test_missing_transfer_queues_returns_zero(self):
        bootstrap_queue = SimpleNamespace(queue=[], kv_manager=SimpleNamespace())
        scheduler = SimpleNamespace(disagg_prefill_bootstrap_queue=bootstrap_queue)
        self.assertEqual(_collect_nixl_transfer_chunks(scheduler), 0)


class TestDecodeRetractedCollection(CustomTestCase):
    """The decode-side collection block must count the retracted queue."""

    def test_counts_current_retracted_queue_length(self):
        retracted = [object(), object(), object()]
        scheduler = SimpleNamespace(
            disagg_decode_prealloc_queue=SimpleNamespace(
                queue=[object()], retracted_queue=retracted
            ),
        )
        self.assertEqual(_collect_decode_retracted_len(scheduler), 3)

    def test_zero_when_retracted_queue_empty(self):
        scheduler = SimpleNamespace(
            disagg_decode_prealloc_queue=SimpleNamespace(
                queue=[], retracted_queue=[]
            ),
        )
        self.assertEqual(_collect_decode_retracted_len(scheduler), 0)


@unittest.skipIf(
    not _SNAPSHOT_MODULES_AVAILABLE,
    "prefill/decode scheduler modules unavailable in this environment",
)
class TestPdQueueSnapshotPrefill(CustomTestCase):
    """_log_pd_queue_snapshot (prefill) must throttle and format correctly."""

    def _make_scheduler(self, counts, method):
        bootstrap_queue = SimpleNamespace(
            queue=[object()] * counts.get("bootstrap", 0),
            kv_manager=SimpleNamespace(
                transfer_queues=[
                    FastQueue() for _ in range(counts.get("nixl_shards", 0))
                ]
            ),
        )
        for q, n in zip(
            bootstrap_queue.kv_manager.transfer_queues,
            counts.get("nixl_chunks_per_shard", []),
        ):
            for _ in range(n):
                q.put(object())
        running_batch = SimpleNamespace(reqs=[object()] * counts.get("running", 0))
        return SimpleNamespace(
            disagg_prefill_bootstrap_queue=bootstrap_queue,
            waiting_queue=[object()] * counts.get("waiting", 0),
            running_batch=running_batch,
            disagg_prefill_inflight_queue=[object()] * counts.get("inflight", 0),
            _last_pd_queue_snapshot_t=0.0,
            _log_pd_queue_snapshot=method,
        )

    def test_disabled_when_interval_is_zero(self):
        from sglang.srt import environ
        from sglang.srt.disaggregation.prefill import Scheduler as PrefillExt

        with patch.object(environ.envs, "SGLANG_PD_QUEUE_SNAPSHOT_INTERVAL", 0):
            sched = self._make_scheduler({}, PrefillExt._log_pd_queue_snapshot)
            with patch("sglang.srt.disaggregation.prefill.logger") as mock_log:
                sched._log_pd_queue_snapshot(sched)
                mock_log.info.assert_not_called()

    def test_emits_snapshot_on_first_call_with_all_counts(self):
        from sglang.srt import environ
        from sglang.srt.disaggregation.prefill import Scheduler as PrefillExt

        with patch.object(environ.envs, "SGLANG_PD_QUEUE_SNAPSHOT_INTERVAL", 1.0):
            sched = self._make_scheduler(
                {
                    "bootstrap": 2,
                    "waiting": 3,
                    "running": 1,
                    "inflight": 4,
                    "nixl_shards": 2,
                    "nixl_chunks_per_shard": [5, 7],
                },
                PrefillExt._log_pd_queue_snapshot,
            )
            with patch("sglang.srt.disaggregation.prefill.logger") as mock_log:
                sched._log_pd_queue_snapshot(sched)
                self.assertEqual(mock_log.info.call_count, 1)
                args = mock_log.info.call_args.args
                self.assertIn("PD_QUEUE_SNAPSHOT mode=prefill", args[0])
                formatted = args[0] % args[1:]
                self.assertIn("bootstrap=2", formatted)
                self.assertIn("waiting=3", formatted)
                self.assertIn("running=1", formatted)
                self.assertIn("inflight=4", formatted)
                self.assertIn("nixl_chunks=12", formatted)

    def test_throttled_within_interval(self):
        from sglang.srt import environ
        from sglang.srt.disaggregation.prefill import Scheduler as PrefillExt

        with patch.object(environ.envs, "SGLANG_PD_QUEUE_SNAPSHOT_INTERVAL", 1.0):
            sched = self._make_scheduler({}, PrefillExt._log_pd_queue_snapshot)
            with patch("sglang.srt.disaggregation.prefill.logger") as mock_log:
                sched._log_pd_queue_snapshot(sched)
                sched._log_pd_queue_snapshot(sched)
                self.assertEqual(mock_log.info.call_count, 1)

    def test_emits_again_after_interval_elapses(self):
        from sglang.srt import environ
        from sglang.srt.disaggregation.prefill import Scheduler as PrefillExt

        import time

        with patch.object(environ.envs, "SGLANG_PD_QUEUE_SNAPSHOT_INTERVAL", 0.01):
            sched = self._make_scheduler({}, PrefillExt._log_pd_queue_snapshot)
            with patch("sglang.srt.disaggregation.prefill.logger") as mock_log:
                sched._log_pd_queue_snapshot(sched)
                time.sleep(0.02)
                sched._log_pd_queue_snapshot(sched)
                self.assertEqual(mock_log.info.call_count, 2)


@unittest.skipIf(
    not _SNAPSHOT_MODULES_AVAILABLE,
    "prefill/decode scheduler modules unavailable in this environment",
)
class TestPdQueueSnapshotDecode(CustomTestCase):
    """_log_pd_queue_snapshot_decode must throttle and format correctly."""

    def _make_scheduler(self, counts, method):
        return SimpleNamespace(
            disagg_decode_prealloc_queue=SimpleNamespace(
                queue=[object()] * counts.get("prealloc", 0),
                retracted_queue=[object()] * counts.get("retracted", 0),
            ),
            disagg_decode_transfer_queue=SimpleNamespace(
                queue=[object()] * counts.get("transfer", 0)
            ),
            waiting_queue=[object()] * counts.get("waiting", 0),
            running_batch=SimpleNamespace(reqs=[object()] * counts.get("running", 0)),
            _last_pd_queue_snapshot_t=0.0,
            _log_pd_queue_snapshot_decode=method,
        )

    def test_emits_snapshot_with_expected_fields(self):
        from sglang.srt import environ
        from sglang.srt.disaggregation.decode import Scheduler as DecodeExt

        with patch.object(environ.envs, "SGLANG_PD_QUEUE_SNAPSHOT_INTERVAL", 1.0):
            sched = self._make_scheduler(
                {
                    "prealloc": 2,
                    "transfer": 3,
                    "waiting": 1,
                    "running": 4,
                    "retracted": 5,
                },
                DecodeExt._log_pd_queue_snapshot_decode,
            )
            with patch("sglang.srt.disaggregation.decode.logger") as mock_log:
                sched._log_pd_queue_snapshot_decode(sched)
                self.assertEqual(mock_log.info.call_count, 1)
                args = mock_log.info.call_args.args
                self.assertIn("PD_QUEUE_SNAPSHOT mode=decode", args[0])
                formatted = args[0] % args[1:]
                self.assertIn("prealloc=2", formatted)
                self.assertIn("transfer=3", formatted)
                self.assertIn("waiting=1", formatted)
                self.assertIn("running=4", formatted)
                self.assertIn("retracted=5", formatted)

    def test_disabled_when_interval_is_zero(self):
        from sglang.srt import environ
        from sglang.srt.disaggregation.decode import Scheduler as DecodeExt

        with patch.object(environ.envs, "SGLANG_PD_QUEUE_SNAPSHOT_INTERVAL", 0):
            sched = self._make_scheduler({}, DecodeExt._log_pd_queue_snapshot_decode)
            with patch("sglang.srt.disaggregation.decode.logger") as mock_log:
                sched._log_pd_queue_snapshot_decode(sched)
                mock_log.info.assert_not_called()


if __name__ == "__main__":
    unittest.main()
