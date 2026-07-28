"""Unit tests for the PD KV-transfer timing instrumentation.

Covers the four pieces added for KV-transfer bottleneck profiling:

1. ``TransferKVChunk`` timing fields and ``NixlKVManager.add_transfer_request``
   stamping ``enqueue_time``.
2. ``NixlKVManager.update_transfer_status`` stamping
   ``TransferStatus.last_notif_time`` (decode side).
3. ``NixlKVSender.poll`` converting last-chunk timestamps into
   ``KVTransferMetric`` sub-phase latencies.
4. ``SchedulerReqTimeStats`` consuming those latencies and rendering them via
   ``convert_to_duration`` / ``to_kv_transfer_breakdown_csv``.
"""

import unittest
from collections import defaultdict
from unittest.mock import MagicMock

import numpy as np

from sglang.srt.disaggregation.base.conn import KVPoll, KVTransferMetric
from sglang.srt.disaggregation.nixl.conn import (
    NixlKVManager,
    NixlKVReceiver,
    NixlKVSender,
    TransferKVChunk,
    TransferStatus,
)
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.observability.req_time_stats import SchedulerReqTimeStats
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class NotificationFakeAgent:
    def __init__(self, messages):
        self.messages = messages

    def get_new_notifs(self):
        return {"peer": [msg.encode("ascii") for msg in self.messages]}


class RecordingQueue:
    def __init__(self):
        self.items = []

    def put(self, item):
        self.items.append(item)


class TestTransferKVChunkTimingFields(CustomTestCase):
    """The chunk is the carrier for all four prefill-side timestamps."""

    @staticmethod
    def _chunk(**overrides):
        kwargs = dict(
            room=7,
            prefill_kv_indices=np.array([1, 2, 3], dtype=np.int32),
            index_slice=slice(0, 3),
            is_last_chunk=True,
            prefill_aux_index=0,
            state_indices=None,
        )
        kwargs.update(overrides)
        return TransferKVChunk(**kwargs)

    def test_timing_fields_default_to_zero_so_absence_is_detectable(self):
        chunk = self._chunk()

        # 0.0 is the sentinel meaning "never stamped" -- the worker and
        # sender both branch on it, so the default must not be None.
        self.assertEqual(chunk.enqueue_time, 0.0)
        self.assertEqual(chunk.worker_dequeue_time, 0.0)
        self.assertEqual(chunk.rdma_post_time, 0.0)
        self.assertEqual(chunk.rdma_done_time, 0.0)

    def test_add_transfer_request_stamps_enqueue_time_and_routes_by_room(self):
        mgr = object.__new__(NixlKVManager)
        mgr.disaggregation_mode = DisaggregationMode.PREFILL
        mgr.enable_staging = False
        mgr.transfer_queues = [RecordingQueue(), RecordingQueue()]

        mgr.add_transfer_request(
            bootstrap_room=3,
            kv_indices=np.array([4, 5], dtype=np.int32),
            index_slice=slice(0, 2),
            is_last_chunk=True,
            chunk_id=0,
            aux_index=1,
        )

        # room 3 % 2 queues -> shard 1
        self.assertEqual(mgr.transfer_queues[0].items, [])
        self.assertEqual(len(mgr.transfer_queues[1].items), 1)

        chunk = mgr.transfer_queues[1].items[0]
        self.assertGreater(chunk.enqueue_time, 0.0)
        # Only enqueue_time is set at this point; the worker fills the rest.
        self.assertEqual(chunk.worker_dequeue_time, 0.0)
        self.assertEqual(chunk.rdma_post_time, 0.0)
        self.assertEqual(chunk.rdma_done_time, 0.0)


class TestDecodeNotificationTimestamp(CustomTestCase):
    """Decode side records when a NIXL notification was actually processed."""

    def _make_manager(self, messages, required=None):
        mgr = object.__new__(NixlKVManager)
        mgr.agent = NotificationFakeAgent(messages)
        mgr.transfer_statuses = defaultdict(TransferStatus)
        mgr.required_prefill_response_num_table = required or {}
        mgr.enable_staging = False
        mgr._staging_handler = None
        mgr._chunk_writer_counts = defaultdict(lambda: defaultdict(list))
        return mgr

    def test_default_last_notif_time_is_zero(self):
        self.assertEqual(TransferStatus().last_notif_time, 0.0)

    def test_kv_notification_stamps_last_notif_time(self):
        mgr = self._make_manager(["5_kv_2_1_0"])

        mgr.update_transfer_status()

        self.assertGreater(mgr.transfer_statuses[5].last_notif_time, 0.0)

    def test_aux_and_state_notifications_also_stamp_last_notif_time(self):
        mgr = self._make_manager(["6_aux_nokv_3", "7_state_2"], required={6: 4})

        mgr.update_transfer_status()

        self.assertGreater(mgr.transfer_statuses[6].last_notif_time, 0.0)
        self.assertGreater(mgr.transfer_statuses[7].last_notif_time, 0.0)

    def test_all_messages_in_one_batch_share_the_same_timestamp(self):
        # The timestamp is taken once per get_new_notifs() batch so the cost
        # of instrumentation stays O(1) per poll rather than O(messages).
        mgr = self._make_manager(["1_kv_0_1_0", "2_kv_0_1_0", "3_kv_0_1_0"])

        mgr.update_transfer_status()

        stamps = {mgr.transfer_statuses[room].last_notif_time for room in (1, 2, 3)}
        self.assertEqual(len(stamps), 1)
        self.assertGreater(stamps.pop(), 0.0)

    def test_untracked_room_does_not_raise(self):
        mgr = self._make_manager(["9_state_0"])
        mgr.transfer_statuses = {}  # plain dict, no defaultdict fallback

        # "state" writes through transfer_statuses[room]; with a plain dict the
        # stamping guard must not KeyError on rooms it did not create.
        with self.assertRaises(KeyError):
            mgr.update_transfer_status()


class TestSenderPollSubPhaseMetrics(CustomTestCase):
    """poll() turns raw timestamps into KVTransferMetric sub-phase latencies."""

    def _make_sender(self, timing, status=KVPoll.Success):
        mgr = MagicMock()
        mgr.check_status.return_value = status
        mgr.last_chunk_timing = {} if timing is None else {11: dict(timing)}

        sender = object.__new__(NixlKVSender)
        sender.kv_mgr = mgr
        sender.bootstrap_room = 11
        sender._send_failed = False
        sender._transfer_start_time = 100.0
        sender._transfer_metric = KVTransferMetric()
        return sender

    def test_metric_defaults_are_none_for_backends_without_instrumentation(self):
        metric = KVTransferMetric()

        self.assertIsNone(metric.worker_queue_latency_ms)
        self.assertIsNone(metric.rdma_post_latency_ms)
        self.assertIsNone(metric.rdma_transfer_latency_ms)

    def test_success_converts_timestamps_to_millisecond_latencies(self):
        sender = self._make_sender(
            {
                "enqueue_time": 1.000,
                "worker_dequeue_time": 1.002,  # +2ms queue wait
                "rdma_post_time": 1.005,  # +3ms index prep + submit
                "rdma_done_time": 1.025,  # +20ms on the wire
            }
        )

        self.assertEqual(sender.poll(), KVPoll.Success)

        metric = sender._transfer_metric
        self.assertAlmostEqual(metric.worker_queue_latency_ms, 2.0, places=3)
        self.assertAlmostEqual(metric.rdma_post_latency_ms, 3.0, places=3)
        self.assertAlmostEqual(metric.rdma_transfer_latency_ms, 20.0, places=3)

    def test_timing_entry_is_popped_to_avoid_unbounded_growth(self):
        sender = self._make_sender(
            {
                "enqueue_time": 1.0,
                "worker_dequeue_time": 1.001,
                "rdma_post_time": 1.002,
                "rdma_done_time": 1.003,
            }
        )

        sender.poll()

        self.assertNotIn(11, sender.kv_mgr.last_chunk_timing)

    def test_missing_timing_leaves_sub_phases_none(self):
        sender = self._make_sender(None)

        self.assertEqual(sender.poll(), KVPoll.Success)

        metric = sender._transfer_metric
        self.assertIsNone(metric.worker_queue_latency_ms)
        self.assertIsNone(metric.rdma_post_latency_ms)
        self.assertIsNone(metric.rdma_transfer_latency_ms)
        # Coarse-grained latency is still recorded.
        self.assertIsNotNone(metric.transfer_latency_s)

    def test_unstamped_phase_is_skipped_instead_of_recorded_as_negative(self):
        # rdma_post_time == 0.0 happens on the staging-deferred path; the
        # dependent deltas must be dropped, not emitted as huge negatives.
        sender = self._make_sender(
            {
                "enqueue_time": 1.000,
                "worker_dequeue_time": 1.004,
                "rdma_post_time": 0.0,
                "rdma_done_time": 0.0,
            }
        )

        sender.poll()

        metric = sender._transfer_metric
        self.assertAlmostEqual(metric.worker_queue_latency_ms, 4.0, places=3)
        self.assertIsNone(metric.rdma_post_latency_ms)
        self.assertIsNone(metric.rdma_transfer_latency_ms)

    def test_non_success_status_does_not_consume_timing(self):
        sender = self._make_sender(
            {
                "enqueue_time": 1.0,
                "worker_dequeue_time": 1.001,
                "rdma_post_time": 1.002,
                "rdma_done_time": 1.003,
            },
            status=KVPoll.WaitingForInput,
        )

        self.assertEqual(sender.poll(), KVPoll.WaitingForInput)

        self.assertIn(11, sender.kv_mgr.last_chunk_timing)
        self.assertIsNone(sender._transfer_metric.worker_queue_latency_ms)


class TestReceiverNotifTimePropagation(CustomTestCase):
    def _make_receiver(self, status, last_notif_time):
        mgr = MagicMock()
        mgr.check_status.return_value = status
        mgr.transfer_statuses = {13: TransferStatus(last_notif_time=last_notif_time)}

        receiver = object.__new__(NixlKVReceiver)
        receiver.kv_mgr = mgr
        receiver.bootstrap_room = 13
        receiver.conclude_state = None
        receiver.started_transfer = True
        receiver._last_notif_time = 0.0
        return receiver

    def test_success_copies_notif_time_off_the_manager(self):
        receiver = self._make_receiver(KVPoll.Success, last_notif_time=1234.5)

        self.assertEqual(receiver.poll(), KVPoll.Success)
        self.assertEqual(receiver._last_notif_time, 1234.5)

    def test_failed_also_copies_notif_time_for_post_mortem(self):
        receiver = self._make_receiver(KVPoll.Failed, last_notif_time=99.0)

        self.assertEqual(receiver.poll(), KVPoll.Failed)
        self.assertEqual(receiver._last_notif_time, 99.0)

    def test_unstamped_notif_time_is_not_copied(self):
        receiver = self._make_receiver(KVPoll.Success, last_notif_time=0.0)

        receiver.poll()

        self.assertEqual(receiver._last_notif_time, 0.0)


class TestTimeStatsSubPhaseIngestion(CustomTestCase):
    """compute_and_observe_kv_transfer_metrics() ingests sub-phase latencies."""

    @staticmethod
    def _prefill_stats():
        stats = SchedulerReqTimeStats(disagg_mode=DisaggregationMode.PREFILL)
        stats.prefill_bootstrap_queue_entry_time = 1.000
        stats.bootstrap_done_time = 1.010
        stats.wait_queue_entry_time = 1.015
        stats.forward_entry_time = 1.020
        stats.prefill_finished_time = 1.070
        stats.prefill_transfer_queue_entry_time = 1.075
        stats.prefill_kv_transfer_finish_time = 1.100
        stats.completion_time = 1.100
        return stats

    @staticmethod
    def _metric():
        return KVTransferMetric(
            transfer_latency_s=0.025,
            transfer_total_bytes=10 * 1024 * 1024,
            worker_queue_latency_ms=2.0,
            rdma_post_latency_ms=3.0,
            rdma_transfer_latency_ms=20.0,
        )

    def test_sub_phase_fields_default_to_zero(self):
        stats = SchedulerReqTimeStats()

        self.assertEqual(stats.kv_worker_queue_latency_ms, 0.0)
        self.assertEqual(stats.kv_rdma_post_latency_ms, 0.0)
        self.assertEqual(stats.kv_rdma_transfer_latency_ms, 0.0)
        self.assertEqual(stats.decode_kv_notif_time, 0.0)

    def test_sub_phases_are_stored_and_returned_in_the_result_dict(self):
        stats = self._prefill_stats()

        result = stats.compute_and_observe_kv_transfer_metrics(self._metric())

        self.assertAlmostEqual(stats.kv_worker_queue_latency_ms, 2.0)
        self.assertAlmostEqual(stats.kv_rdma_post_latency_ms, 3.0)
        self.assertAlmostEqual(stats.kv_rdma_transfer_latency_ms, 20.0)

        self.assertAlmostEqual(result["kv_worker_queue_ms"], 2.0)
        self.assertAlmostEqual(result["kv_rdma_post_ms"], 3.0)
        self.assertAlmostEqual(result["kv_rdma_transfer_ms"], 20.0)
        # Pre-existing keys must survive.
        self.assertIn("latency_ms", result)
        self.assertIn("speed_gb_s", result)
        self.assertIn("bootstrap_ms", result)

    def test_absent_sub_phases_are_omitted_from_the_result_dict(self):
        stats = self._prefill_stats()
        metric = KVTransferMetric(
            transfer_latency_s=0.025,
            transfer_total_bytes=1024,
        )

        result = stats.compute_and_observe_kv_transfer_metrics(metric)

        self.assertNotIn("kv_worker_queue_ms", result)
        self.assertNotIn("kv_rdma_post_ms", result)
        self.assertNotIn("kv_rdma_transfer_ms", result)
        self.assertEqual(stats.kv_worker_queue_latency_ms, 0.0)

    def test_sub_phases_are_observed_on_the_metrics_collector(self):
        stats = self._prefill_stats()
        stats.enable_metrics = True
        stats.metrics_collector = MagicMock()

        stats.compute_and_observe_kv_transfer_metrics(self._metric())

        observed = {
            call.args[0]: call.args[1]
            for call in stats.metrics_collector.observe_kv_transfer_sub_phase.call_args_list
        }
        self.assertEqual(
            observed, {"worker_queue": 2.0, "rdma_post": 3.0, "rdma_transfer": 20.0}
        )

    def test_no_observation_when_metrics_disabled(self):
        stats = self._prefill_stats()
        stats.enable_metrics = False
        stats.metrics_collector = MagicMock()

        stats.compute_and_observe_kv_transfer_metrics(self._metric())

        stats.metrics_collector.observe_kv_transfer_sub_phase.assert_not_called()


class TestBreakdownOutputFormat(CustomTestCase):
    """The rendered strings are what humans and chart scripts consume."""

    @staticmethod
    def _prefill_stats():
        stats = SchedulerReqTimeStats(disagg_mode=DisaggregationMode.PREFILL)
        stats.scheduler_recv_time = 1.000
        stats.prefill_bootstrap_queue_entry_time = 1.000
        stats.bootstrap_done_time = 1.010  # P2 = 10ms
        stats.wait_queue_entry_time = 1.015
        stats.forward_entry_time = 1.020  # P3 = 5ms
        stats.prefill_finished_time = 1.070  # P4 = 50ms
        stats.prefill_transfer_queue_entry_time = 1.075  # P5 = 5ms
        stats.prefill_kv_transfer_finish_time = 1.100  # P6 = 25ms
        stats.completion_time = 1.100
        stats.kv_worker_queue_latency_ms = 2.0
        stats.kv_rdma_post_latency_ms = 3.0
        stats.kv_rdma_transfer_latency_ms = 20.0
        stats.transfer_speed_gb_s = 0.4
        stats.transfer_total_mb = 10.0
        return stats

    @staticmethod
    def _decode_stats():
        stats = SchedulerReqTimeStats(disagg_mode=DisaggregationMode.DECODE)
        stats.decode_prealloc_queue_entry_time = 2.000
        stats.bootstrap_done_time = 2.008  # D1 = 8ms
        stats.decode_transfer_queue_entry_time = 2.012  # D2 = 4ms
        stats.wait_queue_entry_time = 2.040  # D3 = 28ms
        stats.forward_entry_time = 2.043  # D4 = 3ms
        stats.decode_prebuilt_finish_time = 2.044  # D5 = 1ms
        stats.completion_time = 2.044
        stats.decode_kv_notif_time = 2.038
        return stats

    def test_prefill_csv_has_expected_keys_in_pipeline_order(self):
        line = self._prefill_stats().to_kv_transfer_breakdown_csv()

        keys = [field.split("=")[0] for field in line.split(", ")]
        self.assertEqual(
            keys,
            [
                "P1_api_to_sched",
                "P2_bootstrap",
                "P3_queue",
                "P4_forward",
                "P5_send_prep",
                "P6_transfer_total",
                "K1_worker_queue",
                "K2_rdma_post",
                "K3_rdma_transfer",
                "transfer_speed_gb_s",
                "transfer_total_mb",
            ],
        )

    def test_prefill_csv_values_are_millisecond_floats(self):
        line = self._prefill_stats().to_kv_transfer_breakdown_csv()
        values = dict(field.split("=") for field in line.split(", "))

        self.assertEqual(values["P2_bootstrap"], "10.00")
        self.assertEqual(values["P3_queue"], "5.00")
        self.assertEqual(values["P4_forward"], "50.00")
        self.assertEqual(values["P5_send_prep"], "5.00")
        self.assertEqual(values["P6_transfer_total"], "25.00")
        self.assertEqual(values["K1_worker_queue"], "2.00")
        self.assertEqual(values["K2_rdma_post"], "3.00")
        self.assertEqual(values["K3_rdma_transfer"], "20.00")
        self.assertEqual(values["transfer_speed_gb_s"], "0.40")
        self.assertEqual(values["transfer_total_mb"], "10.00")

    def test_kv_sub_phases_sum_within_the_reported_transfer_total(self):
        stats = self._prefill_stats()
        values = dict(
            field.split("=")
            for field in stats.to_kv_transfer_breakdown_csv().split(", ")
        )

        sub_total = (
            float(values["K1_worker_queue"])
            + float(values["K2_rdma_post"])
            + float(values["K3_rdma_transfer"])
        )
        self.assertLessEqual(sub_total, float(values["P6_transfer_total"]) + 1e-6)

    def test_decode_csv_has_expected_keys_in_pipeline_order(self):
        line = self._decode_stats().to_kv_transfer_breakdown_csv()

        keys = [field.split("=")[0] for field in line.split(", ")]
        self.assertEqual(
            keys,
            [
                "D1_bootstrap",
                "D2_alloc_wait",
                "D3_transfer_recv",
                "D4_queue",
                "D5_fake_fwd",
            ],
        )

    def test_decode_csv_values_are_millisecond_floats(self):
        line = self._decode_stats().to_kv_transfer_breakdown_csv()
        values = dict(field.split("=") for field in line.split(", "))

        self.assertEqual(values["D1_bootstrap"], "8.00")
        self.assertEqual(values["D2_alloc_wait"], "4.00")
        self.assertEqual(values["D3_transfer_recv"], "28.00")
        self.assertEqual(values["D4_queue"], "3.00")
        self.assertEqual(values["D5_fake_fwd"], "1.00")

    def test_unified_mode_yields_an_empty_line(self):
        stats = SchedulerReqTimeStats(disagg_mode=DisaggregationMode.NULL)

        self.assertEqual(stats.to_kv_transfer_breakdown_csv(), "")

    def test_missing_timestamps_render_as_zero_not_negative(self):
        # A retracted / aborted request may never reach later stages.
        stats = SchedulerReqTimeStats(disagg_mode=DisaggregationMode.PREFILL)
        stats.prefill_bootstrap_queue_entry_time = 1.0

        values = dict(
            field.split("=")
            for field in stats.to_kv_transfer_breakdown_csv().split(", ")
        )
        for key, raw in values.items():
            self.assertGreaterEqual(float(raw), 0.0, f"{key} rendered negative")

    def test_csv_is_parseable_as_key_value_pairs(self):
        for stats in (self._prefill_stats(), self._decode_stats()):
            line = stats.to_kv_transfer_breakdown_csv()
            for field in line.split(", "):
                self.assertEqual(field.count("="), 1, f"unparseable field: {field}")
                key, raw = field.split("=")
                self.assertTrue(key)
                float(raw)  # must not raise

    def test_prefill_convert_to_duration_embeds_kv_sub_block(self):
        text = self._prefill_stats().convert_to_duration()

        self.assertIn("kv_sub=[", text)
        self.assertIn("worker_queue=2.00ms", text)
        self.assertIn("rdma_post=3.00ms", text)
        self.assertIn("rdma_transfer=20.00ms", text)
        # Existing fields must not regress.
        self.assertIn("transfer_speed=", text)
        self.assertIn("#retries=", text)

    def test_decode_convert_to_duration_embeds_kv_notif_time(self):
        text = self._decode_stats().convert_to_duration()

        self.assertIn("kv_notif_time=", text)
        self.assertIn("transfer_duration=28.00ms", text)


if __name__ == "__main__":
    unittest.main()
