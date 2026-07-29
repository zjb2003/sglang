#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Analyze PD_REQ_TRACE logs to pinpoint PD disaggregation deadlock break points.

Reads the worker logs from the prefill and decode nodes, extracts every
``PD_REQ_TRACE`` line, and aligns each request's lifecycle across the two
sides by ``rid`` (the cross-node-consistent request uuid). Requests whose
lifecycle did not reach a terminal stage (``enter_running`` on decode /
``enter_inflight`` on prefill) are reported as stuck, with the exact stage
where each side halted, so the deadlock break point is visible at a glance.

Log line format produced by the tracing code (see prefill.py / decode.py):

    [2026-07-29 14:16:06 ATTN_CP6 TP6 EP6] PD_REQ_TRACE side=prefill \
        stage=enter_bootstrap_queue rid=abc room=12 ts=14:16:06.123 input_len=20000

Usage:
    python3 scripts/pd_trace_analyze.py \\
        --prefill prefill.log --decode decode.log
    # or pass a single combined log containing both sides:
    python3 scripts/pd_trace_analyze.py --combined both.log
    # show every request, not just the stuck ones:
    python3 scripts/pd_trace_analyze.py --prefill prefill.log --decode decode.log --all
"""

import argparse
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# Stage ordering for a human-readable lifecycle. Higher index = later stage.
PREFILL_STAGES = [
    "enter_bootstrap_queue",
    "bootstrap_done",
    "enter_wait_queue",
    "enter_inflight",
]
DECODE_STAGES = [
    "enter_prealloc_queue",
    "bootstrap_done",
    "enter_transfer_queue",
    "kv_received",
    "enter_wait_queue",
    "enter_running",
]

# Lines look like:  [2026-07-29 14:16:06 ATTN_CP6 TP6 EP6] PD_REQ_TRACE side=prefill ...
# The leading bracket group is optional for combined/mangled logs.
LINE_RE = re.compile(
    r"\[(?P<asctime>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\s*(?P<rank>[^\]]*)\]\s*"
    r"PD_REQ_TRACE\s+(?P<rest>.*)$"
)
# Fallback: no bracket prefix (e.g. when a log aggregator stripped it).
BARE_RE = re.compile(r"PD_REQ_TRACE\s+(?P<rest>.*)$")

# Capture the key=value tail. The fields we care about are explicitly named;
# everything else is kept as a raw dict for display.
KV_RE = re.compile(r"(\w+)=([^\s]+)")


@dataclass
class TraceEvent:
    side: str           # "prefill" | "decode"
    stage: str
    rid: str
    room: Optional[str]
    ts: Optional[str]   # wall-clock from the ts= field
    asctime: Optional[str]  # log-line timestamp prefix
    rank: Optional[str]  # rank prefix, e.g. "ATTN_CP6 TP6 EP6"
    extra: Dict[str, str] = field(default_factory=dict)

    @property
    def sort_key(self) -> str:
        # Prefer the log-line wall-clock (coarsest but monotonic per file);
        # fall back to the ts= field, then a stable placeholder.
        return self.asctime or self.ts or ""


def parse_line(line: str) -> Optional[TraceEvent]:
    """Parse a single log line into a TraceEvent, or None if not a trace line."""
    line = line.rstrip("\n")
    m = LINE_RE.search(line)
    asctime = None
    rank = None
    if m:
        asctime = m.group("asctime")
        rank = (m.group("rank") or "").strip() or None
        rest = m.group("rest")
    else:
        m = BARE_RE.search(line)
        if not m:
            return None
        rest = m.group("rest")

    fields: Dict[str, str] = {}
    for key, value in KV_RE.findall(rest):
        fields[key] = value

    side = fields.get("side")
    stage = fields.get("stage")
    rid = fields.get("rid")
    if not side or not stage or not rid:
        return None

    fields.pop("side", None)
    fields.pop("stage", None)
    fields.pop("rid", None)
    room = fields.pop("room", None)
    ts = fields.pop("ts", None)
    return TraceEvent(
        side=side, stage=stage, rid=rid, room=room, ts=ts,
        asctime=asctime, rank=rank, extra=fields,
    )


def load_events(path: Optional[str]) -> List[TraceEvent]:
    if not path:
        return []
    events: List[TraceEvent] = []
    try:
        with open(path, "r", errors="replace") as f:
            for line in f:
                ev = parse_line(line)
                if ev is not None:
                    events.append(ev)
    except OSError as e:
        print(f"warning: cannot read {path}: {e}", file=sys.stderr)
    return events


def group_by_rid(events: List[TraceEvent]) -> Dict[str, List[TraceEvent]]:
    """Group events by rid, preserving chronological order within each rid."""
    by_rid: Dict[str, List[TraceEvent]] = defaultdict(list)
    for ev in events:
        by_rid[ev.rid].append(ev)
    for rid in by_rid:
        by_rid[rid].sort(key=lambda e: e.sort_key)
    return by_rid


def is_stuck(events: List[TraceEvent]) -> bool:
    """A request is stuck if it never reached the terminal stage on either side.

    Prefill is terminal once it hits ``enter_inflight`` (KV is being sent);
    decode is terminal once it hits ``enter_running`` (decoding). If neither
    terminal stage appears, the request never completed its lifecycle and is
    a candidate for the deadlock.
    """
    stages = {e.stage for e in events}
    terminal = {"enter_inflight", "enter_running"}
    return not (stages & terminal)


def stage_index(side: str, stage: str) -> int:
    order = PREFILL_STAGES if side == "prefill" else DECODE_STAGES
    return order.index(stage) if stage in order else -1


def render_rid(rid: str, events: List[TraceEvent], show_all: bool) -> Optional[str]:
    """Render one rid's aligned timeline. Returns None if it should be skipped."""
    if not show_all and not is_stuck(events):
        return None

    prefill_evs = [e for e in events if e.side == "prefill"]
    decode_evs = [e for e in events if e.side == "decode"]
    lines: List[str] = []
    lines.append(f"rid={rid}  room={events[0].room}  stuck={is_stuck(events)}")

    def fmt_ev(e: TraceEvent) -> str:
        rank = f" [{e.rank}]" if e.rank else ""
        ts = e.ts or "?"
        extra = " ".join(f"{k}={v}" for k, v in e.extra.items())
        extra = f"  {extra}" if extra else ""
        return f"    {ts}{rank}  stage={e.stage}{extra}"

    if prefill_evs:
        lines.append("  PREFILL:")
        for e in prefill_evs:
            lines.append(fmt_ev(e))
    else:
        lines.append("  PREFILL: (no events — request never entered prefill bootstrap)")

    if decode_evs:
        lines.append("  DECODE:")
        for e in decode_evs:
            lines.append(fmt_ev(e))
    else:
        lines.append("  DECODE: (no events — request never reached decode)")

    # Diagnose the break point.
    lines.append("  " + diagnose(events))
    return "\n".join(lines)


def diagnose(events: List[TraceEvent]) -> str:
    """One-line interpretation of where this request is stuck."""
    p_stages = [e.stage for e in events if e.side == "prefill"]
    d_stages = [e.stage for e in events if e.side == "decode"]
    p = set(p_stages)
    d = set(d_stages)

    # No decode events at all.
    if not d_stages:
        if not p_stages:
            return "DIAG: no trace events on either side."
        if "enter_bootstrap_queue" in p and "bootstrap_done" not in p:
            return ("DIAG: prefill stuck in BOOTSTRAPPING (no bootstrap_done); "
                    "decode never saw this rid → handshake never reached decode "
                    "(NIXL bootstrap channel).")
        return "DIAG: prefill progressed but decode never received the request."

    # Decode present but stuck.
    if "kv_received" not in d and "enter_running" not in d:
        if "enter_transfer_queue" in d:
            # In transfer queue, waiting for KV.
            if "enter_inflight" in p:
                return ("DIAG: decode waiting for KV (enter_transfer_queue, no "
                        "kv_received) while prefill is in inflight (sending) → "
                        "KV transfer in progress but not completing.")
            if "bootstrap_done" in p and "enter_inflight" not in p:
                return ("DIAG: decode waiting for KV but prefill halted after "
                        "bootstrap_done (never entered inflight) → prefill not "
                        "forwarding to produce KV.")
            if "enter_bootstrap_queue" in p and "bootstrap_done" not in p:
                return ("DIAG: decode is in transfer_queue (already handshook) "
                        "but THIS rid on prefill is still bootstrapping → "
                        "different request batches; check rid alignment.")
            return ("DIAG: decode waiting for KV; prefill side has no inflight "
                    "events for this rid.")
        if "bootstrap_done" in d and "enter_transfer_queue" not in d:
            return ("DIAG: decode handshook (bootstrap_done) but never entered "
                    "transfer_queue → token pool full, request blocked in "
                    "prealloc.")
        if "enter_prealloc_queue" in d and "bootstrap_done" not in d:
            return ("DIAG: decode in prealloc queue but handshake never "
                    "completed (no bootstrap_done) → waiting on prefill side.")
        return "DIAG: decode lifecycle stalled before running."

    # Reached a terminal-ish stage on decode.
    if "enter_running" in d:
        return "DIAG: reached enter_running (not deadlocked for this rid)."
    if "kv_received" in d and "enter_running" not in d:
        return ("DIAG: KV received but not yet running — between kv_received "
                "and enter_running (transient, not a deadlock).")
    return "DIAG: see events above."


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Align PD_REQ_TRACE logs from prefill/decode nodes to find "
                    "stuck requests and the deadlock break point.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--prefill", help="prefill worker log file")
    parser.add_argument("--decode", help="decode worker log file")
    parser.add_argument("--combined",
                        help="single log containing both sides (use instead of "
                             "--prefill/--decode)")
    parser.add_argument("--all", action="store_true",
                        help="show every request, not just the stuck ones")
    parser.add_argument("--rid", help="show only this rid")
    args = parser.parse_args()

    if args.combined:
        events = load_events(args.combined)
    elif args.prefill or args.decode:
        events = load_events(args.prefill) + load_events(args.decode)
    else:
        parser.error("provide --prefill and --decode, or --combined")

    if not events:
        print("No PD_REQ_TRACE lines found. Did you start the workers with "
              "SGLANG_PD_REQ_TRACE=1?", file=sys.stderr)
        return 1

    by_rid = group_by_rid(events)
    print(f"Total PD_REQ_TRACE events: {len(events)}")
    print(f"Distinct rids: {len(by_rid)}")

    stuck = [rid for rid, evs in by_rid.items() if is_stuck(evs)]
    print(f"Stuck rids (no enter_inflight/enter_running): {len(stuck)}")
    print("=" * 78)

    rids = sorted(by_rid.keys())
    if args.rid:
        rids = [r for r in rids if r == args.rid]
        if not rids:
            print(f"rid={args.rid} not found in logs.", file=sys.stderr)
            return 1

    shown = 0
    for rid in rids:
        rendered = render_rid(rid, by_rid[rid], show_all=args.all)
        if rendered is None:
            continue
        print(rendered)
        print("-" * 78)
        shown += 1

    if shown == 0:
        print("No stuck requests found. Every request reached enter_inflight "
              "(prefill) or enter_running (decode).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
