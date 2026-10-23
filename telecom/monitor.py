import json
from collections import Counter

import numpy as np

from telecom.config import ROOT


def summary():
    rows = []
    for path in sorted((ROOT / "logs").glob("app.jsonl*")):
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    requests = [row for row in rows if row["event"] == "http_request"]
    latencies = [row["latency_ms"] for row in requests]
    report = {"http_requests": len(requests), "http_successes": sum(row["status"] == "success" for row in requests),
              "http_failures": sum(row["status"] == "error" for row in requests),
              "p50_latency_ms": round(float(np.percentile(latencies, 50)), 3) if latencies else None,
              "p95_latency_ms": round(float(np.percentile(latencies, 95)), 3) if latencies else None,
              "routes": dict(Counter(row["route"] for row in rows if row["event"] == "agent_selection")),
              "tools": dict(Counter(row["tool"] for row in rows if row["event"] == "tool_execution"))}
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    summary()
