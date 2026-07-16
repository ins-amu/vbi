"""Standalone worker process wrapping the GPL-3.0-licensed JIDT toolkit.

This module is launched as an independent OS process by
:func:`vbi.feature_extraction.features_utils.call_jidt` -- never imported
into the main VBI process. It embeds its own JVM (via JPype) and loads
``infodynamics.jar`` (the GPL-3.0-licensed Java Information Dynamics
Toolkit) in that separate process. Communication with the main,
Apache-2.0-licensed VBI process happens exclusively over a line-delimited
JSON protocol on stdin/stdout: one JSON request per line in, one JSON
response per line out. This keeps VBI's own process from linking against
GPL-3.0 code at runtime -- the two programs only ever exchange plain
numeric data across that narrow, arm's-length interface. See
``docs/third_party_licenses.rst`` for details.
"""
import sys
import json

import jpype as jp

from vbi.feature_extraction.features_utils import init_jvm, nat2bit


def _calc_mi(req):
    init_jvm()
    calc_class = jp.JPackage(
        "infodynamics.measures.continuous.kraskov"
    ).MutualInfoCalculatorMultiVariateKraskov2
    calc = calc_class()
    calc.setProperty("k", str(int(req["k"])))
    calc.setProperty("NUM_THREADS", str(int(req["num_threads"])))
    calc.setProperty("TIME_DIFF", str(int(req["time_diff"])))
    calc.initialise()
    calc.startAddObservations()
    for ts_i, ts_j in req["pairs"]:
        calc.addObservations(ts_i, ts_j)
    calc.finaliseAddObservations()
    mi = calc.computeAverageLocalOfObservations()

    num_surrogates = req.get("num_surrogates", 0)
    if num_surrogates > 0:
        null_dist = calc.computeSignificance(num_surrogates)
        null_mean = null_dist.getMeanOfDistribution()
        mi = mi - null_mean if mi >= null_mean else 0.0

    mi = nat2bit(mi)
    return mi if mi >= 0 else 0.0


def _calc_te(req):
    init_jvm()
    calc_class = jp.JPackage(
        "infodynamics.measures.continuous.kraskov"
    ).TransferEntropyCalculatorKraskov
    calc = calc_class()
    calc.setProperty("NUM_THREADS", str(int(req["num_threads"])))
    calc.setProperty("DELAY", str(int(req["delay"])))
    calc.setProperty("AUTO_EMBED_RAGWITZ_NUM_NNS", "4")
    calc.setProperty("k", str(int(req["k"])))
    calc.initialise()
    calc.startAddObservations()
    for ts_i, ts_j in req["pairs"]:
        calc.addObservations(ts_i, ts_j)
    calc.finaliseAddObservations()
    te = calc.computeAverageLocalOfObservations()

    num_surrogates = req.get("num_surrogates", 0)
    if num_surrogates > 0:
        null_dist = calc.computeSignificance(num_surrogates)
        null_mean = null_dist.getMeanOfDistribution()
        te = te - null_mean if te >= null_mean else 0.0

    return te if te >= 0 else 0.0


def _calc_entropy(req):
    init_jvm()
    calc_class = jp.JPackage(
        "infodynamics.measures.continuous.kozachenko"
    ).EntropyCalculatorMultiVariateKozachenko
    calc = calc_class()
    ts = req["ts"]

    if req["average"]:
        calc.initialise()
        flat = [v for row in ts for v in row]
        calc.setObservations(flat)
        return nat2bit(calc.computeAverageLocalOfObservations())

    values = []
    for row in ts:
        calc.initialise()
        calc.setObservations(row)
        values.append(nat2bit(calc.computeAverageLocalOfObservations()))
    return values


_OPS = {"mi": _calc_mi, "te": _calc_te, "entropy": _calc_entropy}


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req = json.loads(line)
        try:
            result = _OPS[req["op"]](req)
            resp = {"ok": True, "result": result}
        except Exception as exc:
            resp = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        sys.stdout.write(json.dumps(resp) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
