"""JSON line-protocol bridge between the Electron main process and the Python
engine (Layer 10 - the Node <-> Python bridge, invoked via python-shell).

Reads one JSON request object per line on stdin and writes one JSON response
object per line on stdout. All diagnostics go to stderr so stdout carries only
protocol JSON.

Request:  {"rid": 1, "cmd": "analyze", "path": "C:/img.png", "k": 6}
Response: {"rid": 1, "ok": true, "data": {...}}   |   {"rid": 1, "ok": false, "error": "..."}
"""

import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from viratra import config, embeddings, search           # noqa: E402
from viratra.pipeline import Engine                        # noqa: E402


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def send(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


class Server(object):
    def __init__(self):
        self.engine = Engine()

    # ---- command handlers ------------------------------------------------
    def cmd_ping(self, req):
        return {
            "status": "ok",
            "version": __import__("viratra").__version__,
            "embedding_backend": embeddings.backend_name(),
            "search_backend": search.backend_name(),
            "classifier": os.path.exists(config.CLASSIFIER_PATH),
            "index": os.path.exists(config.INDEX_PATH + ".meta.npy"),
        }

    def cmd_analyze(self, req):
        path = req["path"]
        if not os.path.exists(path):
            raise IOError("file not found: {}".format(path))
        return self.engine.analyze(path, k=int(req.get("k", 6)))

    def cmd_catalog(self, req):
        return {"sarees": self.engine.db.all_sarees()}

    def cmd_saree(self, req):
        from viratra import preprocess
        saree = self.engine.db.get_saree(int(req["id"]))
        if saree is None:
            raise ValueError("no saree with id {}".format(req["id"]))
        path = os.path.join(config.ROOT, saree["image_path"])
        prep = preprocess.prepare(path)
        saree["similar"] = self.engine.similar(prep, k=int(req.get("k", 6)),
                                               exclude_id=saree["id"])
        return saree

    def cmd_stats(self, req):
        db = self.engine.db
        return {
            "counts": {
                "sarees": db.count_sarees(),
                "sales_rows": db.conn.execute("SELECT COUNT(*) FROM sales").fetchone()[0],
            },
            "sales_by_pattern": db.sales_by_pattern(),
            "sales_timeseries": db.sales_timeseries(),
            "backends": {
                "embedding": embeddings.backend_name(),
                "search": search.backend_name(),
            },
        }

    def cmd_predict(self, req):
        model = self.engine.sales
        if model is None:
            raise RuntimeError("sales model not built")
        return {
            "predicted_units": round(model.predict(req["saree"]), 1),
            "drivers": model.feature_importance()[:5],
        }

    # ---- dispatch loop ---------------------------------------------------
    def run(self):
        handlers = {
            "ping": self.cmd_ping, "analyze": self.cmd_analyze,
            "catalog": self.cmd_catalog, "saree": self.cmd_saree,
            "stats": self.cmd_stats, "predict": self.cmd_predict,
        }
        log("viratra engine ready ({} / {})".format(
            embeddings.backend_name(), search.backend_name()))
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            rid = None
            try:
                req = json.loads(line)
                rid = req.get("rid")
                cmd = req.get("cmd")
                if cmd not in handlers:
                    raise ValueError("unknown command: {}".format(cmd))
                send({"rid": rid, "ok": True, "data": handlers[cmd](req)})
            except Exception as exc:  # noqa: BLE001 - report every error to the client
                log("error: {}".format(exc))
                send({"rid": rid, "ok": False, "error": str(exc)})


if __name__ == "__main__":
    Server().run()
