"""Layer 2 - SQLite catalog & inventory storage.

Stores saree records (SKU, fabric, region, price, image path), their extracted
design attributes (pattern class + dominant colours) and the historical sales
rows used by the sales-prediction layer. Pure standard-library sqlite3.
"""

import json
import sqlite3

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS sarees (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    sku             TEXT UNIQUE NOT NULL,
    title           TEXT NOT NULL,
    region          TEXT,
    fabric          TEXT,
    price           REAL,
    image_path      TEXT,
    pattern         TEXT,
    dominant_colors TEXT,          -- JSON array of {rgb,hex,fraction}
    created_at      TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS sales (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    saree_id   INTEGER NOT NULL REFERENCES sarees(id),
    units      INTEGER NOT NULL,
    revenue    REAL NOT NULL,
    sale_date  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_sales_saree ON sales(saree_id);
CREATE INDEX IF NOT EXISTS idx_sarees_pattern ON sarees(pattern);
"""


class Database(object):
    def __init__(self, path=None):
        self.path = path or config.DB_PATH
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")

    def init_schema(self):
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    # ---- sarees ----------------------------------------------------------
    def insert_saree(self, sku, title, region, fabric, price, image_path,
                     pattern=None, dominant_colors=None):
        cur = self.conn.execute(
            """INSERT INTO sarees
               (sku, title, region, fabric, price, image_path, pattern, dominant_colors)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (sku, title, region, fabric, price, image_path, pattern,
             json.dumps(dominant_colors) if dominant_colors is not None else None),
        )
        self.conn.commit()
        return cur.lastrowid

    def update_design(self, saree_id, pattern, dominant_colors):
        self.conn.execute(
            "UPDATE sarees SET pattern = ?, dominant_colors = ? WHERE id = ?",
            (pattern, json.dumps(dominant_colors), saree_id),
        )
        self.conn.commit()

    def get_saree(self, saree_id):
        row = self.conn.execute("SELECT * FROM sarees WHERE id = ?", (saree_id,)).fetchone()
        return self._row_to_saree(row) if row else None

    def all_sarees(self):
        rows = self.conn.execute("SELECT * FROM sarees ORDER BY id").fetchall()
        return [self._row_to_saree(r) for r in rows]

    def count_sarees(self):
        return self.conn.execute("SELECT COUNT(*) FROM sarees").fetchone()[0]

    # ---- sales -----------------------------------------------------------
    def insert_sale(self, saree_id, units, revenue, sale_date):
        self.conn.execute(
            "INSERT INTO sales (saree_id, units, revenue, sale_date) VALUES (?, ?, ?, ?)",
            (saree_id, units, revenue, sale_date),
        )
        self.conn.commit()

    def sales_by_saree(self):
        """Return {saree_id: {units, revenue}} aggregated over all sales."""
        rows = self.conn.execute(
            """SELECT saree_id, SUM(units) AS units, SUM(revenue) AS revenue
               FROM sales GROUP BY saree_id"""
        ).fetchall()
        return {r["saree_id"]: {"units": r["units"], "revenue": r["revenue"]} for r in rows}

    def sales_by_pattern(self):
        rows = self.conn.execute(
            """SELECT s.pattern AS pattern,
                      SUM(sa.units) AS units,
                      SUM(sa.revenue) AS revenue
               FROM sales sa JOIN sarees s ON s.id = sa.saree_id
               GROUP BY s.pattern ORDER BY units DESC"""
        ).fetchall()
        return [dict(r) for r in rows]

    def sales_timeseries(self):
        rows = self.conn.execute(
            """SELECT substr(sale_date, 1, 7) AS month,
                      SUM(units) AS units, SUM(revenue) AS revenue
               FROM sales GROUP BY month ORDER BY month"""
        ).fetchall()
        return [dict(r) for r in rows]

    # ---- helpers ---------------------------------------------------------
    @staticmethod
    def _row_to_saree(row):
        d = dict(row)
        if d.get("dominant_colors"):
            d["dominant_colors"] = json.loads(d["dominant_colors"])
        return d

    def close(self):
        self.conn.close()
