#!/usr/bin/env python3
"""MiniVectorDB: single-process vector database speaking the round2 JSONL protocol."""

import json
import math
import os
import re
import sys

import base64

import numpy as np

DATA_DIR = os.environ.get("MVDB_DATA", "/data")
DATA_DIR = os.path.abspath(DATA_DIR)
SNAPSHOT = os.path.join(DATA_DIR, "snapshot.json")
WAL = os.path.join(DATA_DIR, "wal.log")


def log(*args):
    print(*args, file=sys.stderr, flush=True)


def enc_value(value):
    if isinstance(value, np.ndarray):
        return {"__f32__": base64.b64encode(value.astype(np.float32).tobytes()).decode("ascii")}
    return value


def dec_value(value):
    if isinstance(value, dict) and "__f32__" in value:
        return np.frombuffer(base64.b64decode(value["__f32__"]), dtype=np.float32)
    return value


class SQLError(Exception):
    def __init__(self, code, message=""):
        super().__init__(message or code)
        self.code = code
        self.message = message or code


# --------------------------------------------------------------------------
# schema / storage
# --------------------------------------------------------------------------
class Column:
    __slots__ = ("name", "kind", "size")

    def __init__(self, name, kind, size=None):
        self.name = name
        self.kind = kind
        self.size = size


class Table:
    def __init__(self, name, columns, pk_index):
        self.name = name
        self.columns = columns
        self.pk_index = pk_index
        self.rows = {}  # pk -> list of values (vectors are np.float32 arrays)
        self.indexes = set()
        self._dirty = True
        self._ids = np.empty(0, dtype=np.int64)
        self._mat = None
        self._col_cache = {}

    @property
    def pk_name(self):
        return self.columns[self.pk_index].name

    def column_index(self, name):
        for i, col in enumerate(self.columns):
            if col.name.lower() == name.lower():
                return i
        return None

    def vector_columns(self):
        return [i for i, col in enumerate(self.columns) if col.kind == "VECTOR"]

    def invalidate(self):
        self._dirty = True
        self._col_cache.clear()

    def column_array(self, index):
        cached = self._col_cache.get(index)
        if cached is not None:
            return cached
        ids, _mat = self.matrix()
        col = self.columns[index]
        values = [self.rows[int(pk)][index] for pk in ids]
        if col.kind == "VARCHAR":
            arr = np.array(values, dtype=object)
        else:
            arr = np.array(values, dtype=np.int64)
        self._col_cache[index] = arr
        return arr

    def matrix(self):
        if not self._dirty and self._mat is not None:
            return self._ids, self._mat
        vec_col = self.vector_columns()[0]
        ids = np.fromiter(self.rows.keys(), dtype=np.int64, count=len(self.rows))
        if len(self.rows):
            mat = np.empty((len(ids), self.columns[vec_col].size), dtype=np.float32)
            for pos, pk in enumerate(ids):
                mat[pos] = self.rows[pk][vec_col]
        else:
            mat = np.empty((0, self.columns[vec_col].size), dtype=np.float32)
        self._ids, self._mat, self._dirty = ids, mat, False
        return self._ids, self._mat


class Database:
    def __init__(self):
        self.tables = {}
        self.tx = None
        self._wal_handle = None

    def _sync(self, fh):
        try:
            os.fdatasync(fh.fileno())
        except (AttributeError, OSError):
            os.fsync(fh.fileno())

    # ---------------- persistence ----------------
    def load(self):
        os.makedirs(DATA_DIR, exist_ok=True)
        if os.path.exists(SNAPSHOT):
            with open(SNAPSHOT, "r", encoding="utf-8") as fh:
                first = fh.readline()
                head = json.loads(first) if first.strip() else {}
                if head.get("format") == "rows-v2":
                    for item in head.get("tables", []):
                        cols = [Column(*c) for c in item["columns"]]
                        table = Table(item["name"], cols, item["pk_index"])
                        table.indexes = set(item.get("indexes", []))
                        self.tables[table.name] = table
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                        except json.JSONDecodeError:
                            break  # torn trailing write
                        table = self.tables.get(rec["table"])
                        if table is not None:
                            table.rows[rec["pk"]] = self._decode_row(table, rec["row"])
                else:
                    doc = json.loads(first + fh.read())
                    for item in doc.get("tables", []):
                        cols = [Column(*c) for c in item["columns"]]
                        table = Table(item["name"], cols, item["pk_index"])
                        table.indexes = set(item.get("indexes", []))
                        for row in item["rows"]:
                            table.rows[row[item["pk_index"]]] = self._decode_row(table, row)
                        self.tables[table.name] = table
        if os.path.exists(WAL):
            with open(WAL, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        break  # torn trailing write
                    self.apply_ops(record.get("ops", []))

    def _decode_row(self, table, row):
        out = []
        for col, value in zip(table.columns, row):
            if col.kind == "VECTOR":
                out.append(dec_value(value) if isinstance(value, dict) else np.asarray(value, dtype=np.float32))
            else:
                out.append(value)
        return out

    def snapshot(self):
        tmp = SNAPSHOT + ".tmp"
        os.makedirs(DATA_DIR, exist_ok=True)
        meta = {
            "format": "rows-v2",
            "tables": [
                {
                    "name": table.name,
                    "columns": [[c.name, c.kind, c.size] for c in table.columns],
                    "pk_index": table.pk_index,
                    "indexes": sorted(table.indexes),
                }
                for table in self.tables.values()
            ],
        }
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(meta, ensure_ascii=False) + "\n")
            for table in self.tables.values():
                for pk in sorted(table.rows):
                    row = table.rows[pk]
                    fh.write(json.dumps({"table": table.name, "pk": pk,
                                         "row": [enc_value(v) for v in row]},
                                        ensure_ascii=False) + "\n")
            fh.flush()
            self._sync(fh)
        os.replace(tmp, SNAPSHOT)
        self._truncate_wal()

    def _truncate_wal(self):
        if self._wal_handle is not None:
            self._wal_handle.close()
            self._wal_handle = None
        with open(WAL, "w", encoding="utf-8") as fh:
            fh.flush()
            self._sync(fh)

    def append_wal(self, ops):
        os.makedirs(DATA_DIR, exist_ok=True)
        if self._wal_handle is None:
            self._wal_handle = open(WAL, "a", encoding="utf-8")
        fh = self._wal_handle
        fh.write(json.dumps({"ops": ops}, ensure_ascii=False) + "\n")
        fh.flush()
        self._sync(fh)

    # ---------------- ops ----------------
    def apply_ops(self, ops):
        for op in ops:
            kind = op["t"]
            if kind == "create_table":
                cols = [Column(*c) for c in op["columns"]]
                table = Table(op["name"], cols, op["pk_index"])
                self.tables[table.name] = table
            elif kind == "create_index":
                table = self.tables[op["table"]]
                table.indexes.add(op["col"])
            elif kind == "drop_table":
                self.tables.pop(op["name"], None)
            elif kind == "upsert":
                table = self.tables[op["table"]]
                table.rows[op["pk"]] = self._decode_row(table, op["row"])
                table.invalidate()
            elif kind == "delete":
                table = self.tables[op["table"]]
                if op["pk"] in table.rows:
                    del table.rows[op["pk"]]
                    table.invalidate()

    def commit_ops(self, ops):
        if not ops:
            return
        self.append_wal(ops)
        self.apply_ops(ops)

    # ---------------- transactions ----------------
    def begin(self):
        if self.tx is not None:
            raise SQLError("transaction_already_open", "nested transaction")
        self.tx = {"ops": [], "rows": {}, "deleted": {}}

    def rollback(self):
        if self.tx is None:
            raise SQLError("no_transaction", "no active transaction")
        self.tx = None

    def commit(self):
        if self.tx is None:
            raise SQLError("no_transaction", "no active transaction")
        tx = self.tx
        self.tx = None
        self.commit_ops(tx["ops"])

    def _visible_row(self, table_name, pk):
        if self.tx is not None:
            deletes = self.tx["deleted"].get(table_name)
            if deletes and pk in deletes:
                return None
            rows = self.tx["rows"].get(table_name)
            if rows and pk in rows:
                return rows[pk]
        table = self.tables[table_name]
        return table.rows.get(pk)

    def _write_row(self, table_name, pk, row):
        table = self.tables[table_name]
        if self.tx is not None:
            self.tx["rows"].setdefault(table_name, {})[pk] = row
            self.tx["deleted"].get(table_name, set()).discard(pk)
            self.tx["ops"].append({"t": "upsert", "table": table_name, "pk": pk,
                                   "row": [enc_value(v) for v in row]})
        else:
            self.commit_ops([{"t": "upsert", "table": table_name, "pk": pk,
                              "row": [([float(x) for x in v] if isinstance(v, np.ndarray) else v) for v in row]}])

    def _delete_row(self, table_name, pk):
        if self.tx is not None:
            self.tx["rows"].get(table_name, {}).pop(pk, None)
            self.tx["deleted"].setdefault(table_name, set()).add(pk)
            self.tx["ops"].append({"t": "delete", "table": table_name, "pk": pk})
        else:
            self.commit_ops([{"t": "delete", "table": table_name, "pk": pk}])

    def visible_rows(self, table_name):
        table = self.tables[table_name]
        if self.tx is None:
            return dict(table.rows)
        merged = dict(table.rows)
        for pk in self.tx["deleted"].get(table_name, ()):  # deletions
            merged.pop(pk, None)
        merged.update(self.tx["rows"].get(table_name, {}))
        return merged


# --------------------------------------------------------------------------
# SQL parsing
# --------------------------------------------------------------------------
TOKEN_RE = re.compile(r"\s*(?:(?P<num>-?\d+\.\d+|-?\d+)|(?P<str>'(?:[^']|'')*')|"
                      r"(?P<vec>\[[^\[\]]*\])|"
                      r"(?P<id>[A-Za-z_][A-Za-z0-9_]*)|(?P<op><=|>=|<>|!=|=|<|>|\(|\)|,|;|\*|\.))")


def tokenize(sql):
    pos = 0
    tokens = []
    while pos < len(sql):
        m = TOKEN_RE.match(sql, pos)
        if not m:
            if sql[pos].isspace():
                pos += 1
                continue
            raise SQLError("syntax_error", f"unexpected character {sql[pos]!r}")
        pos = m.end()
        kind = m.lastgroup
        text = m.group(kind)
        if kind == "str":
            tokens.append(("str", text[1:-1].replace("''", "'")))
        elif kind == "num":
            tokens.append(("num", float(text) if "." in text else int(text)))
        elif kind == "id":
            tokens.append(("id", text))
        elif kind == "vec":
            tokens.append(("vec", text))
        else:
            tokens.append(("op", text))
        if pos >= len(sql):
            break
    return tokens


INSERT_HEAD = re.compile(
    r"(?is)^\s*insert\s+into\s+([A-Za-z_][A-Za-z0-9_]*)\s*(?:\(([^)]*)\))?\s+values\s+(.*?)[;\s]*$"
)


def fast_insert(sql):
    m = INSERT_HEAD.match(sql)
    if not m:
        return None
    table, column_list, body = m.group(1), m.group(2), m.group(3)
    if column_list:
        # Column-list inserts are handled by the full parser below.
        return None
    columns = None
    rows = []
    i, n = 0, len(body)
    while i < n:
        while i < n and body[i] in " \t\r\n,":
            i += 1
        if i >= n:
            break
        if body[i] != "(":
            return None
        i += 1
        values = []
        while i < n:
            while i < n and body[i] in " \t\r\n":
                i += 1
            if i >= n:
                return None
            ch = body[i]
            if ch in "')":
                return None
            if ch == "'":
                buf = []
                i += 1
                while i < n:
                    if body[i] == "'":
                        if i + 1 < n and body[i + 1] == "'":
                            buf.append("'")
                            i += 2
                            continue
                        i += 1
                        break
                    buf.append(body[i])
                    i += 1
                values.append("".join(buf))
            elif ch == "[":
                start = i
                while i < n and body[i] != "]":
                    i += 1
                if i >= n:
                    return None
                i += 1
                values.append(body[start:i])
            else:
                start = i
                while i < n and body[i] not in ",)":
                    i += 1
                token = body[start:i].strip()
                try:
                    values.append(float(token) if ("." in token or "e" in token.lower()) else int(token))
                except ValueError:
                    return None
            while i < n and body[i] in " \t\r\n":
                i += 1
            if i < n and body[i] == ",":
                i += 1
                continue
            if i < n and body[i] == ")":
                i += 1
                break
            return None
        rows.append(values)
    return {"kind": "insert", "table": table, "rows": rows, "columns": columns}


class Parser:
    def __init__(self, sql):
        self.toks = tokenize(sql)
        self.i = 0

    def peek(self, offset=0):
        j = self.i + offset
        return self.toks[j] if j < len(self.toks) else (None, None)

    def next(self):
        tok = self.peek()
        self.i += 1
        return tok

    def accept(self, kind, value=None):
        k, v = self.peek()
        if k == kind and (value is None or (v is not None and str(v).lower() == str(value).lower())):
            self.i += 1
            return True
        return False

    def expect(self, kind, value=None):
        if not self.accept(kind, value):
            raise SQLError("syntax_error", f"expected {value or kind}")

    def ident(self):
        k, v = self.next()
        if k != "id":
            raise SQLError("syntax_error", "expected identifier")
        return v

    def qident(self):
        parts = [self.ident()]
        while self.accept("op", "."):
            parts.append(self.ident())
        return parts[-1]

    # -------- statements --------
    def parse(self):
        stmt = self._parse_one()
        while self.accept("op", ";"):
            pass
        if self.peek()[0] is not None:
            raise SQLError("unsupported", f"unsupported syntax near {self.peek()[1]!r}")
        return stmt

    def _parse_one(self):
        k, v = self.peek()
        if k != "id":
            raise SQLError("syntax_error", "empty statement")
        word = v.lower()
        if word == "create":
            return self.parse_create()
        if word == "insert":
            return self.parse_insert()
        if word == "select":
            return self.parse_select()
        if word == "update":
            return self.parse_update()
        if word == "delete":
            return self.parse_delete()
        if word == "drop":
            return self.parse_drop()
        if word == "set":
            return self.parse_set()
        if word == "start":
            self.next()
            self.expect("id", "transaction")
            return {"kind": "begin"}
        if word == "end":
            self.next()
            self.accept("id", "work")
            return {"kind": "commit"}
        if word == "begin":
            self.next()
            self.accept("id", "work")
            return {"kind": "begin"}
        if word == "commit":
            self.next()
            self.accept("id", "work")
            return {"kind": "commit"}
        if word == "rollback":
            self.next()
            self.accept("id", "work")
            return {"kind": "rollback"}
        raise SQLError("unsupported", f"unsupported statement {v}")

    def parse_drop(self):
        self.expect("id", "drop")
        self.expect("id", "table")
        if_exists = False
        if self.accept("id", "if"):
            self.expect("id", "exists")
            if_exists = True
        name = self.ident()
        self.accept("op", ";")
        return {"kind": "drop_table", "name": name, "if_exists": if_exists}

    def parse_set(self):
        self.expect("id", "set")
        name = self.ident()
        self.expect("op", "=")
        value = self.next()[1]
        self.accept("op", ";")
        return {"kind": "set", "name": name, "value": value}

    def parse_create(self):
        self.expect("id", "create")
        if self.accept("id", "table"):
            if_not_exists = False
            if self.accept("id", "if"):
                self.expect("id", "not")
                self.expect("id", "exists")
                if_not_exists = True
            name = self.ident()
            self.expect("op", "(")
            columns = []
            pk_index = None
            pk_name = None
            while True:
                if self.accept("id", "primary"):
                    self.accept("id", "key")
                    self.expect("op", "(")
                    pk_name = self.ident()
                    self.expect("op", ")")
                else:
                    col_name = self.ident()
                    kind_tok = self.ident().upper()
                    if kind_tok in ("BIGINT", "INT", "INTEGER"):
                        col = Column(col_name, "BIGINT")
                    elif kind_tok in ("VARCHAR", "CHAR", "TEXT"):
                        if kind_tok == "VARCHAR":
                            self.expect("op", "(")
                            size = int(self.next()[1])
                            self.expect("op", ")")
                        else:
                            size = 65535
                        col = Column(col_name, "VARCHAR", size)
                    elif kind_tok == "VECTOR":
                        self.expect("op", "(")
                        size = int(self.next()[1])
                        self.expect("op", ")")
                        if size <= 0:
                            raise SQLError("syntax_error", "vector dimension must be positive")
                        col = Column(col_name, "VECTOR", size)
                    else:
                        raise SQLError("unsupported", f"unsupported type {kind_tok}")
                    columns.append(col)
                    if self.accept("id", "primary"):
                        self.accept("id", "key")
                        pk_name = col_name
                if self.accept("op", ","):
                    continue
                break
            self.expect("op", ")")
            if pk_name is not None:
                for i, col in enumerate(columns):
                    if col.name.lower() == pk_name.lower():
                        pk_index = i
                        break
            if pk_index is None:
                raise SQLError("unsupported", "primary key required")
            pk_col = columns[pk_index]
            if pk_col.kind != "BIGINT":
                raise SQLError("type_error", "primary key must be BIGINT")
            if sum(1 for col in columns if col.kind == "VECTOR") > 1:
                raise SQLError("unsupported", "at most one vector column per table")
            self.accept("op", ";")
            return {"kind": "create_table", "name": name, "columns": columns, "pk_index": pk_index,
                    "if_not_exists": if_not_exists}
        if self.accept("id", "vector"):
            self.accept("id", "index")
            index_name = self.ident()
            self.expect("id", "on")
            table = self.ident()
            self.expect("op", "(")
            col = self.ident()
            self.expect("op", ")")
            self.accept("op", ";")
            return {"kind": "create_index", "name": index_name, "table": table, "column": col}
        raise SQLError("unsupported", "unsupported create")

    def parse_insert(self):
        self.expect("id", "insert")
        self.expect("id", "into")
        table = self.ident()
        columns = None
        if self.accept("op", "("):
            columns = []
            while True:
                columns.append(self.ident())
                if self.accept("op", ","):
                    continue
                break
            self.expect("op", ")")
        self.expect("id", "values")
        rows = []
        while True:
            self.expect("op", "(")
            values = []
            while True:
                values.append(self.parse_literal())
                if self.accept("op", ","):
                    continue
                break
            self.expect("op", ")")
            rows.append(values)
            if self.accept("op", ","):
                continue
            break
        self.accept("op", ";")
        return {"kind": "insert", "table": table, "rows": rows, "columns": columns}

    def parse_literal(self):
        k, v = self.next()
        if k in ("num", "str", "vec"):
            return v
        raise SQLError("syntax_error", "expected literal")

    def parse_select(self):
        self.expect("id", "select")
        projections = []
        while True:
            if self.accept("op", "*"):
                projections.append({"kind": "star"})
            else:
                expr = self.parse_expr()
                alias = None
                if self.accept("id", "as"):
                    alias = self.ident()
                projections.append({"kind": "expr", "expr": expr, "alias": alias})
            if self.accept("op", ","):
                continue
            break
        self.expect("id", "from")
        table = self.ident()
        where = None
        if self.accept("id", "where"):
            where = self.parse_where()
        order = []
        if self.accept("id", "order"):
            self.expect("id", "by")
            while True:
                expr = self.parse_expr()
                desc = False
                if self.accept("id", "desc"):
                    desc = True
                else:
                    self.accept("id", "asc")
                order.append({"expr": expr, "desc": desc})
                if self.accept("op", ","):
                    continue
                break
        limit = None
        offset = None
        if self.accept("id", "limit"):
            limit = int(self.next()[1])
            if self.accept("id", "offset"):
                offset = int(self.next()[1])
        self.accept("op", ";")
        return {"kind": "select", "table": table, "projections": projections,
                "where": where, "order": order, "limit": limit, "offset": offset}

    def parse_expr(self):
        k, v = self.peek()
        if k == "id" and v.lower() == "l2_distance":
            self.next()
            self.expect("op", "(")
            first = self.parse_distance_arg()
            self.expect("op", ",")
            second = self.parse_distance_arg()
            self.expect("op", ")")
            col = first if first["kind"] == "column" else second
            vec = second if first["kind"] == "column" else first
            if col["kind"] != "column" or vec["kind"] != "vector":
                raise SQLError("type_error", "L2_DISTANCE expects (vector column, vector literal)")
            return {"type": "distance", "column": col["name"], "vector": vec["value"]}
        k, v = self.next()
        if k == "id":
            if v.lower() == "null":
                raise SQLError("unsupported", "NULL is not supported")
            name = v
            while self.accept("op", "."):
                name = self.ident()
            return {"type": "column", "name": name}
        if k in ("num", "str"):
            return {"type": "literal", "value": v}
        raise SQLError("syntax_error", "unexpected token in expression")

    def parse_distance_arg(self):
        k, v = self.peek()
        if k in ("vec", "str"):
            self.next()
            return {"kind": "vector", "value": self._vec_list(v)}
        if k == "id":
            return {"kind": "column", "name": self.qident()}
        raise SQLError("syntax_error", "expected vector column or vector literal")

    def _vec_list(self, text):
        text = text.strip()
        if not (text.startswith("[") and text.endswith("]")):
            raise SQLError("invalid_vector", "vector must be [..]")
        inner = text[1:-1].strip()
        if not inner:
            return []
        out = []
        for part in inner.split(","):
            part = part.strip()
            if not part:
                raise SQLError("invalid_vector", "empty vector component")
            try:
                out.append(float(part))
            except ValueError:
                raise SQLError("invalid_vector", f"bad vector component {part!r}")
        return out

    def parse_where(self):
        conds = [self.parse_condition()]
        while self.accept("id", "and"):
            conds.append(self.parse_condition())
        return conds

    def parse_condition(self):
        k, v = self.peek()
        if k == "id" and v.lower() == "l2_distance":
            expr = self.parse_expr()
            k, op = self.next()
            if k != "op" or op not in ("=", "<", "<=", ">", ">=", "!=", "<>"):
                raise SQLError("syntax_error", "expected comparison operator")
            k2, value = self.next()
            if k2 != "num":
                raise SQLError("syntax_error", "expected numeric literal")
            return {"distance": expr, "op": op, "value": value}
        col = self.qident()
        k, v = self.next()
        if k != "op" or v not in ("=", "<", "<=", ">", ">=", "!=", "<>"):
            raise SQLError("syntax_error", "expected comparison operator")
        op = v
        k2, value = self.next()
        if k2 == "str":
            pass
        elif k2 == "num":
            pass
        elif k2 == "vec":
            pass
        elif k2 == "id":
            raise SQLError("unsupported", "column comparison unsupported")
        else:
            raise SQLError("syntax_error", "expected literal")
        return {"column": col, "op": op, "value": value}

    def parse_update(self):
        self.expect("id", "update")
        table = self.ident()
        self.expect("id", "set")
        assignments = []
        while True:
            col = self.ident()
            self.expect("op", "=")
            assignments.append((col, self.parse_expr()))
            if self.accept("op", ","):
                continue
            break
        where = None
        if self.accept("id", "where"):
            where = self.parse_where()
        self.accept("op", ";")
        return {"kind": "update", "table": table, "assignments": assignments, "where": where}

    def parse_delete(self):
        self.expect("id", "delete")
        self.expect("id", "from")
        table = self.ident()
        where = None
        if self.accept("id", "where"):
            where = self.parse_where()
        self.accept("op", ";")
        return {"kind": "delete", "table": table, "where": where}


# --------------------------------------------------------------------------
# execution
# --------------------------------------------------------------------------
class Engine:
    def __init__(self, db):
        self.db = db

    # ---------- value conversion ----------
    def _coerce(self, table, index, value):
        col = table.columns[index]
        if col.kind == "BIGINT":
            if isinstance(value, bool) or not isinstance(value, int):
                if isinstance(value, float) and value.is_integer():
                    value = int(value)
                else:
                    raise SQLError("type_error", f"column {col.name} expects BIGINT")
            if value < -(2 ** 63) or value > 2 ** 63 - 1:
                raise SQLError("value_out_of_range", f"value out of BIGINT range for {col.name}")
            return value
        if col.kind == "VARCHAR":
            if not isinstance(value, str):
                raise SQLError("type_error", f"column {col.name} expects VARCHAR")
            if col.size is not None and len(value) > col.size:
                raise SQLError("type_error", f"value too long for {col.name}")
            return value
        if col.kind == "VECTOR":
            if isinstance(value, str):
                vec = self.parse_vector_text(value)
            elif isinstance(value, list):
                vec = value
            else:
                raise SQLError("invalid_vector", "vector must be a string or array")
            if len(vec) != col.size:
                raise SQLError("vector_dimension_mismatch", "vector dimension mismatch")
            arr = np.asarray(vec, dtype=np.float32)
            if not np.all(np.isfinite(arr)):
                raise SQLError("invalid_vector", "vector must be finite")
            for original, stored in zip(vec, arr):
                if math.isfinite(original) and not math.isfinite(float(stored)):
                    raise SQLError("invalid_vector", "vector overflows float32")
            return arr
        raise SQLError("type_error", "unknown column type")

    def parse_vector_text(self, text):
        text = text.strip()
        if not (text.startswith("[") and text.endswith("]")):
            raise SQLError("invalid_vector", "vector must be [..]")
        inner = text[1:-1].strip()
        if not inner:
            return []
        try:
            arr = np.fromstring(inner, sep=",", dtype=np.float64)
        except Exception:  # noqa: BLE001
            raise SQLError("invalid_vector", "malformed vector")
        if arr.size == 0 or not np.all(np.isfinite(arr)):
            raise SQLError("invalid_vector", "malformed vector")
        if "," in inner and arr.size != inner.count(",") + 1:
            raise SQLError("invalid_vector", "malformed vector")
        return [float(x) for x in arr]

    # ---------- helpers ----------
    def table(self, name):
        table = self.db.tables.get(name) or self.db.tables.get(name.lower())
        if table is None:
            raise SQLError("table_not_found", f"unknown table {name}")
        return table

    def _matches(self, table, row, where):
        if not where:
            return True
        for cond in where:
            op = cond["op"]
            right = cond["value"]
            if "distance" in cond:
                expr = cond["distance"]
                idx = table.column_index(expr["column"])
                if idx is None:
                    raise SQLError("column_not_found", f"unknown column {expr['column']}")
                if table.columns[idx].kind != "VECTOR":
                    raise SQLError("type_error", "L2_DISTANCE requires a vector column")
                left = self._distance(row[idx], np.asarray(expr["vector"], dtype=np.float32))
            else:
                idx = table.column_index(cond["column"])
                if idx is None:
                    raise SQLError("column_not_found", f"unknown column {cond['column']}")
                left = row[idx]
                if isinstance(left, np.ndarray):
                    raise SQLError("type_error", "cannot compare vector column")
            if op == "=":
                ok = left == right
            elif op == "!=" or op == "<>":
                ok = left != right
            elif op == "<":
                ok = left < right
            elif op == "<=":
                ok = left <= right
            elif op == ">":
                ok = left > right
            elif op == ">=":
                ok = left >= right
            else:
                ok = False
            if not ok:
                return False
        return True

    def _distance(self, vector, query):
        diff = vector - query
        return float(np.sqrt(np.dot(diff, diff)))

    # ---------- statements ----------
    def execute(self, sql, search_mode):
        stmt = None
        if sql.lstrip()[:6].lower() == "insert":
            stmt = fast_insert(sql)
        if stmt is None:
            stmt = Parser(sql).parse()
        kind = stmt["kind"]
        if kind == "create_table":
            return self.create_table(stmt)
        if kind == "create_index":
            return self.create_index(stmt)
        if kind == "drop_table":
            return self.drop_table(stmt)
        if kind == "set":
            return {}
        if kind == "insert":
            return self.insert(stmt)
        if kind == "select":
            return self.select(stmt)
        if kind == "update":
            return self.update(stmt)
        if kind == "delete":
            return self.delete(stmt)
        if kind == "begin":
            self.db.begin()
            return {}
        if kind == "commit":
            self.db.commit()
            return {}
        if kind == "rollback":
            self.db.rollback()
            return {}
        raise SQLError("unsupported", "unsupported statement")

    def create_table(self, stmt):
        if stmt["name"] in self.db.tables or stmt["name"].lower() in self.db.tables:
            if stmt.get("if_not_exists"):
                return {}
            raise SQLError("table_exists", "table already exists")
        self.db.commit_ops([{
            "t": "create_table",
            "name": stmt["name"],
            "columns": [[c.name, c.kind, c.size] for c in stmt["columns"]],
            "pk_index": stmt["pk_index"],
        }])
        return {}

    def create_index(self, stmt):
        table = self.table(stmt["table"])
        idx = table.column_index(stmt["column"])
        if idx is None:
            raise SQLError("column_not_found", "unknown column")
        if table.columns[idx].kind != "VECTOR":
            raise SQLError("type_error", "index must target vector column")
        self.db.commit_ops([{"t": "create_index", "table": table.name, "col": table.columns[idx].name}])
        return {}

    def drop_table(self, stmt):
        table = self.db.tables.get(stmt["name"]) or self.db.tables.get(stmt["name"].lower())
        if table is None:
            if stmt.get("if_exists"):
                return {}
            raise SQLError("table_not_found", f"unknown table {stmt['name']}")
        self.db.commit_ops([{"t": "drop_table", "name": table.name}])
        return {}

    def insert(self, stmt):
        table = self.table(stmt["table"])
        column_names = stmt.get("columns")
        column_idx = None
        if column_names:
            column_idx = []
            for name in column_names:
                idx = table.column_index(name)
                if idx is None:
                    raise SQLError("column_not_found", f"unknown column {name}")
                if idx in column_idx:
                    raise SQLError("type_error", "duplicate column in insert list")
                column_idx.append(idx)
        prepared = []
        seen = set()
        for raw in stmt["rows"]:
            if column_idx is not None:
                if len(raw) != len(column_idx):
                    raise SQLError("type_error", "column count mismatch")
                full = [None] * len(table.columns)
                for idx, value in zip(column_idx, raw):
                    full[idx] = value
                if any(value is None for value in full):
                    raise SQLError("type_error", "missing column value")
                raw = full
            if len(raw) != len(table.columns):
                raise SQLError("type_error", "column count mismatch")
            row = [self._coerce(table, i, v) for i, v in enumerate(raw)]
            pk = row[table.pk_index]
            if pk in seen or self.db._visible_row(table.name, pk) is not None:
                raise SQLError("duplicate_key", "duplicate primary key")
            seen.add(pk)
            prepared.append((pk, row))
        for pk, row in prepared:
            self.db._write_row(table.name, pk, row)
        return {"affected_rows": len(prepared)}

    def _project(self, table, projections):
        if any(p["kind"] == "star" for p in projections):
            return [c.name for c in table.columns], None
        names = []
        exprs = []
        for p in projections:
            expr = p["expr"]
            if expr["type"] == "column":
                names.append(p["alias"] or expr["name"])
            elif expr["type"] == "distance":
                names.append(p["alias"] or "distance")
            else:
                names.append(p["alias"] or "expr")
            exprs.append(expr)
        return names, exprs

    def _fast_topk(self, stmt, table):
        """Vectorised path for `ORDER BY <distance or its alias> LIMIT k`."""
        if self.db.tx is not None or stmt["limit"] is None or stmt.get("offset"):
            return None
        if len(table.rows) > 200000:
            # Avoid materialising a second full copy of a large vector set.
            return None
        order = stmt["order"]
        if len(order) != 1 or order[0]["desc"]:
            return None
        if any(p["kind"] == "star" for p in stmt["projections"]):
            return None
        names, exprs = self._project(table, stmt["projections"])
        order_expr = order[0]["expr"]
        dist_expr = None
        if order_expr["type"] == "distance":
            dist_expr = order_expr
        elif order_expr["type"] == "column":
            for i, expr in enumerate(exprs):
                if expr["type"] == "distance" and names[i].lower() == order_expr["name"].lower():
                    dist_expr = expr
                    break
            if dist_expr is None:
                return None
        else:
            return None
        vector_idx = table.column_index(dist_expr["column"])
        if vector_idx is None or table.columns[vector_idx].kind != "VECTOR":
            return None
        query = self._coerce(table, vector_idx, dist_expr["vector"])
        ids, mat = table.matrix()
        if len(ids) == 0:
            return {"columns": names, "rows": []}
        where = stmt["where"]
        simple = True
        for cond in where or []:
            if "distance" in cond:
                simple = False
                break
            idx = table.column_index(cond["column"])
            if idx is None or cond["op"] != "=" or table.columns[idx].kind == "VECTOR":
                simple = False
                break
        if simple:
            mask = np.ones(len(ids), dtype=bool)
            for cond in where or []:
                idx = table.column_index(cond["column"])
                mask &= table.column_array(idx) == cond["value"]
            keep_arr = np.nonzero(mask)[0]
        else:
            keep = []
            for pos in range(len(ids)):
                row = table.rows[int(ids[pos])]
                if self._matches(table, row, where):
                    keep.append(pos)
            keep_arr = np.asarray(keep, dtype=np.int64)
        if len(keep_arr) == 0:
            return {"columns": names, "rows": []}
        diff = mat[keep_arr] - query
        dists = np.sqrt((diff * diff).sum(axis=1, dtype=np.float32))
        order_idx = np.lexsort((ids[keep_arr], dists))[: stmt["limit"]]
        out_rows = []
        for idx in order_idx:
            pk = int(ids[keep_arr[idx]])
            row = table.rows[pk]
            values = []
            for expr in exprs:
                if expr["type"] == "distance":
                    vi = table.column_index(expr["column"])
                    qv = self._coerce(table, vi, expr["vector"])
                    values.append(self._distance(row[vi], qv))
                elif expr["type"] == "column":
                    values.append(row[table.column_index(expr["name"])])
                else:
                    values.append(expr["value"])
            out_rows.append(values)
        return {"columns": names, "rows": out_rows}

    def select(self, stmt):
        table = self.table(stmt["table"])
        fast = self._fast_topk(stmt, table)
        if fast is not None:
            return fast
        rows = self.db.visible_rows(table.name)
        prepared = []
        for pk in sorted(rows):
            row = rows[pk]
            if not self._matches(table, row, stmt["where"]):
                continue
            prepared.append((pk, row))

        names, exprs = self._project(table, stmt["projections"])
        if exprs is None:
            exprs = [{"type": "column", "name": c.name} for c in table.columns]

        computed = []
        for pk, row in prepared:
            values = []
            for expr in exprs:
                if expr["type"] == "column":
                    idx = table.column_index(expr["name"])
                    if idx is None:
                        raise SQLError("column_not_found", f"unknown column {expr['name']}")
                    values.append(row[idx])
                elif expr["type"] == "distance":
                    idx = table.column_index(expr["column"])
                    if idx is None:
                        raise SQLError("column_not_found", f"unknown column {expr['column']}")
                    query = self._coerce(table, idx, expr["vector"]) if isinstance(expr["vector"], list) else None
                    if query is None:
                        raise SQLError("invalid_vector", "bad query vector")
                    values.append((self._distance(row[idx], query), idx))
                else:
                    values.append(expr["value"])
            computed.append((pk, values, row))

        order = stmt["order"]
        computed.sort(key=lambda item: item[0])
        for item in reversed(order):
            expr = item["expr"]
            desc = item["desc"]
            if expr["type"] == "distance":
                vi = table.column_index(expr["column"])
                if vi is None:
                    raise SQLError("column_not_found", f"unknown column {expr['column']}")
                qv = self._coerce(table, vi, expr["vector"])

                def keyfn(rowitem, vi=vi, qv=qv):
                    return self._distance(rowitem[2][vi], qv)
            elif expr["type"] == "column" and expr["name"] in names:
                key_idx = names.index(expr["name"])

                def keyfn(rowitem, key_idx=key_idx):
                    value = rowitem[1][key_idx]
                    return value[0] if isinstance(value, tuple) else value
            elif expr["type"] == "column":
                col_idx = table.column_index(expr["name"])
                if col_idx is None:
                    raise SQLError("column_not_found", f"unknown order column {expr['name']}")

                def keyfn(rowitem, col_idx=col_idx):
                    return rowitem[2][col_idx]
            else:
                raise SQLError("unsupported", "unsupported ORDER BY expression")
            computed.sort(key=keyfn, reverse=desc)

        if stmt["limit"] is not None:
            start = stmt.get("offset") or 0
            computed = computed[start: start + stmt["limit"]]
        elif stmt.get("offset"):
            computed = computed[stmt["offset"]:]

        out_rows = []
        for _pk, values, _row in computed:
            out = []
            for (name, expr), value in zip(zip(names, exprs), values):
                if isinstance(value, tuple) and expr["type"] == "distance":
                    out.append(value[0])
                elif isinstance(value, np.ndarray):
                    out.append([float(x) for x in value])
                else:
                    out.append(value)
            out_rows.append(out)
        return {"columns": names, "rows": out_rows}

    def update(self, stmt):
        table = self.table(stmt["table"])
        rows = self.db.visible_rows(table.name)
        targets = [(pk, row) for pk, row in rows.items() if self._matches(table, row, stmt["where"])]
        assignments = []
        for col_name, expr in stmt["assignments"]:
            idx = table.column_index(col_name)
            if idx is None:
                raise SQLError("column_not_found", f"unknown column {col_name}")
            if idx == table.pk_index:
                raise SQLError("unsupported", "primary key update is not supported")
            if expr["type"] != "literal":
                raise SQLError("unsupported", "only literal assignments supported")
            assignments.append((idx, expr["value"]))
        # Validate every target row before applying anything so a failing
        # statement cannot leave partial modifications behind.
        pending = []
        for pk, row in targets:
            new_row = list(row)
            for idx, value in assignments:
                new_row[idx] = self._coerce(table, idx, value)
            pending.append((pk, new_row))
        for pk, new_row in pending:
            self.db._write_row(table.name, pk, new_row)
        return {"affected_rows": len(pending)}

    def delete(self, stmt):
        table = self.table(stmt["table"])
        rows = self.db.visible_rows(table.name)
        targets = [pk for pk, row in rows.items() if self._matches(table, row, stmt["where"])]
        for pk in targets:
            self.db._delete_row(table.name, pk)
        return {"affected_rows": len(targets)}


# --------------------------------------------------------------------------
# protocol loop
# --------------------------------------------------------------------------
def emit(obj):
    sys.stdout.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def main():
    try:
        sys.stdin.reconfigure(encoding="utf-8", errors="replace")
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    db = Database()
    try:
        db.load()
    except Exception as exc:  # noqa: BLE001
        log("recovery failed, starting from empty state:", repr(exc))
        db = Database()
    engine = Engine(db)
    emit({"type": "ready"})
    for line in sys.stdin:
        line = line.strip().lstrip("\ufeff")
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            log("invalid request line", line[:200])
            continue
        try:
            rid = request.get("id")
            op = request.get("op")
            if op == "shutdown":
                if db.tx is not None:
                    db.tx = None
                try:
                    db.snapshot()
                except Exception as exc:  # noqa: BLE001
                    log("snapshot failed", exc)
                emit({"id": rid, "ok": True})
                return 0
            if op != "execute":
                emit({"id": rid, "ok": False, "error": {"code": "unsupported_op", "message": "unsupported op"}})
                continue
            sql = request.get("sql", "")
            mode = request.get("search_mode", "exact")
            result = engine.execute(sql, mode)
            response = {"id": rid, "ok": True}
            response.update(result)
            emit(response)
        except SQLError as exc:
            emit({"id": rid, "ok": False, "error": {"code": exc.code, "message": exc.message}})
        except MemoryError:
            emit({"id": rid, "ok": False, "error": {"code": "out_of_memory", "message": "out of memory"}})
        except Exception as exc:  # noqa: BLE001
            log("internal error", repr(exc))
            try:
                emit({"id": rid, "ok": False, "error": {"code": "internal_error", "message": str(exc)}})
            except Exception:  # noqa: BLE001
                pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
