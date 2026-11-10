# MiniVectorDB（2026-competition-round2）

单机、单进程、单客户端的轻量向量数据库，通过 stdin/stdout 的 JSON Lines 协议与评测驱动通信。

## 构建与启动

```bash
docker build -t minivectordb:latest .
mkdir -p data && chmod 777 data
docker run --rm -i --network=none --cpus=2 --memory=2g \
  --mount type=bind,src="$PWD/data",dst=/data minivectordb:latest
```

容器前台运行，启动后先输出 `{"type":"ready"}`，随后逐行接收：

```json
{"id":1,"op":"execute","sql":"SELECT id FROM docs;","search_mode":"exact"}
{"id":2,"op":"shutdown"}
```

每条请求返回一行 JSON，`id` 原样回传；查询返回 `columns`/`rows`，修改返回 `affected_rows`，失败返回 `{"ok":false,"error":{"code":"...","message":"..."}}`。日志只写 stderr。

## 数据目录

所有持久化文件位于 `/data`（可用环境变量 `MVDB_DATA` 覆盖）：

- `wal.log`：每条已确认语句的 redo 记录，写前 fsync；
- `snapshot.json`：正常关闭时的全量快照，启动时先加载快照再重放 WAL。

启动时若存在 WAL，会逐行重放；末尾不完整的行（进程被强杀时的撕裂写）会被忽略。已确认提交的语句在强杀后仍可恢复，未提交事务只存在于内存，不写入 WAL。

## 支持的 SQL

- `CREATE TABLE name (id BIGINT PRIMARY KEY, name VARCHAR(n), vec VECTOR(n), ...)`（每表最多一个向量列）；
- `INSERT INTO t VALUES (...), (...);`（多行，原子）；
- `SELECT [*|列|L2_DISTANCE(col,'[...]') AS d] FROM t [WHERE a = 1 AND b = 'x'] [ORDER BY 列|别名 [ASC|DESC]] [LIMIT n];`
- `UPDATE t SET col = literal [, ...] [WHERE ...];`
- `DELETE FROM t [WHERE ...];`
- `CREATE VECTOR INDEX idx ON t(vec);`
- `BEGIN; COMMIT; ROLLBACK;`

语义约定：BIGINT 保持 64 位整数；VARCHAR 按 Unicode 字符数校验长度；向量按 float32 存储，输入拒绝 NaN/Infinity/维度不符/float32 溢出；`L2_DISTANCE` 返回欧氏距离；距离并列按主键升序；`ORDER BY` 并列按主键升序；限定 `LIMIT` 时先完成过滤再取 TopK；不支持 `ALTER TABLE` 等语句，返回 `ok:false` 且不影响事务内已成功语句。

`search_mode` 为 `ann` 时当前实现仍走精确扫描，保证返回结果正确（允许的配置选择）。

## 事务与恢复

- 未显式 `BEGIN` 时，每条成功的修改语句自动提交；
- 事务内的读操作看到本事务已完成的修改；
- 语句失败只回滚该语句，事务继续可用；
- `ROLLBACK` 丢弃事务内修改；
- `COMMIT` 先把 redo 记录追加并 fsync 到 `/data/wal.log`，再返回成功；
- 正常关闭时回滚未提交事务并写快照；异常强杀后由 WAL 重放恢复。

## 本地验证

```bash
python3 -m pip install numpy
python3 run_public_cases.py     # 20 个公开用例
```

公开用例覆盖 SQL CRUD、比较/排序、主键冲突、Unicode、BIGINT、向量距离、float32 语义、过滤与并列、索引维护、事务提交/回滚、关闭回滚、重启持久化。

## 已知限制

- 全量数据常驻内存，超过内存预算的超大数据集不在当前实现范围内；
- ANN 索引只记录元数据，检索始终精确；
- 不支持 JOIN、聚合、子查询、NULL、ALTER TABLE、主键更新。
