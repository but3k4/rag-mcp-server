# PostgreSQL Basics

## Tables and Indexes

Tables store rows under a defined schema. A primary key enforces uniqueness
and is automatically indexed. Secondary indexes speed up lookups on other
columns. B-tree is the default and works for equality and range scans, while
GIN suits JSONB and full-text search. Indexes have a write cost so add them
based on actual query patterns.

## Transactions and Isolation

Every statement runs in a transaction. The default isolation level is read
committed, which prevents dirty reads but allows non-repeatable reads. Use
`SERIALIZABLE` for stricter guarantees at the cost of more aborts. Wrap
multi-statement work in `BEGIN` and `COMMIT` to make it atomic.

## VACUUM and Bloat

PostgreSQL marks deleted rows as dead tuples rather than reclaiming space
immediately. Autovacuum reclaims that space and updates planner statistics
in the background. Tables with high update or delete churn may need tuned
autovacuum thresholds to avoid bloat and stale statistics.
