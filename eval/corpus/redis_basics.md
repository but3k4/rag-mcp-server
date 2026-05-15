# Redis Basics

## Keys and Data Types

Redis is an in-memory key-value store. Beyond plain strings, values can be
lists, sets, sorted sets, hashes, and streams. Keys live in a single flat
namespace per database, and conventional colon-separated names like
`user:42:session` are used to imply hierarchy.

## Persistence

Two persistence modes are available. RDB takes periodic point-in-time
snapshots of the dataset and is cheap but loses recent writes on crash. AOF
appends every write command to a log and can be replayed on restart for near
zero data loss at the cost of larger files and more disk I/O. Both can be
enabled together.

## Pub/Sub and Streams

Pub/sub publishes messages to channels with no persistence, so subscribers
that are offline miss messages. Streams are an append-only log with consumer
groups, suitable when at-least-once delivery and replay matter.
