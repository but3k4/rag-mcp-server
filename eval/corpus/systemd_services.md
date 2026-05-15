# systemd Services

## Unit Files

A service is defined by a unit file under `/etc/systemd/system/` or
`/usr/lib/systemd/system/`. The `[Service]` section sets `ExecStart`, the
working directory, and the user the process runs as. The `[Install]`
section's `WantedBy` decides which target pulls the unit in when enabled.

## Managing Services

`systemctl start`, `stop`, `restart`, and `reload` control a running unit.
`enable` makes the unit start at boot, `disable` removes that link.
`status` shows the current state and the most recent log lines. After
editing a unit file, run `systemctl daemon-reload` so systemd re-reads it.

## Logs and Targets

`journalctl -u name` shows the log for one unit and `-f` follows new
entries. Targets group units that should reach the same state together.
`multi-user.target` is the typical default for servers. Failed units are
listed by `systemctl --failed` and can be inspected with the same status
and journal commands.
