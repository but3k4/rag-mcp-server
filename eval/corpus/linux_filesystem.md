# Linux Filesystem

## Directory Layout

The Filesystem Hierarchy Standard organises a Linux root filesystem. `/etc`
holds system configuration, `/var` holds variable state like logs and caches,
`/usr` holds read-only program files, and `/home` holds user data. `/tmp` is
for transient files that may be wiped on reboot.

## Permissions

Every file has an owner, group, and mode. The mode encodes read, write, and
execute bits for owner, group, and others, displayed as nine characters by
`ls -l`. Change ownership with `chown` and the mode with `chmod`. Setuid and
setgid bits change the effective user or group when a binary runs.

## Mounting

A filesystem is attached to the tree at a mount point with `mount`, and
detached with `umount`. Entries in `/etc/fstab` describe persistent mounts
applied at boot. Bind mounts expose the same content under a second path,
useful for chroots and containers.
