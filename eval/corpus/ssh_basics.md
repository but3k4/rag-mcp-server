# SSH Basics

## Key-Based Authentication

SSH supports password and public-key authentication. Generate a keypair
with `ssh-keygen`, copy the public half to the server's
`~/.ssh/authorized_keys`, and the private half stays on the client. Protect
the private key with a passphrase and load it into an agent so each
connection does not re-prompt.

## Client Configuration

The `~/.ssh/config` file defines per-host shortcuts: a `Host` block sets
`HostName`, `User`, `Port`, `IdentityFile`, and other options for a chosen
alias. After configuring an entry you can connect with just `ssh alias`.
Wildcards match groups of hosts so common settings can be shared.

## Port Forwarding

Local forwarding with `-L local:host:remote` exposes a remote service on a
local port. Remote forwarding with `-R` does the reverse. Dynamic
forwarding with `-D` opens a SOCKS proxy through the SSH tunnel, useful for
ad-hoc browsing through a bastion host.
