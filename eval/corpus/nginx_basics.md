# Nginx Basics

## Reverse Proxy

Nginx is most commonly deployed as a reverse proxy in front of application
servers. The `proxy_pass` directive forwards a request to an upstream backend,
optionally rewriting headers along the way. Group backends into an `upstream`
block to enable load balancing across multiple instances.

## Virtual Hosts

A `server` block defines a virtual host. The `listen` directive selects the
port and `server_name` selects which hostnames the block answers to. Multiple
`server` blocks can share the same port and Nginx routes by `Host` header.

## TLS Termination

To terminate TLS, point `ssl_certificate` and `ssl_certificate_key` at a PEM
chain and private key, then add `listen 443 ssl`. Disable old protocols by
setting `ssl_protocols TLSv1.2 TLSv1.3` and prefer modern ciphers. HTTP-to-HTTPS
redirect is a separate `server` block listening on port 80.
