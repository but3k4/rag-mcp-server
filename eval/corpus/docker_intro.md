# Docker Introduction

## Images

A Docker image is a read-only template containing the application and its
dependencies. Build an image with `docker build`. Tag it with a name and
version for distribution. Images live in a registry such as Docker Hub or a
private one your team controls.

## Containers

A container is a running instance of an image. Start one with `docker run`.
Each container has its own filesystem, network namespace, and process tree.
Containers are ephemeral by default. Use volumes to persist data across
container restarts or to share data with the host.

## Networking

Containers attach to a Docker network. The default bridge network gives each
container an internal IP. Use `-p host:container` to publish a port to the
host. For multi-container apps, define a user-defined bridge network and
refer to other containers by their service name.
