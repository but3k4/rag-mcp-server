# Kubernetes Pods

## Pods and Deployments

A pod is the smallest deployable unit in Kubernetes and groups one or more
containers that share network and storage. Pods are usually managed by a
higher-level object such as a Deployment, which keeps a desired number of
identical replicas running and handles rolling updates. Deleting a pod
managed by a Deployment causes the controller to recreate it.

## Services

A Service exposes a set of pods under a stable virtual IP and DNS name.
The selector matches pod labels rather than naming pods directly, so
replicas can come and go without breaking clients. Service types include
ClusterIP for in-cluster access, NodePort for fixed node ports, and
LoadBalancer for cloud-provisioned external entry points.

## Configuration

ConfigMaps hold non-sensitive configuration as key-value pairs and can be
mounted as files or injected as environment variables. Secrets carry the
same shape for sensitive values and are base64-encoded at rest. Both are
referenced from pod specs and updated independently of the container
image.
