# Terraform Basics

## Providers and Resources

Terraform configurations are written in HCL and grouped into `.tf` files.
A `provider` block authenticates against a platform such as AWS, GCP, or
Kubernetes. A `resource` block declares one piece of infrastructure with
its type, a local name, and the arguments the provider needs. References
between resources use the `type.name.attribute` form.

## State

Terraform tracks what it manages in a state file. By default the state is
local, but production usage stores it in a remote backend such as S3 with
DynamoDB locking, so multiple operators do not corrupt each other's runs.
`terraform plan` shows the diff between desired configuration and current
state. `terraform apply` executes it.

## Modules and Variables

A module is a reusable group of resources in its own directory. The root
module calls child modules through `module` blocks, passing inputs and
reading outputs. Inputs are declared with `variable` blocks and outputs
with `output` blocks, so modules behave like functions over
infrastructure.
