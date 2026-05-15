# HTTP Basics

## Methods

HTTP defines a small set of methods with conventional semantics. `GET`
retrieves a resource and should be safe and idempotent. `POST` submits
data and is neither safe nor idempotent. `PUT` replaces a resource and
`DELETE` removes it. Both are idempotent. `PATCH` applies a partial
update.

## Status Codes

Responses begin with a three-digit status. `2xx` codes indicate success,
with `200 OK` being the most common. `3xx` codes indicate redirection.
`4xx` codes indicate a client error, including `401 Unauthorized` and
`404 Not Found`. `5xx` codes indicate a server error and the request can
usually be retried.

## Headers and Caching

Headers carry metadata about a request or response. `Content-Type`
declares the body's media type, `Authorization` carries credentials, and
`Cache-Control` instructs caches how to store and revalidate the
response. `ETag` and `Last-Modified` let clients perform conditional
requests so unchanged responses can return `304 Not Modified` without a
body.
