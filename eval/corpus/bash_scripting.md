# Bash Scripting

## Variables and Quoting

Bash variables are untyped strings. Assignment uses `name=value` with no
spaces around the equals sign, and expansion uses `$name` or `${name}`.
Always quote expansions to preserve whitespace and prevent word splitting,
and prefer `"${name}"` for unambiguous interpolation inside longer strings.

## Loops and Conditionals

`for` iterates over a word list, `while` loops on a command's exit status,
and `if` branches on commands as well. Use `[[ ... ]]` for tests rather than
the older `[ ... ]`. It handles empty variables and pattern matching more
safely. The `&&` and `||` operators chain commands by success or failure.

## Redirection and Pipes

`>` redirects stdout to a file (truncating), `>>` appends, and `2>&1`
merges stderr into stdout. A pipe `|` connects one command's stdout to the
next command's stdin. `set -euo pipefail` at the top of a script aborts on
error, undefined variable, or failure inside a pipeline.
