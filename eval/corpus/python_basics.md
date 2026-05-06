# Python Basics

## Lists

Python lists are ordered, mutable sequences. To add an item to the end of a
list, use the `append` method. To insert at a specific position, use `insert`.
To remove an item, use `remove` (by value) or `pop` (by index). Lists are
zero-indexed and can hold elements of mixed types.

## Dictionaries

A dictionary maps keys to values. Keys must be hashable. Use square brackets
to look up by key, or `get` for safe access with a default. To iterate
key-value pairs, use the `items` method. Dictionaries preserve insertion order
since Python 3.7.

## Comprehensions

List, dict, and set comprehensions provide a concise syntax for building
collections. Prefer them over equivalent for-loops when the body is simple.
Use generator expressions for large or lazy data to avoid loading everything
into memory.
