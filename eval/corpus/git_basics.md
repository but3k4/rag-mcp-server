# Git Basics

## Commits

A commit captures a snapshot of your tracked files. Use `git commit` to create
one. The commit message should briefly describe the change in the imperative
mood. Use `git log` to view history.

## Branches

A branch is a movable pointer to a commit. Create one with `git checkout -b
<name>`. To merge two branches, switch to the target branch and run `git
merge <source>`. Conflicts must be resolved manually before the merge can
complete.

## Reverting

To undo a local commit on a feature branch, use `git reset`. To undo a
published commit without rewriting history, use `git revert`. Revert creates
a new commit that inverts the changes. Reset rewrites history and should not
be used on shared branches.
