# pstack model configuration for Claude Code

Use this project file for all pstack model settings, including reads, fallbacks, writes, and setup reruns. It replaces ~/.claude/rules/pstack-models.md and the older ~/.agents/pstack-models.md within this repository. Keep these preferences project-local.

Pass each entry's model and effort separately to Agent. Panel lists run one seat per entry.

# budget: large (xhigh)
feature, refactoring: opus xhigh
bug-fix: opus xhigh
perf-issue: opus xhigh
hillclimb: opus xhigh
judgment and prose: opus xhigh
hardest tasks: opus xhigh
how explorer: opus xhigh
how explainer: opus xhigh
why investigators: opus xhigh
why synthesizer: opus xhigh
reflect tooling: opus xhigh
reflect judgment, divergent, synthesizer: opus xhigh
arena runners: opus xhigh, sonnet xhigh
arena cross-judge pool: opus xhigh, sonnet xhigh
swarm workers: opus xhigh
architect runners: opus xhigh, sonnet xhigh
interrogate reviewers: opus xhigh, sonnet xhigh
