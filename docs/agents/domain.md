# Domain Docs

## Layout

This repository uses a **single-context** layout:

- Root `GLOSSARY.md`: shared domain vocabulary.
- `docs/adr/`: architecture decision records.

## Before exploring

Read `GLOSSARY.md` and ADRs relevant to the area being changed. When they are absent, proceed silently; create them lazily through domain modeling when terms or decisions are actually resolved.

## Vocabulary and decisions

Use glossary terms consistently in issues, designs, hypotheses, code, and tests. When a needed concept is missing, check existing usage before proposing a new term. Surface any conflict with an existing ADR, naming the ADR and explaining why it may need reconsideration.
