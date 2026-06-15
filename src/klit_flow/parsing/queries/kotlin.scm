; ── Classes and interfaces ────────────────────────────────────────────────────
; Both are `class_declaration` in this grammar.
; Python code checks for an `interface` keyword child to distinguish them.
(class_declaration) @class

; ── Functions and methods (distinguished in Python via parent-node check) ─────
(function_declaration) @function

; ── Imports ───────────────────────────────────────────────────────────────────
(import_header) @import
