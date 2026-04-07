# CadQuery 2.7 Compatibility Notes

## Recommended API Patterns
- Use `import cadquery as cq`.
- Prefer robust operations: `circle()`, `rect()`, `extrude()`, `cutThruAll()`, `cutBlind()`, `hole()`, `fillet()`, `chamfer()`.
- Prefer selector strings on workplanes and solids:
  - `faces(">Z")`
  - `faces("<Z")`
  - `edges("|Z")`
  - `workplane()`

## Disallowed / Risky Patterns In This Project
- Do not use `cadquery.selectors.StringSelector`.
- Do not use `cq.selectors.StringSelector`.
- Do not use `arcTo` on `Workplane` in this environment.
- Avoid global `edges().chamfer(...)` without constrained selection.
- Avoid chaining cuts on potentially empty selectors.

## Safer Modeling Advice
- Build base solid first, then apply holes and fillets.
- Apply fillet/chamfer on narrowly selected edges only.
- When adding holes, prefer `workplane(...).hole(...)` or `cutThruAll()` with explicit sketch.
- Keep boolean order simple to reduce OCC failures (`BRep_API: command not done`).

## Output Contract
- Final shape must be assigned to variable `result`.
