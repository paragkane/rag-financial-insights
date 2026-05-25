# shared/

Single source of truth for cross-language type contracts.

- `schema/*.schema.json` — JSON Schema (draft 2020-12)
- `codegen/gen_pydantic.py` — emits Python models into `backend/src/agents/types.py`
- `codegen/gen_typescript.ts` — emits TS types into `workers/src/lib/schema.ts` and `frontend/src/lib/api.ts`

Regenerate after any schema change so the Python / Workers / Frontend layers stay in lockstep.
