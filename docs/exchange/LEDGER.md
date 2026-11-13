# Alpha–Beta relay ledger

Written only by `scripts/beta_relay.py`, one row per relay run that launched Codex (a `REFUSED` run launches nothing and writes no row). Rules: `docs/exchange/RELAY_PROTOCOL.md`.

| Thread | Round | Date | Request sha256 | Reply sha256 | Positions | Verdict | Other from_beta changes |
|---|---|---|---|---|---|---|---|
| X-0000 | closed | 2026-11-12 | — (hand-entered, pre-relay) | — (no file) | Q4 NO OBJECTION (given in session, not in a file) | CLOSED (hand-entered) | — |
| X-RELAY-CHECK | R1 | 2026-11-12 18:58 PST | `f828d7072721dbcbc10fa1d4bc416b056812a1c46f01e2749ae4b381da07bc5b` | `da6f97fb6884412d2fe97033c64f37ac885df90de4c77ca7cd741573eee69959` | Q1 OBJECTION | REVIEW RECEIVED | — |
| X-RELAY-CHECK | R2 | 2026-11-12 19:26 PST | `d9fa53f74fedaabc39756e73b39dc648dbf57f8130ceeae0110becae343eb1a2` | `50593a679d889944936ee51fe2e32d6bdd3b3af3b96bdba236ec473f98b3144f` | Q1 OBJECTION | REVIEW RECEIVED | — |
| X-RELAY-CHECK | R3 | 2026-11-12 19:33 PST | `505cf1c5b48d78033c13ed2fd1d34d974b6f36fe1a8b40ce1fac1c76a0545db4` | `90eb41b6258e443fa3c8f5c86ed275a950736ae46b42e04847909719bea63c98` | Q1 OBJECTION | REVIEW RECEIVED | — |
| X-0001 | R1 | 2026-11-12 20:19 PST | `8e2c585089793e17b8e79c90b10774cb39d2b10f2f0ab0bf7db7466b4cf6e4e4` | `9b722947d63aa7d2b118d1d88f70b600bb5ebf23f064fa21c2ee3cbb91652bef` | Q1 NO OBJECTION; Q2 NO OBJECTION; Q3 NO OBJECTION; Q4 NO OBJECTION; Q5 NO OBJECTION; Q6 OBJECTION (THEORETICAL); Q7 NO OBJECTION; Q8 SUGGESTION; Q9 NO OBJECTION; Q10 SUGGESTION | REVIEW RECEIVED | — |
| X-0001 | R2 | 2026-11-12 20:27 PST | `ae58a690f7b8411aca3b012d2acc9e5f066a4cfcd184d507a3ffae40d648dd4d` | `edc83cd68f41d8488d0c893a8f9336ed41d5fbaad2fb89ef386e52db4c37e1f3` | Q1 NO OBJECTION; Q2 NO OBJECTION; Q3 NO OBJECTION | REVIEW RECEIVED | — |
