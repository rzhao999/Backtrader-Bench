# Accepted vs Discarded Question Analysis

This analysis covers all current summary artifacts in `db/summary`: the per-attempt JSONL summaries and their matching final summary JSON rollups.

## Scope

- Included per-attempt summary files: 10 JSONL files.
- Included final summary files: 10 JSON files.
- Unique mining runs represented by the per-attempt summaries: 10.
- Total attempts represented in the per-attempt summaries: 100.
- Total verified well-posed questions: 98.

The final summary JSON files duplicate run-level counts already present in their matching JSONL files, so aggregate question counts below are based on the per-attempt JSONL summaries to avoid double-counting.

## Aggregate Outcome

| Outcome | Count | Share of all attempts | Share of well-posed questions |
| --- | ---: | ---: | ---: |
| Accepted / tool-solver-correct | 38 | 38.0% | 38.8% |
| Discarded as easy | 60 | 60.0% | 61.2% |
| Parse failed | 0 | 0.0% | n/a |
| Verifier failed | 2 | 2.0% | n/a |
| Tool solver wrong | 0 | 0.0% | 0.0% |

Across the successful, well-posed candidates, the dominant filter is not whether the question is valid. Nearly all generated questions are valid. The main split is whether the no-tools model can already solve the question.

## Main Difference

Accepted questions are questions where:

- The base verifier succeeds.
- The perturbation checks pass, so the question is treated as well-defined.
- The no-tools solver is incorrect or fails to produce a usable answer.
- The tool-using solver answers correctly.

Discarded questions are questions where:

- The base verifier succeeds.
- The perturbation checks pass, so the question is also well-defined.
- The no-tools solver answers correctly.
- The tool-using solver is usually not run, because the question is already too easy for the no-tools baseline.

In short: accepted questions are not merely "valid"; they are valid questions that create a useful gap between no-tools and tool-enabled solving. Discarded questions are usually valid too, but they do not create that gap.

## Evidence From the Summaries

The accepted records consistently have `no_tools_correct: false` and then a successful tool-solver pass, recorded either as `decision: "accepted"` in the earlier run format or `decision: "tool_solver_correct"` in the later format.

The easy-discarded records consistently have `no_tools_correct: true`, with decisions named either `discarded_easy` in the earlier format or `easy_discarded` in the later format.

There were no `tool_solver_wrong` outcomes in the included final summaries. That suggests the current accepted set is gated very strongly by the tool solver: once a candidate survives the no-tools-easy filter, the tool solver has been able to solve every checked accepted candidate in these summaries.

## Run-Level Pattern

Most full 10-attempt runs save between 3 and 4 questions:

| Run | Accepted | Easy discarded | Other failures |
| --- | ---: | ---: | ---: |
| `mining_20260515_122637_rounds_10` | 4 | 6 | 0 |
| `mining_20260515_131330_rounds_10` | 3 | 6 | 1 verifier failed |
| `mining_20260515_133505_rounds_10` | 4 | 6 | 0 |
| `mining_20260515_142344_rounds_10` | 6 | 4 | 0 |
| `mining_20260515_145029_rounds_10` | 2 | 8 | 0 |
| `mining_20260515_151401_rounds_10` | 4 | 6 | 0 |
| `mining_20260515_153141_rounds_10` | 4 | 6 | 0 |
| `mining_20260515_160406_rounds_10` | 4 | 6 | 0 |
| `mining_20260515_163053_rounds_10` | 3 | 6 | 1 verifier failed |
| `mining_20260515_164951_rounds_10` | 4 | 6 | 0 |

The highest-yield run was `mining_20260515_142344_rounds_10`, with 6 accepted questions and 4 easy discards. The lowest-yield complete run was `mining_20260515_145029_rounds_10`, with 2 accepted questions and 8 easy discards.

## Domain Breakdown

I also classified the 98 included accepted/easy question records by the question text and light code cues. The buckets are exclusive and heuristic, so a question with both slippage and an analyzer metric is placed in the first matching bucket rather than counted twice.

| Domain bucket | Accepted | Easy discarded | Total | Acceptance rate |
| --- | ---: | ---: | ---: | ---: |
| Plain MA crossover to final equity | 20 | 23 | 43 | 46.5% |
| MA crossover plus timed or percent exit rules | 4 | 15 | 19 | 21.1% |
| MA/ATR regime with ATR-based exits | 2 | 3 | 5 | 40.0% |
| Analyzer or non-final-equity metric | 1 | 4 | 5 | 20.0% |
| Bracket or trailing-stop order mechanics | 9 | 6 | 15 | 60.0% |
| Final equity with slippage | 1 | 5 | 6 | 16.7% |
| Inline tabular OHLC/CSV feed | 1 | 2 | 3 | 33.3% |
| Breakout-style entry with filters | 0 | 2 | 2 | 0.0% |

### MCQ Type Descriptions

Plain MA crossover to final equity questions ask for the final broker value after a deterministic SMA/EMA crossover strategy. The core mechanics are indicator warm-up, crossover timing, fixed-size market orders, commissions, and final mark-to-market value. These are the most common generated questions; they are accepted fairly often, but many are discarded because the no-tools solver can sometimes infer the right option from a familiar template or from answer-choice structure.

MA crossover plus timed or percent exit rules questions add exits such as "sell after holding N bars", "sell after a fixed percent gain/loss", or "exit when price crosses a simple threshold." They look more complex than plain crossovers, but the added rule is usually explicit and local. In this sample they skew easy, suggesting that simple exit overlays do not reliably force tool use unless they interact with pending orders, incomplete fills, or subtle Backtrader state.

MA/ATR regime with ATR-based exits questions use moving-average entry logic but set stops or take-profits from ATR values. These require computing a dynamic volatility indicator and applying it to order/exit logic. They are moderately promising, but the sample is small; ATR helps when the question depends on the exact indicator series and entry price, but not when choices are easy to eliminate.

Analyzer or non-final-equity metric questions ask for outputs such as max drawdown, net profit, closed trade count, or another analyzer-style statistic rather than final portfolio value. These were often discarded as easy. The likely issue is that some metrics are near zero, integer-like, or otherwise guessable from the scenario and options unless the strategy creates enough trade path complexity.

Bracket or trailing-stop order mechanics questions involve Backtrader-specific order behavior such as parent/child bracket orders, stop/limit children, `StopTrail`, moving stop levels, cancellation behavior, or interactions between protective orders and crossover exits. This is the strongest category in the current data. These questions are more likely to be accepted because they require precise knowledge of Backtrader's event loop, order notifications, execution timing, and state transitions.

Final equity with slippage questions are final-value backtests where slippage is enabled or described as part of the broker setup. In this sample, slippage alone did not make questions difficult enough. It may be too easy when the slippage setting has a small numerical effect, when the number of trades is simple, or when answer choices make the adjusted value obvious.

Inline tabular OHLC/CSV feed questions paste a small OHLCV table directly into the prompt instead of describing a synthetic generator. These questions make the data visible to the solver. That can help human readability, but it may also make the problem easier for a no-tools model when the table is short or the strategy has few trades. The category is small and mixed.

Breakout-style entry with filters questions use rules like buying when close breaks above the previous N-bar high, often with indicator filters or exits. Both examples were discarded as easy, so the sample is too small to judge. As generated here, breakout conditions may have been easier to reason about than Backtrader order-mechanics questions.

The strongest accepted skew is in bracket and trailing-stop mechanics: these questions account for 15 rows and are accepted 60% of the time. Plain moving-average crossover questions are the largest bucket and perform better than the overall average, but they also generate the largest absolute number of easy discards because the template appears so often.

The weakest buckets in this sample are slippage-only final-equity questions, analyzer/non-final-equity metric questions, and crossover questions with simple timed or percent exits. These often sound complex, but the no-tools solver still answered many of them correctly. Breakout-style entries were all discarded, though there are only two examples.

The likely reason is that exotic Backtrader order mechanics create more hidden event-loop detail. Bracket and trailing-stop questions depend on parent/child order behavior, stop adjustment, fill timing, and interaction with crossover exits. By contrast, timed holds, percent exits, drawdown/trade-count metrics, and slippage phrasing often remain answerable from the provided choices or from simpler reasoning about the generated scenario.

## What Does Not Explain the Difference

- Well-posedness does not explain the accepted/discarded split. Both accepted and easy-discarded questions usually have `well_defined: true` and empty `well_defined_reasons`.
- Base verifier success does not explain the split. Both groups usually have `base_verifier_returncode: 0` and `base_verifier_timed_out: false`.
- Perturbation count does not strongly explain the split. Most accepted and discarded records use 3 perturbations, with occasional accepted records using more.
- Answer format does not cleanly explain the split. Both groups include dollar values; discarded questions also include some direct categorical/date/percentage answers, but the summaries do not include enough prompt text to conclude that answer type alone is causal.

## Practical Takeaways

The generator is producing mostly valid questions, but a majority are too easy for the no-tools baseline. Improving yield likely means making the generated questions require more execution, simulation, or detailed Backtrader behavior rather than direct inspection or simple arithmetic.

The most useful accepted questions appear to be those where the no-tools model either fails outright or gives a wrong option, while the tool solver can still recover the verified answer. Future mining prompts should bias toward bracket orders, trailing stops, parent/child order behavior, stateful order lifecycle details, and interactions between indicators and open orders. Simple timed holds, percent exits, slippage-only variants, and analyzer metrics may need additional hidden state or less guessable answer choices to avoid being discarded as easy.

The current verifier pipeline is conservative in a good way: invalid or unstable questions are rare, and no included final summary shows a tool-solver-wrong case. The main bottleneck is difficulty calibration, not verification quality.
