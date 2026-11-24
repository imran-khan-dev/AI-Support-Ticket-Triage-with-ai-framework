# F4.5 — Agent Performance, Token, and Cost Comparison

## Preliminary Results

These measurements use deterministic scripted providers and controlled tool handlers. Sample sizes are small (3 repetitions per scenario), so results are **preliminary** and should not be used to declare a statistically significant performance winner. Small latency differences are not presented as meaningful production advantages.

Benchmark executed: 2026-11-24T03:39:31.756529+00:00

## Configuration

- Warm-up trials: 2
- Measured repetitions: 3
- Live provider enabled: False
- Total scenarios: 14
- Shared scenarios (both implementations): 13
- Excluded scenarios: 1

## Environment

- **python_version:** 3.14.7 (tags/v3.14.7:823f032, Aug  5 2026, 10:51:32) [MSC v.1944 64 bit (AMD64)]
- **os:** Windows-10-10.0.19045-SP0
- **processor:** Intel64 Family 6 Model 140 Stepping 1, GenuineIntel
- **warm_up_trials:** 2
- **measured_repetitions:** 3
- **live_enabled:** False
- **deterministic_only:** True
- **note:** Deterministic results use scripted providers and do not represent real LLM provider latency or token usage.

## Paired-Comparison Summary

- Shared scenarios: 13
- Successful paired trials: 33
- Failed paired trials: 6
- Custom success rate: 36/42
- Framework success rate: 33/39

### Excluded Scenarios

| Scenario | Supported Implementations | Reason |
|---|---|---|
| agent-cancellation | custom | unsupported_by_one_implementation |

## Results by Scenario

| Scenario | Shared | Implementation | Trials | Success | Median Total (ms) | Mean Total (ms) | Median Provider (ms) | Median Tool (ms) | Overhead (ms) |
|---|---|---|---|---|---|---|---|---|---|
| agent-simple-direct-response | yes | custom | 3 | 3/3 | 0.255499966442585 | 0.2597666655977567 | 0.003400025889277458 | 0.0 | 0.2562666777521372 |
| agent-simple-direct-response | yes | framework | 3 | 3/3 | 1.0136000346392393 | 1.0193333340187867 | 0.01109996810555458 | 0.0 | 1.008066659172376 |
| agent-simple-direct-response | yes | comparison | -- | -- | -- | -- | -- | -- | -- | rel_diff=-0.7479282185171564 |
| agent-single-tool-success | yes | custom | 3 | 3/3 | 1.600600080564618 | 1.5960000455379486 | 0.014500226825475693 | 1.153100049123168 | 0.4278332150230805 |
| agent-single-tool-success | yes | framework | 3 | 3/3 | 2.6844998355954885 | 2.7087999042123556 | 0.03150012344121933 | 1.0700998827815056 | 1.6075332338611286 |
| agent-single-tool-success | yes | comparison | -- | -- | -- | -- | -- | -- | -- | rel_diff=-0.4037622728296553 |
| agent-multiple-sequential-tools | yes | custom | 3 | 3/3 | 2.619700040668249 | 2.6200333765397468 | 0.02350006252527237 | 2.0051999017596245 | 0.5914999637752771 |
| agent-multiple-sequential-tools | yes | framework | 3 | 3/3 | 4.29240008816123 | 4.2967333768804865 | 0.07069995626807213 | 1.9183000549674034 | 2.283466669420401 |
| agent-multiple-sequential-tools | yes | comparison | -- | -- | -- | -- | -- | -- | -- | rel_diff=-0.38968875527386565 |
| agent-conditional-tool-routing | yes | custom | 3 | 3/3 | 1.544599886983633 | 1.5500333004941542 | 0.013499986380338669 | 1.113800099119544 | 0.41693332605063915 |
| agent-conditional-tool-routing | yes | framework | 3 | 3/3 | 2.7187999803572893 | 2.715533288816611 | 0.03179977647960186 | 1.0778999421745539 | 1.5955000805358093 |
| agent-conditional-tool-routing | yes | comparison | -- | -- | -- | -- | -- | -- | -- | rel_diff=-0.4318817499841785 |
| agent-tool-execution-failure | yes | custom | 3 | 3/3 | 1.2437000405043364 | 1.2505000146726768 | 0.013299984857439995 | 0.8175000548362732 | 0.413700162122647 |
| agent-tool-execution-failure | yes | framework | 3 | 3/3 | 2.4973000399768353 | 2.479633316397667 | 0.036800047382712364 | 0.8000999223440886 | 1.6387333162128925 |
| agent-tool-execution-failure | yes | comparison | -- | -- | -- | -- | -- | -- | -- | rel_diff=-0.5019821324649989 |
| agent-provider-failure | yes | custom | 3 | 0/3 | 0.22460008040070534 | 0.22493341627220312 | 0.001700129359960556 | 0.0 | 0.22326665930449963 |
| agent-provider-failure | yes | framework | 3 | 0/3 | 0.9519997984170914 | 0.979333184659481 | 0.003400025889277458 | 0.0 | 0.9760331983367602 |
| agent-provider-failure | yes | comparison | -- | -- | -- | -- | -- | -- | -- | rel_diff=-0.7640754958413307 |
| agent-tool-retry-behaviour | yes | custom | 3 | 3/3 | 1.5360999386757612 | 1.5343666697541873 | 0.013400102034211159 | 1.0899999178946018 | 0.42889996742208797 |
| agent-tool-retry-behaviour | yes | framework | 3 | 3/3 | 2.6581999845802784 | 2.6765000075101852 | 0.032800016924738884 | 1.0251998901367188 | 1.6071334636459749 |
| agent-tool-retry-behaviour | yes | comparison | -- | -- | -- | -- | -- | -- | -- | rel_diff=-0.422127775341814 |
| agent-maximum-tool-call-limit | yes | custom | 3 | 3/3 | 4.103199811652303 | 4.103133299698432 | 0.02640020102262497 | 3.6064002197235823 | 0.46869972720742226 |
| agent-maximum-tool-call-limit | yes | framework | 3 | 3/3 | 5.026200087741017 | 5.8460666332393885 | 0.05330005660653114 | 3.1992001459002495 | 1.7991665129860241 |
| agent-maximum-tool-call-limit | yes | comparison | -- | -- | -- | -- | -- | -- | -- | rel_diff=-0.18363778997575667 |
| agent-loop-prevention | yes | custom | 3 | 0/3 | 4.102699924260378 | 4.112000033880274 | 0.03200024366378784 | 3.4251997712999582 | 0.6458000279963017 |
| agent-loop-prevention | yes | framework | 3 | 0/3 | 4.387499997392297 | 4.388066629568736 | 0.05959998816251755 | 2.1683000959455967 | 2.1663999650627375 |
| agent-loop-prevention | yes | comparison | -- | -- | -- | -- | -- | -- | -- | rel_diff=-0.0649116976185047 |
| agent-missing-invalid-arguments | yes | custom | 3 | 3/3 | 2.1817998494952917 | 2.257999964058399 | 0.014200108125805855 | 1.7285998910665512 | 0.43006652655700844 |
| agent-missing-invalid-arguments | yes | framework | 3 | 3/3 | 3.346600104123354 | 3.3162334002554417 | 0.03510015085339546 | 1.7015000339597464 | 1.6003333342572053 |
| agent-missing-invalid-arguments | yes | comparison | -- | -- | -- | -- | -- | -- | -- | rel_diff=-0.3480548073828448 |
| agent-tenant-isolation | yes | custom | 3 | 3/3 | 1.516600139439106 | 1.5262334297100704 | 0.013799872249364853 | 1.093100057914853 | 0.41296682320535183 |
| agent-tenant-isolation | yes | framework | 3 | 3/3 | 2.727799816057086 | 2.7763332861165204 | 0.03220001235604286 | 1.0810999665409327 | 1.6413332584003608 |
| agent-tenant-isolation | yes | comparison | -- | -- | -- | -- | -- | -- | -- | rel_diff=-0.44402073403198467 |
| agent-approval-required | yes | custom | 3 | 3/3 | 2.1710998844355345 | 2.1954666202267012 | 0.014800112694501877 | 1.7245998606085777 | 0.43196650221943855 |
| agent-approval-required | yes | framework | 3 | 3/3 | 3.2818999607115984 | 3.693133359774947 | 0.03160000778734684 | 1.6537001356482506 | 1.87263327340285 |
| agent-approval-required | yes | comparison | -- | -- | -- | -- | -- | -- | -- | rel_diff=-0.3384625032980026 |
| agent-cancellation | no | custom | 3 | 3/3 | 0.2359999343752861 | 0.2360666791598002 | 0.0029001384973526 | 0.0 | 0.2330666563163201 |
| agent-final-answer-after-tool | yes | custom | 3 | 3/3 | 1.6314000822603703 | 1.641533337533474 | 0.014099758118391037 | 1.1917001102119684 | 0.426500104367733 |
| agent-final-answer-after-tool | yes | framework | 3 | 3/3 | 2.6895999908447266 | 2.820933315282067 | 0.031700124964118004 | 1.0577000211924314 | 1.7283666723718245 |
| agent-final-answer-after-tool | yes | comparison | -- | -- | -- | -- | -- | -- | -- | rel_diff=-0.3934413712769258 |

## Aggregate Comparison

| Metric | Custom | Framework |
|---|---|---|
| median_total_latency_ms | 1.5724000986665487 | 2.727799816057086 |
| mean_total_latency_ms | 1.793433346652559 | 3.0551256182102056 |
| median_provider_latency_ms | 0.01409999094903469 | 0.032599782571196556 |
| median_tool_latency_ms | 1.1384000536054373 | 1.077400054782629 |
| execution_overhead_ms | 0.4148190242371389 | 1.6557461259743342 |
| mean_estimated_cost_usd | None | None |
| aggregate_estimated_cost_usd | None | None |
| success_rate | 0.8571428571428571 | 0.8461538461538461 |

- Relative latency difference (custom - framework) / framework: -0.42356470243502564

## Notes

- Deterministic trials use scripted providers and controlled tool handlers. They measure adapter and orchestration overhead, not real LLM performance.
- Token usage is unavailable for deterministic providers and is reported as null.
- Cost estimates require actual token usage and configured model pricing; they are null when either is missing.
- Live Groq trials are opt-in (F45_LIVE_ENABLED) and are not mixed with deterministic results.

- Timing boundaries: total latency includes provider waiting, tool execution, retry delays, serialization, and adapter overhead. Provider latency is the time spent in provider.complete()/invoke(). Tool latency is the time spent in tool executor.execute(). Execution overhead = total - provider - tool.
- Timing boundary note: The Custom adapter counts limit-rejected tool calls in tool_latency_ms (they reach the executor and return ExecutionLimitError). The Framework adapter does not count limit-rejected calls in tool_latency_ms (they are blocked before reaching the executor). This difference is documented and affects execution_overhead_ms comparability between implementations.

- Token extraction from real Groq provider responses is not yet implemented in the comparison layer. All deterministic trials report null token usage.