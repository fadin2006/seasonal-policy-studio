# Seasonal Tree-Policy Lab: results of the run of 2026-11-22

Stylised single-node study case built on the scenario data of the generation-expansion manuscript (Muna, Malik and Syed, submitted to Smart Cities, 3 November 2026). The maintenance decision, block size, availability, penalty and the stressed environments are assumptions; nothing here is a result of the expansion model. All costs are expected unserved energy in GWh per year plus a charge for unfinished maintenance.

## 1. The case

- Eight seasonal-diurnal splits, three joint demand-wind-solar states per split, probabilities from the 156 reduced scenarios.
- Fleet: conventional 36,638 MW, wind 8,000 MW, solar 4,700 MW (manuscript tables). Availability of units not in maintenance 92% (assumed).
- Decision per split: 0, 1 or 2 maintenance blocks out; one block = 3,664 MW (assumed 10% of the conventional fleet). 10 block-splits are due per year; an unfinished block costs 1,003 GWh.
- State: split, backlog, rank of the last net load. 174 reachable states, 3 actions, 26 candidate tests for a tree.

| Environment | In library | Demand factor (spring, summer, fall, winter) | Wind factor | Bound (GWh) |
|---|---|---|---|---|
| Design year | yes | 1.00, 1.00, 1.00, 1.00 | 1.00, 1.00, 1.00, 1.00 | 312.2 |
| Extended heat | yes | 1.15, 1.07, 1.15, 1.00 | 1.00, 1.00, 1.00, 1.00 | 3,270.7 |
| Cold wind lull | yes | 1.00, 1.00, 1.02, 1.10 | 0.80, 0.80, 0.50, 0.30 | 546.0 |
| Mild heat (held out) | no | 1.10, 1.05, 1.10, 1.00 | 1.00, 1.00, 1.00, 1.00 | 2,155.7 |
| Mild wind lull (held out) | no | 1.00, 1.00, 1.01, 1.06 | 0.90, 0.90, 0.65, 0.45 | 469.2 |
| Heat and wind lull (unseen) | no | 1.12, 1.05, 1.12, 1.07 | 0.80, 0.80, 0.50, 0.40 | 2,531.6 |

The bound is the optimum of the unrestricted Markov decision process (dynamic programming); no tree can do better.

## 2. Tree policies found by search plus mixed-integer programming

| Tree | Depth | Rules | Cost (GWh) | Bound (GWh) | Above bound | Proven optimal | Time (s) | MIP size at last depth (variables / constraints / binary) |
|---|---|---|---|---|---|---|---|---|
| design | 2 | 3 | 312.2 | 312.2 | 0.0% | yes | 15 | 3,918 / 5,576 / 90 |
| heat | 2 | 4 | 3,270.7 | 3,270.7 | 0.0% | yes | 24 | 3,918 / 5,576 / 90 |
| lull | 2 | 4 | 546.0 | 546.0 | 0.0% | yes | 11 | 3,918 / 5,576 / 90 |
| robust | 2 | 4 | 1,444.6 | 1,376.3 | 5.0% | no (gap ≤ 4.7%) | 391 | 7,862 / 12,544 / 206 |

Method: smallest depth first; at each depth a local search with exact evaluation gives a starting tree, and the mixed-integer program (occupancy-measure formulation, only the tree choices binary) runs with that value as a cutoff. The search stops as soon as a tree meets the bound. The robust tree minimises the mean cost over the three library environments.

**design tree**

- split is summer night or earlier?
  - yes:
    - split is spring night or earlier?
      - yes:
        - 2 blocks out
      - no:
        - no maintenance
  - no:
    - 2 blocks out

**heat tree**

- slack ≤ 3 blocks?
  - yes:
    - backlog ≤ 4 blocks?
      - yes:
        - 2 blocks out
      - no:
        - 1 block out
  - no:
    - backlog ≤ 7 blocks?
      - yes:
        - no maintenance
      - no:
        - 2 blocks out

**lull tree**

- slack ≤ 3 blocks?
  - yes:
    - split is fall night or earlier?
      - yes:
        - 2 blocks out
      - no:
        - 1 block out
  - no:
    - backlog ≤ 7 blocks?
      - yes:
        - no maintenance
      - no:
        - 2 blocks out

**robust tree**

- split is summer night or earlier?
  - yes:
    - split is spring night or earlier?
      - yes:
        - 2 blocks out
      - no:
        - no maintenance
  - no:
    - split is fall day or earlier?
      - yes:
        - 1 block out
      - no:
        - 2 blocks out

### Exact cost of every tree in every environment (GWh per year)

| Policy | Design year | Extended heat | Cold wind lull | Mild heat (held out) | Mild wind lull (held out) | Heat and wind lull (unseen) |
|---|---|---|---|---|---|---|
| MIP tree: design | 312.2 | 3,739.9 | 736.5 | 2,155.7 | 469.2 | 2,780.7 |
| simulator-only tree: design | 312.2 | 3,739.9 | 736.5 | 2,155.7 | 469.2 | 2,780.7 |
| MIP tree: heat | 318.1 | 3,270.7 | 3,632.9 | 2,160.4 | 1,795.8 | 4,362.0 |
| simulator-only tree: heat | 318.1 | 3,270.7 | 3,632.9 | 2,160.4 | 1,795.8 | 4,362.0 |
| MIP tree: lull | 312.2 | 3,739.9 | 546.0 | 2,155.7 | 469.2 | 2,780.7 |
| simulator-only tree: lull | 312.2 | 3,739.9 | 546.0 | 2,155.7 | 469.2 | 2,780.7 |
| MIP tree: robust | 312.2 | 3,285.1 | 736.5 | 2,155.7 | 469.2 | 2,531.6 |
| simulator-only tree: robust | 312.2 | 3,285.1 | 736.5 | 2,155.7 | 469.2 | 2,531.6 |
| bound (dynamic programming) | 312.2 | 3,270.7 | 546.0 | 2,155.7 | 469.2 | 2,531.6 |

### Simulator-only search (framework 3)

| Tree | Sample-average cost ± 95% | Exact cost | Candidates | Simulated years | Time (s) |
|---|---|---|---|---|---|
| design | 302.2 ± 10.6 | 312.2 | 6,000 | 2,425,000 | 1 |
| heat | 3,179.7 ± 71.5 | 3,270.7 | 6,000 | 2,425,000 | 1 |
| lull | 534.6 ± 18.5 | 546.0 | 6,000 | 2,425,000 | 2 |
| robust | 1,440.7 ± 32.7 | 1,444.6 | 6,000 | 7,275,000 | 4 |

## 3. Runtime monitor

Depth-2 tree (CART) over features of the last 4 splits; training accuracy 91.3%. Features: mean demand gap, largest demand gap, mean wind gap, largest wind gap, share outside design box. A window outside the 0.2%-99.8% range of its predicted class is labelled unknown. Assumed measurement noise: demand 1.5% relative, wind capacity factor 0.03 absolute.

- share outside design box ≤ 0.125?
  - yes:
    - mean wind gap ≤ 0.003?
      - yes:
        - Cold wind lull (54% of training windows in this leaf)
      - no:
        - Design year (84% of training windows in this leaf)
  - no:
    - largest wind gap ≤ 0.092?
      - yes:
        - Extended heat (95% of training windows in this leaf)
      - no:
        - Cold wind lull (98% of training windows in this leaf)

Share of windows given each label on fresh simulations:

| True environment | Design year | Extended heat | Cold wind lull | Unknown |
|---|---|---|---|---|
| Design year | 96% | 1% | 2% | 1% |
| Extended heat | 3% | 95% | 0% | 1% |
| Cold wind lull | 15% | 3% | 80% | 1% |
| Mild heat (held out) | 6% | 91% | 1% | 2% |
| Mild wind lull (held out) | 18% | 4% | 69% | 9% |
| Heat and wind lull (unseen) | 0% | 17% | 48% | 34% |

## 4. Three frameworks in the loop

Three-year paths; year 1 design year; change at a uniformly random split of year 2; 8,000 paths per scenario; mean GWh per year ± 95% interval.

| Change to | Design-year tree, no monitor | F1 robust tree | F2 monitor + library | F3 simulator-only trees | After review: confident switching | Library with true label | Lower bound | Mean detection delay (splits) | Wrong label before change |
|---|---|---|---|---|---|---|---|---|---|
| Design year | 314 ± 5 | 314 ± 5 | 314 ± 5 | 314 ± 5 | 317 ± 5 | 314 ± 5 | 314 ± 5 | 0.0 | 3.8% |
| Extended heat | 1,968 ± 26 | 1,722 ± 25 | 1,739 ± 24 | 1,738 ± 24 | 1,718 ± 25 | 1,714 ± 25 | 1,714 ± 25 | 1.9 | 3.7% |
| Cold wind lull | 549 ± 7 | 549 ± 7 | 459 ± 7 | 473 ± 8 | 729 ± 9 | 432 ± 7 | 432 ± 7 | 2.7 | 3.7% |
| Mild heat (held out) | 1,195 ± 18 | 1,195 ± 18 | 1,200 ± 18 | 1,199 ± 18 | 1,202 ± 18 | 1,197 ± 18 | 1,195 ± 18 | 2.1 | 3.9% |
| Mild wind lull (held out) | 392 ± 6 | 392 ± 6 | 425 ± 6 | 428 ± 6 | 769 ± 8 | 392 ± 6 | 392 ± 6 | 3.0 | 3.8% |
| Heat and wind lull (unseen) | 1,504 ± 21 | 1,370 ± 20 | 1,833 ± 22 | 1,797 ± 21 | 1,951 ± 21 | 1,370 ± 20 | 1,370 ± 20 | 6.5 | 3.7% |

### Scores

| Criterion | Weight | F1 | F2 | F3 | Basis |
|---|---|---|---|---|---|
| Cost when the environment changes | 25 | 3.8 | 0.0 | 0.0 | Measured: share of the avoidable extra cost removed, between the design-year tree without a monitor (0) and the lower bound that knows the change in advance (5); mean of six scenarios. |
| Optimality certificate | 20 | 3.0 | 4.0 | 2.0 | F1: robust tree not proven, gap reported. F2: 3 of 3 library trees proven optimal; the monitor is tested, not proven. F3: confidence intervals only. |
| Readability | 15 | 5.0 | 3.0 | 2.0 | Measured: rules an operator must read (F1 4, F2 19, F3 26); 5 for at most 6, 4 for at most 12, 3 for at most 20. |
| Works without a model | 15 | 1.0 | 2.0 | 5.0 | Judgement: F1 and F2 need the transition model of each environment; F3 needs only a simulator. |
| Scaling with the state space | 15 | 3.5 | 3.5 | 5.0 | Measured: the plain mixed-integer program was solved to optimality in 2 of 3 sizes within 240 s; the simulator search took under 1 s at the largest size and ended 1% above the bound. |
| Reuse of the existing optimisation model | 10 | 5.0 | 4.0 | 2.0 | Judgement: F1 is one mixed-integer program; F2 adds one per environment and a monitor; F3 is new code. |
| **Weighted total (0-100)** | 100 | **69.3** | **49.5** | **48.0** | Highest: F1, one robust tree by mixed-integer programming, no monitor |

Mean extra cost over the lower bound (GWh per year): nominal 84, hybrid 212, F1 21, F2 92, F3 89.

Three ways to extend the model were built and scored on this case. F1 scores highest (69 of 100), ahead of F2 (50) and F3 (48). Two criteria are measured on the simulations, two on the solver runs, and two are judgements, marked as such. Weights are a choice; the table lets a reader change them.

### Improved after review

After the review the best-scoring framework was combined with the monitor: the robust tree runs by default and a specialised tree takes over only on a heat or wind-lull label from a leaf that is at least 90% pure. This does not lower the mean extra cost over the lower bound (212 against 21 GWh per year), so the page applies the best-scoring framework unchanged and uses the monitor as an alarm only.

## 5. When does selection pay off? Sensitivity to the maintenance requirement (exact)

| Block-splits due | Environment | Bound | Own tree | Robust tree | Design-year tree | Cost of not selecting |
|---|---|---|---|---|---|---|
| 8 | Design year | 312.2 | 312.2 | 312.2 | 312.2 | 0.0 |
| 8 | Extended heat | 3,266.5 | 3,266.5 | 3,266.5 | 3,739.9 | 0.0 |
| 8 | Cold wind lull | 546.0 | 546.0 | 546.0 | 736.5 | 0.0 |
| 10 | Design year | 312.2 | 312.2 | 312.2 | 312.2 | 0.0 |
| 10 | Extended heat | 3,270.7 | 3,270.7 | 3,285.1 | 3,739.9 | 14.4 |
| 10 | Cold wind lull | 546.0 | 546.0 | 736.5 | 736.5 | 190.6 |
| 12 | Design year | 318.1 | 318.1 | 662.5 | 318.1 | 344.4 |
| 12 | Extended heat | 3,744.2 | 3,744.2 | 4,054.3 | 3,744.2 | 310.1 |
| 12 | Cold wind lull | 1,147.7 | 1,147.7 | 2,842.8 | 3,632.9 | 1,695.1 |

Exact expected cost in GWh per year. 'Cost of not selecting' is the robust tree minus the tree built for that environment. With 8 block-splits the three differences sum to 0 GWh; with 10 block-splits the three differences sum to 205 GWh; with 12 block-splits the three differences sum to 2,350 GWh.

## 6. Scaling (benchmark view)

| Memory | Reachable states | MIP variables | MIP constraints | MIP time (s) | MIP gap | MIP tree cost | Bound | Local search (s) / cost | Simulator search (s) / cost |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 174 | 3,918 | 5,575 | 24 | 0.00% | 3,270.7 | 3,270.7 | 1 / 3,289.3 | 0 / 3,289.3 |
| 2 | 522 | 11,574 | 16,711 | 34 | 0.00% | 3,270.7 | 3,270.7 | 1 / 3,289.3 | 0 / 3,289.3 |
| 3 | 1,566 | 34,542 | 50,119 | 240 | 0.57% | 3,289.3 | 3,270.7 | 1 / 3,289.3 | 0 / 3,289.3 |

Heat objective, depth-2 tree, plain mixed-integer program without a starting solution, time limit 240 s. Longer memory multiplies the states by three each step; the tests of the tree stay the same. The local search with exact evaluation took 1, 1, 1 s and the dynamic-programming bound 0.01, 0.01, 0.07 s at the three sizes.

## 7. Limits

- Single node; the network, unit commitment and zonal limits of the expansion model are not represented.
- The environments, noise levels, block size, availability and penalty are assumptions chosen to make a test case; the maintenance requirement was set where the three environments need different schedules, and section 5 shows the result for other values.
- The monitor is a CART tree, not an optimal tree, and its thresholds were learned on simulated data from the same generators it is tested on (fresh seeds; two milder variants and one compound environment were held out).
- 'Proven optimal' refers to the tree class used here (complete trees up to depth 3 over the listed tests).
- The simulator-only search has no optimality proof; its exact cost is known here only because the model is available.
