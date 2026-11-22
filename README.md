# Seasonal Policy Studio

Track a seasonal stochastic system, design a decision-tree policy, and see at once how far it is from the best possible policy.

**Live page:** https://fadin2006.github.io/seasonal-policy-studio/

A research prototype by M Fahrudin Muna (King Fahd University of Petroleum and Minerals). It extends a multi-stage stochastic generation-expansion model (Muna, Malik and Syed, *Reliable low-carbon power supply for cities: a deliverability-aware stochastic generation expansion approach*, submitted to Smart Cities, 3 November 2026) towards runtime monitoring and readable decision-tree policies.

## What it does
- **System and data.** Eight seasonal day and night splits with three joint demand, wind and solar states each. Upload hourly observations or a state table (CSV), change the fleet, or build a mock-up environment with the scenario generator.
- **Track.** A depth-2 monitor tree names the environment from the last four splits and flags conditions it has not seen.
- **Design and check.** Build a tree policy by hand. The studio reports its cost, the best possible cost (dynamic programming), the optimality gap, and the split where the loss arises.
- **Optimise.** Search for the best tree with a live curve of the best tree against the bound. With the back end running, a mixed-integer program proves optimality for a given depth.
- **Compare.** Three frameworks under a change of environment: one robust tree, a monitor with a policy library, and simulator-only search.

## What is data and what is assumed
Taken from the expansion model: the eight splits, the 24 joint states and their probabilities (156 reduced scenarios), and the fleet sizes. Assumed for this single-node study case: the maintenance decision, block size, availability, penalty for unfinished maintenance, measurement noise and the stressed environments. Nothing here is a result of the expansion model. All costs are expected unserved energy in GWh per year plus a charge for unfinished maintenance.

## Run it
- Online: the live page runs entirely in the browser.
- With the solver back end: `python server.py`, then open http://localhost:8765 (needs Python with numpy and scipy). This enables the "Prove with the solver" buttons.

## Repository
- `docs/` front end: `index.html`, `app.js` (interface), `engine.js` (model, bound, search, monitor, simulation, CSV import), `data.js` (built-in case and verified results).
- `server.py` back end: serves `docs/` and solves the tree mixed-integer program (SciPy HiGHS) at `/api/solve_tree`.
- `code/` the verified Python run (`lab_core.py`, `lab_sim.py`, `lab_run.py`), report and data builders, and tests of the browser engine and back end against the Python run.
- `REPORT.md` all numbers of the verified run of 22 November 2026; `results/results.json` the raw results.

Direct links to a view: add `#tab,environment,policy` to the address, for example `#design,heat,design`.

## Main findings of the verified run
- The trees for the design year, extended heat and cold wind lull meet the dynamic-programming bound and are proven optimal at depth 2 in 11 to 24 s. The robust tree is 5% above its bound and not proven in 391 s.
- Over 8,000 three-year paths per scenario, one robust tree costs 21 GWh per year more than a bound that knows the change of environment; the monitor with a library costs 92 and no adaptation 84. Switching on the monitor label helps in the wind lull and hurts under unseen conditions.
- Selecting a policy matters only when the maintenance schedule is tight.
- A plain mixed-integer program reaches a 240 s limit at 1,566 states; the search with exact evaluation takes about one second at every size.

## Limits
Single node; no network, unit commitment or zonal limits. The monitor is a heuristic tree trained on simulated windows. The in-browser search proves optimality only when it reaches the bound. Uploaded hourly data are grouped by k-means, a simpler method than the copula and scenario reduction of the manuscript.
