# cnn-adversarial-distance
Dissertation on the importance of distance metrics for adversarial attacks on TSC DNNs

## Running the code

### dissertation_main (runnable)
**The following lines can be (un)commented independently of each other to use their functionality or not.**
**Nested entries are not independent and one should not be modified without the other.**
- Re-generate dataframes (optional): 29-41
- All classes demonstration plot: 79-80
- Control parameters and distance measure utilised: 485-495. Metric name and method should coincide.
- Execute PGD: 499-500
  - Append results to PGD run log: 512, 514
  - Save results: 529. First parameter allows choice of PGD file name.
- Execute RPS: 501-502
  - Append results to RPS run log: 513, 515
  - Save results: 529. First parameter allows choice of RPS file name.
- Load pre-executed results: 503. First parameter allows choice of PGD or RPS results.
- Combing: 525-527
  - Append results to PGD run log: 531
  - Append results to RPS run log: 532
- Show generated example successful adversaries: 538

### disseration_metrics (auxiliary)
- Modify L<sub>0</sub> equality approximation: 6

### hardcode_plotting (runnable)
**This script was used to generate various plots from the report. The values were copied by hand from the run logs of the different algorithms.**

## Pre-generated results
**Pickles folder contains pre-generated results labeled accordingly. `extra` parameter in `dissertation_main` can be used to avoid overwriting files.
