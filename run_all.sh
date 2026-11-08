#!/usr/bin/env bash
# Reproduce all results and figures of the paper, in dependency order.
# Prerequisites: data_raw/ holds the 20 Investing.com exports, data_external/VIX_History_CBOE.csv (see README).
set -euo pipefail
cd "$(dirname "$0")/code"
mkdir -p ../processed ../raw_audit ../figures
run() { echo "== $*"; python "$@" 2>&1 | tee "../raw_audit/$(basename "$1" .py).log"; }
run build_returns.py
run final_inference.py            # Sections 3.1-3.4: C1-C6 (surrogates, bootstrap, Gaussian reference)
run final_gc_gradient.py          # Section 3.1: first-order gradients
run final_regional_redundancy.py  # Section 3.3: regional systems
run final_regional_gauss_ref.py   # Section 3.4: regional Gaussian references
run final_state_gauss_ref.py      # Section 3.4: VIX-state recalibration (sensitivity)
run routeA_composition.py         # Section 3.2: composition / within-region regressions
run routeC_local_oinfo.py         # Section 3.4: local O-information series
run routeC_async_check.py         # Section 2.5: trading-time alignment variants
run final_aligned_gauss_ref.py    # Section 2.5: aligned-variant Gaussian references
run tcopula_reference.py          # Appendix C.5
run tcopula_reference_groups.py   # Appendix C.5, Table C1
run validate_entropy_estimator.py # Appendix C: estimator validation
run fig_global_scaling.py         # Figures 1-2
run fig_composition.py            # Figure 3
run fig_local.py                  # Figure 4
run fig_state.py                  # Figure 5
