# Corrected Dynamic Delay Validation

## Model

CO:
- Dynamic candidate within validated support.
- Fixed 32-second fallback.
- Whole-heat fallback if time mapping is not monotonic.

CO2:
- Fixed 34-second baseline.

## Time mapping

analyzer_time = event_time + delay(event_time)

The mapping must be strictly increasing.

## Cumulative emissions

CO and CO2 cumulative emissions must be non-decreasing.

## Output

- corrected_alignment_summary.csv
- rolling_window_delay_estimates.csv
- heat_XX_corrected_aligned.csv
- figure_corrected_flow_delay.png
- figure_corrected_flow_delay.pdf
- experiment.json

Previous results are preserved.
