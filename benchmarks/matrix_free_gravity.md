# Matrix-free gravity benchmark

This bounded benchmark uses packaged Simple Example 02 rather than private
data. It exercises 70 free OD cells, 270 measurements, and two frozen-positive
cells. The logical operator has 18,900 entries (75,600 dense float32 bytes), but
the benchmark constructs no global measurement-by-OD matrix.

The recorded run compiled exactly one forward and one transpose product. Each
was executed once cold and once warm. It then compiled the complete adjoint
gravity objective, evaluated two parameter vectors, and completed a one-iteration
checkpointed estimation. The JSON report records preparation phase timings,
input shapes and dtype, backend/devices, RSS, objective compilation and execution
times, routing product counts, and checkpoint size.

Run it with:

```bash
uv run --frozen python benchmarks/benchmark_matrix_free_gravity.py
```

Wall-clock values are descriptive and are not CI thresholds. The regression
test checks structural properties, especially that no global operator was
constructed and that the expected bounded products were prepared.
