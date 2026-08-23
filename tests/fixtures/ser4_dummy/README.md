# ser4_dummy fixture contract

The SER E2E tests generate this fixture in a temporary directory so that binary
shards and decoder checkpoints never become source fixtures. The immutable
parameters are recorded in `fixture_spec.json`.

The generated flow contains MSP-Podcast train/validation/test, HCUDB1
train/validation/test, and IEMOCAP external test samples for all four labels.
Features are small deterministic frame arrays; no restricted audio or encoder
dependency is required.
