# Public release checklist

## Identity and ownership

- [ ] Perform formal trademark and package-name clearance for “pyOptoTrace.”
- [ ] Reserve the repository name and package name only after choosing the final identity.
- [ ] Confirm the copyright owner and whether institutional review is required.
- [ ] Resolve every item in `THIRD_PARTY_NOTICES.md`.
- [ ] Add repository and documentation URLs to `pyproject.toml` and `CITATION.cff`.

## Privacy and data

- [ ] Run secret, large-file, email, absolute-path, and subject-identifier scans on the final tree.
- [ ] Review Git history as well as the working tree before making the remote public.
- [ ] Verify that examples are synthetic or have explicit redistribution permission.
- [ ] Add a small, versioned public test dataset only if its provenance is documented.

## Scientific validation

- [ ] Lock expected outputs for representative voltage and cAMP traces.
- [ ] Compare CPU/GPU results within documented tolerances.
- [ ] Test positive- and negative-going indicator sign conventions.
- [ ] Validate timing, dropped-frame, and acquisition-end behavior.
- [ ] Document supported acquisition hardware and metadata variants.

## Software quality

- [ ] Run tests on Python 3.10, 3.11, and 3.12.
- [ ] Test optional dependency groups on clean CPU and CUDA machines.
- [ ] Build and inspect both wheel and source distribution.
- [ ] Add API documentation and one end-to-end tutorial.
- [ ] Replace print-based progress and diagnostics with logging where appropriate.
- [ ] Tag `v0.1.0` only after the alpha warnings match actual support.

## Publication

- [ ] Enable private vulnerability reporting and branch protection.
- [ ] Add an issue template and pull-request template.
- [ ] Create a DOI/archive record if the software supports a paper.
- [ ] Publish to TestPyPI before PyPI.
- [ ] Verify the installation instructions from a clean environment.
