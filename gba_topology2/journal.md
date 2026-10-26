# Development Journal: `gba_topology2`

This journal tracks atomic engineering units, algorithmic decisions, empirical findings, and validation milestones across the `gba_topology2` project.

---

## Standards Reference
All work follows the coding standards outlined in `/AGENTS.md`:
- Atomic units of work (minimal logical increments).
- Post-unit workflow gates:
  1. Unit tests in `pytest`.
  2. Formatting & linting via `ruff`, type checking via `mypy`.
  3. Docstrings and documentation entries in this journal.
- Package and execution orchestration via `uv`.
- Pre-commit enforcement hooks.

---

## Work Log

### [Entry 001] Setup Coding Standards & Tooling Infrastructure
- **Objective**: Establish development standards, directory layout, dev dependencies, and pre-commit automation for `gba_topology2`.
- **Changes**:
  - Authored root `/AGENTS.md` specifying:
    - Atomic units of work.
    - Required testing (`pytest`), formatting/linting (`ruff`), and type checking (`mypy`).
    - Documentation expectations (docstrings + `journal.md`).
    - Dedicated source structure under `gba_topology2/src/`.
    - Automated pre-commit hooks running before each commit.
  - Updated `pyproject.toml` with `[dependency-groups].dev` including `mypy`, `pre-commit`, `ruff`, and `pytest`.
  - Configured `.pre-commit-config.yaml` to enforce `ruff`, `ruff-format`, `mypy`, and `pytest` locally.
  - Installed git hooks via `pre-commit install`.
  - Created initial test structure `tests/test_sanity.py` and `gba_topology2/src/__init__.py`.
- **Validation**:
  - `uv run pre-commit run --all-files`: Passed.
  - `uv run ruff check gba_topology2 tests`: Passed.
  - `uv run ruff format --check gba_topology2 tests`: Passed.
  - `uv run mypy gba_topology2 tests`: Passed.
  - `uv run pytest`: 1 passed.

---

### [Entry 002] Step 1: Data Ingestion & Base Morse Detection (`mesh_geometry.py`, `morse_detector.py`)
- **Objective**: Implement discrete, modular subroutines for Step 1 of the `gba_topology2` roadmap:
  1. `mesh_geometry.py`: Calculate sphere radius $R$, average Euclidean edge length $\overline{\ell}_e$, and intrinsic angular mesh pitch $\delta\theta_{\text{mesh}} = 2\arcsin(\overline{\ell}_e / 2R)$.
  2. `morse_detector.py`: Pure Discrete Morse classification on spherical triangle meshes at $\tau = 0.0$ persistence. Classify vertices into local minima ($L^-(v) = \emptyset$), maxima ($L^+(v) = \emptyset$), and saddles ($L^-(v)$ has $k \ge 2$ connected components). Include Simulation of Simplicity (SoS) tie-breaking to guarantee integer Poincaré-Hopf theorem sum $N_{\max} + N_{\min} - N_{\text{sad}} = 2$.
- **Changes**:
  - Implemented `gba_topology2/src/mesh_geometry.py` with standalone functions: `compute_sphere_radius`, `extract_unique_edges`, `compute_average_edge_length`, `compute_mesh_pitch`, and `analyze_mesh_geometry`.
  - Implemented `gba_topology2/src/morse_detector.py` with `CriticalPoint`, `MorseDetectionResult`, and `compute_discrete_morse_cps`.
  - Added unit test suites `tests/test_mesh_geometry.py` and `tests/test_morse_detector.py` validating octahedron geometry, height function extrema, quadrupole saddle configurations, and dimension validation.
- **Empirical Findings on `Pd_20K.plt`**:
  - $N_{\text{vertices}} = 20,174$, $N_{\text{triangles}} = 40,344$, $\overline{\ell}_e \approx 0.0384$ Å, $\delta\theta_{\text{mesh}} \approx 1.61^\circ$.
  - Without SoS tie-breaking, float discretization plateaus on Electron Density yielded $\chi = 3$. With SoS relative perturbation ($10^{-15}$), $\chi = 2$ is verified across all condensed fields (`Electron Density`, `V`, `ρ mean curvature`, etc.).

---

### [Entry 003] Step 2: Micro-Cluster Identification & Interactive 3D Picking (`micro_cluster.py`, `sphere_viewer.py`)
- **Objective**: Implement modular micro-clustering scaled by $\delta\theta_{\text{mesh}}$ and create the Step 2 interactive Trame 3D viewer with direct ray-pick inspection on `Pd_20K.plt`.
- **Changes**:
  - Implemented `gba_topology2/src/micro_cluster.py` with `MicroCluster`, `BoundaryRingPoint`, `generate_boundary_ring`, and `compute_micro_clusters`.
    - Groups critical points within $\theta_{ij} \le k \cdot \delta\theta_{\text{mesh}}$.
    - Calculates projected spherical centroid, angular/spatial diameter, boundary ring vertices, and local Euler index $\chi_{\text{micro}} = N_{\max} + N_{\min} - \sum \text{mult}_{\text{sad}}$.
  - Implemented `tests/test_micro_cluster.py` testing isolated CPs, cluster grouping, boundary ring coordinates, and parameter validation.
  - Implemented `gba_topology2/src/sphere_viewer.py`:
    - Renders atom sphere with scalar flood, level-set isocontours, and mesh wireframe.
    - Renders Discrete Morse critical points (Red = Maxima, Blue = Minima, Green = Saddles).
    - Renders black boundary rings around multi-CP micro-clusters.
    - Interactive 3D Picking: Ray-picks clusters and critical points directly in 3D canvas and via sidebar table to highlight in 3D and display full inspector cards.
- **Validation**:
  - Pre-commit hook suite (`ruff check`, `ruff format`, `mypy`, `pytest`): 10 passed, 0 errors.
  - Successfully ran `sphere_viewer.py` on `Pd_20K.plt` verifying 47 micro-clusters (39 multi-CP) and live HTTP rendering.

---

### [Entry 004] Saddle Multiplicity Transparency & Index Accounting (`micro_cluster.py`, `sphere_viewer.py`)
- **Objective**: Clarify topological index accounting for clusters containing higher-order / multi-pass saddles (specifically resolving empirical observation on `Pd_20K.plt` Cluster MC7 on Shape Index where 4 Maxima + 1 Saddle gave net index $+1$).
- **Background & Root Cause**:
  - On a 2D triangulated mesh, a saddle vertex with $n_{\text{lower}}$ disconnected lower link components carries topological index $-(n_{\text{lower}} - 1) = -\mu$.
  - In Cluster MC7, the single saddle vertex has $n_{\text{lower}} = 4$ descending components, yielding multiplicity $\mu = 3$ (index $-3$).
  - Naive display of `"4 Max, 1 Sad"` led users to expect $4 - 1 = +3$, whereas discrete Morse theory rigorously dictates $4 + 0 - 3 = +1$.
- **Changes**:
  - `micro_cluster.py`: Added `total_saddle_index` to `MicroCluster`. Updated `composition_summary` to annotate multi-index saddles when `total_saddle_index > n_saddles` (e.g. `"4 Max, 1 Sad (mult 3)"`).
  - `sphere_viewer.py`: Exposed `total_saddle_index`, individual CP `multiplicity`, and `index_contrib` in the UI state. Added an explicit arithmetic index breakdown to the cluster inspector card (`+N_max + N_min - total_saddle_index = net_index`) and annotated multi-component saddles in the constituent critical point list.
  - `tests/test_micro_cluster.py`: Added `test_higher_order_saddle_cluster_accounting` verifying cluster serialization, local Euler index evaluation, and composition summary formatting with higher-order saddles.
- **Validation**:
  - Unit tests: 11 passed in `pytest`.
  - Format, lint, & typecheck: `ruff format`, `ruff check`, and `mypy` passing cleanly.

---

### [Entry 005] Step 3 Evaluation: Adjacent Extrema Scanning & Architecture Hook (`field_scanner.py`)
- **Objective**: Implement a dedicated diagnostic tool to scan all condensed scalar fields on `Pd_20K.plt` for adjacent same-type critical points ($d_G = 1$) to evaluate the necessity of Step 3 extrema fusion.
- **Implementation**:
  - Authored `gba_topology2/src/field_scanner.py` using 1-skeleton graph edge adjacency and Discrete Morse classification.
  - Ran scan across all 13 condensed scalar fields on Atom #1 of `Pd_20K.plt` with unperturbed floating-point data ($\epsilon = 0.0$) and Simulation of Simplicity ($\epsilon = 10^{-15}$).
- **Empirical Findings**:
  - With raw unperturbed data ($\epsilon = 0.0$), flat zero-curvature or valley plateaus produce rare adjacent same-type extrema (e.g. Min-Min at vertices `659` and `663` in MC29 on `Electron Density` and MC35 on `V`).
  - With default Simulation of Simplicity ($\epsilon = 10^{-15}$), infinitesimal tie-breaking resolves plateaus deterministically without splitting extrema, producing exactly 0 adjacent same-type extrema ($d_G = 1$) and preserving $\chi = 2$.
  - The resulting critical point is consistent within mesh edge spacing ($\sim 0.038$ Å) with a fused centroid, and downstream Step 8 (Harmonic Nudge) optimizes the physical location.
- **Architectural Decision**:
  - Documented extension hook in `sphere_viewer.py` right before micro-clustering.
  - Documented architectural role in `field_scanner.py` and `plan.md`.
  - Bypassed active Step 3 runtime fusion in favor of proceeding directly to Step 4.

---

### [Entry 006] Unit 1: Tangent PCA, Cluster Morphology & Elliptical Boundaries (`micro_cluster.py`)
- **Objective**: Implement local geometric shape characterization (PCA on sphere tangent plane) in `micro_cluster.py` to classify multi-CP clusters into compact (Type 1) vs elongated string (Type 2) morphologies and generate oriented elliptical boundary rings.
- **Changes**:
  - `micro_cluster.py`:
    - Implemented `compute_tangent_pca`: projects member unit vectors onto the tangent plane at the centroid, diagonalizes the 2D covariance tensor, and extracts aspect ratio $\alpha = \sqrt{\lambda_1 / \lambda_2}$ and principal 3D tangent vector.
    - Implemented `generate_elliptical_boundary_ring`: generates a closed oriented elliptical boundary ring parameterized by semi-major radius $a$ and semi-minor radius $b$ along the principal tangent axes.
    - Updated `MicroCluster` dataclass with `morphology` (`"compact"` vs `"string"`), `aspect_ratio`, `principal_axis_unit`, `semi_major_rad`, and `semi_minor_rad`.
    - Updated `compute_micro_clusters` to automatically fit elliptical rings when $\alpha \ge 2.0$ for multi-CP clusters.
  - `tests/test_micro_cluster.py`:
    - Added `test_string_morphology_and_elliptical_ring` and `test_compact_morphology`.
    - Added validation for `aspect_ratio_threshold >= 1.0`.
- **Validation**:
  - Unit tests: 7 passed in `pytest`.
  - Format, lint, & typecheck: `ruff format`, `ruff check`, and `mypy` passing cleanly.

---

### [Entry 007] Unit 2: Macro-String Aggregation (`string_aggregator.py`)
- **Objective**: Implement collinear string aggregation (`string_aggregator.py`) to chain intrinsically elongated Type 2 clusters and collinear neighboring clusters along low-field valleys into unified `MacroString` entities with oriented elliptical excision boundaries.
- **Changes**:
  - `string_aggregator.py`:
    - Implemented `MacroString` container capturing constituent clusters, merged critical point members, tangent PCA, aspect ratio, length, width, and oriented boundary ring.
    - Implemented `aggregate_cluster_strings`: bridges string-like micro-clusters within $k_{\text{bridge}} \cdot \delta\theta_{\text{mesh}}$, computes global tangent PCA across all points in the string, and fits an oriented elliptical boundary ring $(a, b, \vec{t}_1)$.
  - `tests/test_string_aggregator.py`:
    - Implemented unit tests validating string aggregation, conservation of constituent critical points, local Euler characteristic, spherical constraint on elliptical ring points, and exclusion of isotropic compact clusters.
- **Validation**:
  - Full test suite: 16 passed across all tests.
  - Format, lint, & typecheck: `ruff format`, `ruff check`, and `mypy` passing cleanly.

---

### [Entry 008] Phase 1: Polarity Detection & Effective Extrema (`polarity_detector.py`)
- **Objective**: Implement outward radial slope analysis on the sphere to unambiguously classify composite micro-clusters and macro-strings with positive Euler index ($\chi_{\text{local}} = +1$) into either Mountain/Peak ($E\text{-MAX}$) or Basin/Pit ($E\text{-MIN}$).
- **Changes**:
  - `polarity_detector.py`:
    - Implemented `evaluate_spherical_radial_slope`: samples mesh vertices in an annular ring $[r_{\text{inner}}, r_{\text{outer}}]$ around the cluster centroid, computes the directional radial derivative $\partial f / \partial \theta$ via linear regression, and evaluates directional concordance confidence.
    - Implemented `classify_cluster_polarity`: probes terrain around a `MicroCluster`, assigns $E\text{-MAX}$ (if $\partial f / \partial \theta < 0$, field descends outward) or $E\text{-MIN}$ (if $\partial f / \partial \theta > 0$, field ascends outward), extracts the representative peak/pit value, and wraps into an `EffectiveExtremum` dataclass.
  - `tests/test_polarity_detector.py`:
    - Implemented unit tests for analytical spherical Gaussian peak ($df/d\theta < 0$), analytical basin ($df/d\theta > 0$), and composite micro-cluster classification (both peak and basin cases with concordance $\ge 0.7$).
- **Validation**:
  - Full test suite: 20 passed in `pytest`.
  - Format, lint, & typecheck: `ruff format`, `ruff check`, and `mypy` passing cleanly.

---

### [Entry 009] Phase 2: Catastrophe Classification & Sensitivity Metrics (`catastrophe_classifier.py`)
- **Objective**: Implement 3-fold (monkey saddle, $\chi = -2$) and 4-fold (octupolar cross, $\chi = -3$) catastrophe constellation identification in the meso-scale angular window, computing scalar barrier depth and angular span metrics to distinguish numerical noise (spurious unfolding) from genuine physical symmetry breaking.
- **Changes**:
  - `catastrophe_classifier.py`:
    - Implemented `evaluate_constellation_metrics`: computes pairwise angular span, barrier depth $\Delta f_{\text{internal}}$, relative barrier depth $\eta = \Delta f / \text{span}(f)$, and cyclic angular standard deviation across flanking saddles projected onto the tangent plane.
    - Implemented `classify_catastrophes`: searches around positive-index core clusters for candidate flanking saddles satisfying the topological index relation $\chi = 1 - k$ ($k \in \{4, 3\}$), prioritizes higher-order 4-fold constellations before 3-fold, computes a bifurcation confidence score $\mathcal{B} \in [0.0, 1.0]$, classifies constellations as `"spurious_unfolding"` (fused $E\text{-MSAD}$ / $E\text{-4SAD}$) vs `"physical_symmetry_breaking"` based on user-tunable `barrier_tol` and `angular_tol_mult`, and fits an enclosing boundary ring on the sphere.
  - `tests/test_catastrophe_classifier.py`:
    - Implemented unit tests for 3-fold spurious unfolding with negligible barrier (fused $E\text{-MSAD}$), 3-fold physical symmetry breaking with finite barrier ($\mathcal{B} > 0.8$, preserved separate saddles), 4-fold octupolar detection (5 clusters merging into $E\text{-4SAD}$, $\chi = -3$), and input validation.
- **Validation**:
  - Full test suite: 24 passed in `pytest`.
  - Format, lint, & typecheck: `ruff format`, `ruff check`, and `mypy` passing cleanly.

---

### [Entry 010] Phase 3: Excision Boundary Profiling & Port Extraction (`excision_boundary.py`)
- **Objective**: Implement closed 1D excision boundary loop profiling and discrete port extraction to detect alternating valley and ridge ports, independently verifying the topological index formula $\chi = 1 - k$ directly on the boundary.
- **Changes**:
  - `excision_boundary.py`:
    - Implemented `interpolate_scalar_at_points`: samples field values along arbitrary 3D boundary points using Inverse Distance Weighting (IDW) from sphere mesh vertices.
    - Implemented `extract_boundary_ports`: detects strict local 1D extrema on the closed periodic ring $f(\phi)$, filters peaks/valleys by topological prominence, and extracts `BoundaryPort` entities with type (`"valley"` vs `"ridge"`), azimuthal angles ($\phi$), 3D positions, and scalar values.
    - Implemented `profile_boundary_loop`: profiles boundary loops end-to-end, evaluates topological balance (equal valley and ridge counts), and computes the implied Euler index $\chi = 1 - k$.
  - `tests/test_excision_boundary.py`:
    - Implemented unit tests for 2-fold simple saddle fields (2 valleys, 2 ridges $\implies \chi = -1$), 3-fold monkey saddle fields (3 valleys, 3 ridges $\implies \chi = -2$), 4-fold octupolar cross fields (4 valleys, 4 ridges $\implies \chi = -3$), end-to-end mesh IDW interpolation, exact vertex interpolation, and empty loop validation.
- **Validation**:
  - Full test suite: 30 passed in `pytest`.
  - Format, lint, & typecheck: `ruff format`, `ruff check`, and `mypy` passing cleanly.

---

### [Entry 011] Phase 4: Harmonic Nudge Energy Minimization (`harmonic_nudge.py`)
- **Objective**: Implement symmetry energy functionals and gradient-free pattern optimization on the sphere to refine and center effective critical points and launch ports tailored to their fold order $k \in \{0, 2, 3, 4\}$.
- **Changes**:
  - `harmonic_nudge.py`:
    - Implemented `compute_k_fold_symmetry_energy`: evaluates circular variance for extrema ($k=0$), and Fourier spectral power penalty of non-$k$ harmonics for saddles ($k=2, 3, 4$).
    - Implemented `harmonic_nudge_critical_point`: performs spherical pattern search on the tangent plane with adaptive step halving and maximum displacement clamping scaled by $\delta\theta_{\text{mesh}}$.
    - Implemented `perfect_boundary_ports`: optimizes azimuthal angles of boundary ports toward ideal $360^\circ / k$ symmetry spacing using circular phase offset minimization.
  - `tests/test_harmonic_nudge.py`:
    - Implemented unit tests for circular variance extremum energy, 2-fold quadrupole saddle Fourier penalty, pattern search convergence on a perturbed extremum, and 3-fold ($120^\circ$) boundary port angle perfection.
- **Validation**:
  - Full test suite: 34 passed in `pytest`.
  - Format, lint, & typecheck: `ruff format`, `ruff check`, and `mypy` passing cleanly.

---

### [Entry 012] Phase 5: Interactive 3D Catastrophe & Boundary Port Visualization (`sphere_viewer.py`)
- **Objective**: Integrate catastrophe classification and boundary port extraction into the Trame 3D viewer (`sphere_viewer.py`), enabling interactive inspection of monkey saddles ($k=3$, $\chi = -2$), 4-fold cross saddles ($k=4$, $\chi = -3$), and their alternating valley/ridge ports directly on `Pd_20K.plt`.
- **Changes**:
  - `sphere_viewer.py`:
    - Added dedicated VTK pipelines for:
      - Catastrophe boundary rings (purple tubes on the sphere surface).
      - Boundary ports: Valley Ports (vivid orange spheres) and Ridge Ports (cyan spheres).
    - Added Step 5 interactive sidebar card:
      - Toggle switches for Catastrophe Rings and Boundary Ports.
      - User sliders for relative barrier depth tolerance (`cat_barrier_tol`, default $0.5\%$) and angular span tolerance (`cat_angular_tol_mult`, default $4.0\,\delta\theta_{\text{mesh}}$).
      - Catastrophes summary table displaying ID, Type (`E-MSAD`/`E-4SAD`), net index $\chi$, and classification (`Fused` vs `Split`).
    - Added Catastrophe Inspector card displaying:
      - Net index, fold order $k$, exact port breakdown ($k\text{V} + k\text{R}$).
      - Bifurcation confidence score ($\mathcal{B}$).
      - Scalar barrier depth, relative barrier percentage, and angular span in degrees.
      - Constituent micro-cluster IDs.
- **Empirical Validation on `Pd_20K.plt` (`V (condensed)`)**:
  - Identified all 8 expected monkey saddle constellations:
    - **`CAT_1`** through **`CAT_6`**: Octahedral/cubic symmetry group (including **`MC29`** + flanking saddles `MC44`, `MC53`, `MC56`, and **`MC33`** + flanking saddles `MC46`, `MC50`, `MC54`). Each possesses 3 Valley ports and 3 Ridge ports ($3\text{V} + 3\text{R}$) with net $\chi = -2$.
    - **`CAT_7`** & **`CAT_8`**: Symmetry-equivalent pole configurations (including **`MC41`** + `MC61`, `MC63`, `MC65`). Each resolves with $3\text{V} + 3\text{R}$, net $\chi = -2$, and is recognized as a spurious numerical unfolding ($\Delta f / \text{span}(f) < 0.5\%$) that fuses cleanly into an effective monkey saddle.
- **Validation**:
  - Full test suite: 34 passed in `pytest`.
  - Format, lint, & typecheck: `ruff format`, `ruff check`, and `mypy` passing cleanly.

---

### [Entry 013] Step 6 Assessment Toggles: Effective Critical Points & Harmonic Nudge (`sphere_viewer.py`)
- **Objective**: Expose the multi-CP collapse and harmonic nudge stages as **independent** interactive switches in the viewer, so each algorithmic stage can be judged visually before we commit to a fully automated pipeline.
- **Design Decisions**:
  - **Four independent switches** (not a master switch) so stages can be isolated:
    1. `reduce_catastrophes` — collapse a catastrophe constellation (e.g. 1 core + 3 flanking saddles) into a single effective saddle `E-SAD`.
    2. `reduce_multi_clusters` — collapse a standalone multi-CP micro-cluster into a single `E-MAX`/`E-MIN` via `classify_cluster_polarity`.
    3. `nudge_cp_positions` — run `harmonic_nudge_critical_point` symmetry-energy pattern search on effective CPs.
    4. `perfect_port_angles` — run `perfect_boundary_ports` to enforce ideal $360^\circ / k$ port spacing.
  - **Standard glyphs, boundary retained**: effective CPs reuse the normal red/blue/green glyphs. The black cluster ring and purple catastrophe ring stay visible after collapse, so a lone glyph inside a ring reads unambiguously as "this set was collapsed".
  - **Nudge is opt-in and lazy**: the pattern search only executes when `nudge_cp_positions` is on, keeping the default interaction responsive.
- **Changes**:
  - `harmonic_nudge.py`:
    - Added `reposition_ports_on_ring`: `perfect_boundary_ports` only rewrites the azimuthal angle `phi_rad`, so ports were visually unchanged in 3D. This subroutine recovers the ring frame $(\hat{c}, \hat{t}_1, \hat{t}_2, a)$ from the boundary samples and projects perfected angles back onto the excision loop, re-interpolating scalars via IDW.
    - Added `_ring_frame` helper and a `probe_radius_pitch` argument to `harmonic_nudge_critical_point` so constellations spanning several pitches probe outside their own footprint (per the excision-radius rule in `micro_cluster`).
  - `sphere_viewer.py`:
    - Deferred CP glyph population to the end of `update_topology()` so collapsed members can be hidden while rings persist.
    - Added `raw_cp_to_ui` / `effective_cp_to_ui` builders producing a uniform CP list for the table, picking, and inspector.
    - Added `state.display_counts` with a **multiplicity-weighted** index sum, so the displayed $\chi$ is invariant under reduction (a multiplicity-2 saddle glyph carries index $-2$; an effective saddle carries the net index of its set).
    - Added Step 6 sidebar card: 4 switches, max-nudge slider, reduction/nudge/perfection stats, and an effective-CP table.
    - CP inspector now shows origin set, net index, pre-nudge center, displacement, iteration count, symmetry-energy drop, and the constituent CP list.
    - 3D picking resolves effective glyphs (world-space and screen-space fallback) and skips raw CPs hidden by reduction.
  - `tests/test_harmonic_nudge.py`: 4 new tests for `reposition_ports_on_ring` (ring round-trip, equispaced 3D placement, IDW re-interpolation, edge cases).
  - `gba_topology2/verify_step6_toggles.py`: headless harness exercising all 8 toggle permutations on real data.
- **Empirical Validation on `Pd_20K.plt` (`V (condensed)`, 20174 nodes, $\delta\theta = 1.606^\circ$)**:
  - Raw Morse: 70 Max / 136 Min / 200 Sad, $\chi = 2$. 66 clusters (34 multi-CP), 8 catastrophes.
  - `reduce_catastrophes`: 48 raw CPs $\to$ 8 effective saddles; displayed $\chi$ stays exactly $2$.
  - `reduce_multi_clusters`: 352 raw CPs $\to$ 28 effective extrema; $\chi$ stays $2$.
  - Both: 400 CPs $\to$ 36 effective CPs (22 Max / 12 Min / 8 Sad), $\chi = 2$.
  - `nudge_cp_positions`: catastrophe centroids move mean $0.843^\circ$, max $1.204^\circ$ ($\approx 0.5$–$0.75\,\delta\theta_{\text{mesh}}$); symmetry energy never increases (verified descent across all 36 effective CPs).
  - `perfect_port_angles`: all 48 ports re-projected onto their excision loops (verified to lie on the ring's small circle to $<10^{-3}$ rad).
- **Validation**:
  - Full test suite: 38 passed in `pytest`.
  - Format, lint, & typecheck: `ruff format`, `ruff check`, and `mypy` passing cleanly.
- **Open Questions / Next Steps**:
  - The nudge displacement of $\approx 0.5\,\delta\theta_{\text{mesh}}$ for catastrophes is meaningful and should be measured against the ideal $k$-fold port phase once ports are perfected jointly with positions.
  - Effective saddles currently inherit the constellation centroid as the pre-nudge seed; a barrier-weighted centroid may be a better seed than the unweighted one.
  - Ready to feed the perfected ports into Step 6 basin tracing (`basin_tracer.py`).

---

### [Entry 014] Chi-Gated Reduction: Annihilating Topologically Neutral Sets (`polarity_detector.py`, `sphere_viewer.py`)

**Date**: 2026-10-07

- **Problem (user-reported)**: With `reduce_multi_clusters` on, cluster MC10
  (2 Max, 8 Sad, 6 Min; net $\chi_{\text{local}} = 0$) was rendered as a red
  *maximum* glyph while honestly reporting net index $+0$. A glyph carrying zero
  index is a category error: extrema carry $+1$, saddles carry $\le -1$.
- **Root cause**: `classify_cluster_polarity` is a *binary* classifier by design
  (its contract covers $\chi = +1$ sets: radial slope $< 0 \Rightarrow$ E-MAX,
  $> 0 \Rightarrow$ E-MIN). The viewer's Step 5 reduction loop called it for
  **every** standalone multi-CP cluster with no index gate, so all 16 $\chi = 0$
  ridge strings (MC1-MC16, the 2 Max/8 Sad/6 Min periodic stripe family,
  AR $\approx 5.7$) were forced into an extremum glyph.
- **Topological contract adopted** (`effective_reduction_decision`):
  - $\chi = 0$  -> **annihilate**: hide all member glyphs, emit **no** effective
    CP, keep the boundary ring (ghost-grey annulment style) as the visual record.
  - $\chi = +1$ -> E-MAX / E-MIN via the polarity classifier (unchanged).
  - $\chi \le -1$ -> E-SAD carrying the net index, fold order $k = 1 - \chi$.
  - $\chi \ge +2$ -> **irreducible**: one effective CP carries at most $+1$, so
    the set stays expanded (counted, never silently collapsed).
- **Changes**:
  - `polarity_detector.py`: new pure `effective_reduction_decision(chi, polarity)`
    + `ReductionDecision` literal; exported.
  - `sphere_viewer.py`: gate the cluster reduction loop on the decision; new
    ghost-grey `annulled_rings_actor` (tube radius 0.014, opacity 0.85, offset
    1.014R) re-emitting annulled cluster boundaries; `is_annulled` /
    `is_irreducible` flags on cluster UI rows with table chips; Step 6 card stat
    lines; `n_clusters_annihilated` / `n_clusters_irreducible` in
    `reduction_stats`.
  - `verify_step6_toggles.py`: mirrors the gate; new assertions (no ECP carries
    index 0 or $\ge +2$; every standalone multi-CP cluster is reduced, annulled,
    or irreducible; annihilation hides members; $\chi = 2$ invariant).
  - `tests/test_polarity_detector.py`: 6 unit tests for the decision rule
    (all branches + invalid-polarity `ValueError`).
- **Empirical validation (Pd_20K, V (condensed), k=2.0)**: 28 standalone multi-CP
  clusters -> 12 reduced ($\chi=+1$: 6 E-MAX, 6 E-MIN), **16 annulled**
  ($\chi=0$), 0 irreducible. Displayed: 8 Max / 28 Min / 30 Sad, $\chi = 2$
  invariant preserved. With catastrophes too: 6/12/8 displayed, 20 ECPs, $\chi=2$.
- **MC32 note**: MC32 (1 mult-2 Sad + 3 Min, $\chi=+1$) is a CAT_4 constituent;
  the green saddle near it is `E-SAD_CAT_4` (net $\chi=-2$, correct monkey
  saddle). Standalone it reduces to E-MIN $\chi=+1$ (3-lobed basin), which is
  topologically sound. Open question for next session: whether a $\chi=+1$ basin
  containing a mult-2 monkey saddle should prefer the saddle glyph with a
  basin annotation.
- **Tooling lessons**:
  - `insert_edit_into_file` auto-generated three bogus tests in
    `test_polarity_detector.py` (calling the new function with a
    `classify_cluster_polarity` signature). Always `grep -n "^def test"` after
    inserting into test files; delete artifacts with `sed -i '' 'A,Bd'`.
  - trame widgets are **not callable** (`Div(...)(...)` -> TypeError); children
    must go through `with` blocks. `v_text` is unproven in this codebase - use
    mustache positional children.
  - `ruff format .` reformats 6 legacy files outside the pre-commit scope
    (`^(gba_topology2/|tests/)`); scope format commands to the touched files.
  - Headless trame: `state.flush()` does not fire `@state.change` handlers
    outside the server loop; use the verify harness for pipeline assertions and
    the `s.start` stub only for UI-build smoke tests.

---

### [Entry 015] Catastrophe Claim Collision Fix & Reset to Defaults (`sphere_viewer.py`)

- **Reported symptom (Pd_20K, modified Willmore energy)**: on symmetric
  micro-clusters MC21 / MC22, enabling "Reduce Multi-CP Clusters" produced a
  saddle for one and *nothing* for the other; with both reduction toggles on,
  neither showed a saddle. The clusters are long noisy strings (25-28 CPs,
  ~12 Max / 13-14 Sad, $\chi = -1$, aspect ratio ~8) linking two maxima along
  one direction and two minima along the perpendicular direction.
- **Root cause**: `cat_cluster_ids` was built **unconditionally** from every
  detected catastrophe's constituents, and the standalone multi-cluster stage
  skipped any cluster in that set. When `reduce_catastrophes` was **off**, the
  catastrophe stage collapsed nothing, so a claimed cluster was reduced by
  *neither* stage: its member glyphs vanished and no effective CP was emitted.
  Ownership of a cluster must follow the stage that actually performs the
  collapse.
- **Fix**: extracted `catastrophe_claimed_cluster_ids(catastrophes,
  reduce_catastrophes)` as a module-level, unit-testable subroutine. It returns
  the empty set when catastrophe reduction is disabled, and the full set of
  constellation members when enabled. `verify_step6_toggles.py` mirrors the
  rule.
- **Reset to Defaults**: added `DEFAULT_UI_STATE` as the single canonical
  baseline, now consumed by *both* the initial state seeding (replacing 19
  literal assignments) and the new `ctrl.reset_defaults` action, so the two
  cannot drift. `selected_field` is deliberately excluded - the field is a data
  choice, not an analysis parameter. A `suppress_param_change` latch makes the
  19-key batched write rebuild the pipeline once instead of 19 times. Button
  added to the toolbar next to "Reset Camera".
- **Empirical validation (Pd_20K, modified Willmore energy, 314 Max / 67 Min /
  363 Sad, $\chi = 2$)**:

  | k_pitch | catastrophes | E-SAD (old rule) | E-SAD (new rule) | recovered |
  |---------|--------------|------------------|------------------|-----------|
  | 2.0     | 0            | 20               | 20               | +0        |
  | 2.4     | 6            | 8                | 26               | **+18**   |
  | 3.0     | 8            | 0                | 24               | **+24**   |

  At $k = 3.0$ the old rule collapsed **every** effective saddle to zero - the
  six 26-28 CP $\chi = -1$ strings (MC1-MC6, AR ~7.7-8.8) were all swallowed by
  catastrophes and stranded. The new rule restores all 24.
- **Key empirical finding for the string problem**: the long saddle strings are
  already detected as single $\chi = -1$ clusters at $k = 2.0$ (MC1-MC6,
  28 CPs, AR ~8.7, morphology `string`) and reduce correctly once the collision
  is fixed. Raising $k$ *merges* neighbouring strings together and triggers
  spurious catastrophe detection, which is what made the behaviour look
  erratic. The prescreen/aggregation work is therefore about **stabilising**
  these clusters against the catastrophe stage, not about merging them.
- **Tests**: `tests/test_sphere_viewer_reduction.py` (8 tests) - claim rule
  both toggle states, empty list, shared cluster, defaults coverage against the
  `@state.change` key list, pinned baseline values, non-mutation of the shared
  table. Suite: 53 passed. `ruff format` / `ruff check` / `mypy` clean.
- **Open next**: (a) ~~gate catastrophe *detection* so a $\chi \le -1$ string is
  not consumed as a flanking saddle~~ -> **done in Entry 016**;
  (b) quadrupolar prescreen (2 Max longitudinal + 2 Min transverse) to
  recognise these corridors explicitly and bridge fragments such as MC11+MC55;
  (c) wire `string_aggregator.py` into the pipeline.

---

### [Entry 016] Flanking-Saddle Eligibility Gate (`catastrophe_classifier.py`)

- **Symptom**: on the modified Willmore energy, raising `k_pitch` produced
  spurious catastrophes (0 at $k=2.0$ -> 6 at $2.4$ -> 8 at $3.0$) that
  swallowed the long $\chi = -1$ saddle strings MC1-MC6 (26-28 CPs, AR ~7.7-8.8).
  Symmetric clusters behaved asymmetrically because constellation assembly is
  order-dependent once strings enter the pool.
- **Root cause**: the flanking pool was `n_saddles > 0`. A $+1$ core plus three
  $\chi = -1$ strings sums to $1 + 3(-1) = -2$, which matches the monkey-saddle
  template exactly. Index arithmetic alone cannot distinguish a simple saddle
  from a composite string of the same net index.
- **Fix**: `is_flanking_saddle_candidate(cluster)` - admit a cluster only when it
  has saddles, **no extrema**, and net index exactly $-1$. This is the
  topological statement of "simple saddle" required by the index balance
  $\chi_{\text{core}} + \sum_i \chi_i = 1 - k$. It rejects composite strings
  (extrema present), degenerate multi-saddle sets ($\chi \le -2$), and neutral
  dipoles ($\chi = 0$).
- **Empirical A/B (gate off -> gate on)**:

  | field | k | cats off | cats on |
  |-------|---|----------|---------|
  | V (condensed) | 2.0 | 8 | **8** |
  | V (condensed) | 2.4 | 0 | 0 |
  | V (condensed) | 3.0 | 0 | 0 |
  | modified Willmore | 2.0 | 0 | 0 |
  | modified Willmore | 2.4 | 6 | **0** |
  | modified Willmore | 3.0 | 8 | **0** |

  Genuine catastrophes are untouched (all 8 monkey saddles on `V (condensed)`
  preserved, identical members). The $k=2.4$ drop on `V (condensed)` is
  **pre-existing** search-window geometry, not this gate - verified by A/B.
  Willmore E-SAD count is now stable across $k$: 20 / 26 / 24 / 24, with
  **0 strings swallowed** at every $k$.
- **Fixture lesson**: the real MC1 reaches $\chi = -1$ via a **multiplicity-2
  saddle** (12 Max + 2 Min, 14 saddles of summed multiplicity 15). A test
  fixture using 14 unit saddles gives $\chi = 0$ and silently tests the wrong
  branch. `total_saddle_index`, not `n_saddles`, drives `local_euler_index`.
- **Harness fix**: `verify_step6_toggles.py`'s standalone-pool assertion
  hard-coded the *old* unconditional claim rule on its right-hand side, so it
  failed once Entry 015's toggle-gated claim rule was in place. It now derives
  the pool from `catastrophe_claimed_cluster_ids`, keeping harness and viewer
  on one definition.
- **Tests**: `tests/test_flanking_saddle_gate.py` (9 tests) - predicate
  admit/reject branches plus three integration cases. Verified gate-sensitive:
  disabling the gate fails 2 of them. Suite: 62 passed.
- **Open next**: (b) quadrupolar prescreen for corridor recognition and
  fragment bridging (MC11+MC55); (c) wire `string_aggregator.py` in.

---

### [Entry 017] Corridor Bridging Into Reduction Units (`string_aggregator.py`, `sphere_viewer.py`)

- **Goal**: a ridge/valley corridor fragmented by the isotropic clustering
  radius must collapse to **one** effective saddle at its centre. Target case on
  the modified Willmore energy: MC32 + MC43 + MC34.
- **New abstraction**: `ReductionUnit` in `sphere_viewer.py` -- one set of CPs
  reduced to at most one effective CP, either a lone micro-cluster (`MC12`) or a
  bridged corridor (`MS1`). `build_reduction_units()` partitions clusters into
  units; `macro_string_to_cluster()` adapts a `MacroString` to the `MicroCluster`
  interface so `classify_cluster_polarity` and `harmonic_nudge` consume a
  corridor unchanged, with no parallel code path. The Step 6 loop iterates units.
- **Bridging rule (the substantive finding)**: centroid separation is the wrong
  metric. For elongated clusters it is dominated by the two half-lengths, so
  collinear fragments sitting tip-to-tip look far apart while long strings
  crossing at 60 degrees look close. `bridge_all` mode now links two clusters
  when the **uncovered gap** (`centroid_dist - semi_major_i - semi_major_j`) is
  within `k_bridge` delta_theta units **and** their principal axes are within
  30 degrees. Measured on Pd_20K modified Willmore at k_pitch=2.4:

  | pair | gap | axis angle | verdict |
  |------|-----|-----------|---------|
  | MC32-MC43 | 1.45 deg (0.9 dth) | 11.9 | merge |
  | MC43-MC34 | 1.45 deg | 11.9 | merge |
  | MC32-MC34 | 8.03 deg | 23.7 | not adjacent (MC43 is the bridge) |
  | MC21-MC22 | 13.09 deg | 90.0 | perpendicular, correctly separate |
  | MC17-MC19 (and 8 more) | -7.69 deg | 59.9 | crossing, must not merge |

  Gap alone would merge the 60-degree crossings; axis angle alone would not
  separate the collinear pair. **Both tests are required.**
- **Qualification moved to the pooled group**: a multi-cluster group becomes a
  corridor only when its own tangent-plane aspect ratio reaches
  `aspect_ratio_threshold`, which keeps isotropic blobs from fusing. A lone
  compact cluster is never promoted.
- **Result (Pd_20K, modified Willmore, k_pitch=2.4, k_bridge=3.0)**:
  54 clusters -> 50 units, 6 bridged.
  `MS1 ['MC32','MC43','MC34'] 21CP chi=-1 -> E-SAD`
  `MS2 ['MC33','MC44','MC35'] 21CP chi=-1 -> E-SAD` (the symmetry twin).
  Index sum preserved: clusters 2 == units 2.
- **Invariants kept**: catastrophe classification still runs on raw clusters, so
  detection is identical with aggregation on or off. The claim rule matches
  through `origin_ids`, so a corridor is claimed when any fragment is and
  aggregation cannot smuggle a catastrophe's index into a second glyph.
  `aggregate_strings` defaults to **off**, making the initial state byte-identical
  to the pre-aggregation pipeline.
- **Ordering**: units are sorted by (-size, earliest source cluster) to match
  `compute_micro_clusters`. Sorting by `unit_id` interleaves MC10 before MC2.
- **Test-fixture lesson**: a synthetic string whose members all sit on the
  centroid has a degenerate pooled PCA, so the aspect-ratio gate rejects every
  merge and the collinearity test passes vacuously. Members must be spread along
  the axis. Verified load-bearing by A/B: dropping `and collinear` fails
  `test_non_collinear_fragments_rejected_when_ar_qualifies`.
- **Cosmetic**: a one-constituent macro-string keeps its `MC` id (renaming it
  `MS3` implies a merge that never happened); pooled AR clamped at 999 for
  display, since collinear 3-point clusters report AR ~4e5.
- **Tests**: `tests/test_corridor_aggregation.py` (12 tests) -- bridge/no-bridge
  geometry, disjointness, index preservation, identity partition when off, id
  preservation, adapter contents, empty input. Suite: 74 passed.
- **Open next**: quadrupolar prescreen (2 Max longitudinal + 2 Min transverse) to
  tag corridors explicitly rather than inferring them from geometry alone.

## 2026-10-10 -- Local atom coordinates (off-origin atoms)

- **Symptom** (reported on `ethene3.plt`): boundary rings stood normal to the
  sphere surface instead of lying flat on it.
- **Root cause**: every angular formula in the package derives the outward normal
  as `p / ||p||` and rebuilds surface points as `dir * R`. That is only the
  normal when the sphere is centred on the world origin. `Pd_20K.plt` happens to
  sit at the origin, which hid the assumption for the whole project.
- **Measured** (`gba_topology2/inspect_atom_spheres.py`, read-only diagnostic):

  | file | atom | offset | R (aux) | R (fitted) | R naive | R inflation | δθ true | δθ naive |
  |---|---|---|---|---|---|---|---|---|
  | ethene3 | C1 | 1.2618 Å | 0.412733 | 0.412725 | 1.306666 | 3.17x | 2.2278° | 0.7036° |
  | Pd_20K | Pd1 | 0.000011 Å | 1.038471 | 1.038471 | 1.038471 | 1.00x | 1.6058° | 1.6058° |

  Aux `SphereRadius` agrees with the fitted radius to 0.002%, so the metadata is
  authoritative. The naive radius inflated R 3.17x and shrank δθ to 0.32x, so the
  cluster radius `k · δθ` was 3.2x too small -- CPs that belong together were
  split, and the ring generator built rings on a sphere about the origin, which
  is exactly the "ring normal to the surface" report.
- **Fix, applied once at the boundary**: `extract_atom_sphere_mesh()` now recovers
  the atom's own frame (`compute_atom_frame`: centre = vertex centroid, radius =
  mean centre-relative distance, overridden by the zone's aux `SphereRadius` when
  present), translates the mesh in place, and records `center`, `center_offset`,
  `fitted_radius` and `radius_source` in `atom_info`. Every algorithm keeps
  assuming a centred sphere, so `micro_cluster`, `catastrophe_classifier`,
  `excision_boundary`, `harmonic_nudge` and `polarity_detector` are untouched --
  no `sphere_center` parameter threaded through them.
- **Frame contract** (the part to remember):
  - **Local / centred**: `pts_np`, `CriticalPoint.position`, cluster and
    catastrophe centroids, boundary rings, ports, nudge positions, R, δθ.
  - **Global / molecular**: the UI dicts (`all_cps_list`, `clusters_list`,
    `catastrophes_list`), all VTK geometry, and the picker's `pick_pos`.
    The UI dicts feed the inspector text, the glyphs and the picker, so making
    them global satisfies all three consumers with one convention.
  - `runtime_cps` / `runtime_clusters` stay local, so `on_scene_click` converts
    the pick with `to_local` before measuring against them.
  - `centroid_unit` is a direction, so it is translation invariant -- unchanged.
- **`gba_topology2/src/sphere_frame.py`**: `compute_atom_frame`, `to_local`,
  `to_global`, `to_local_array`, `to_global_array`, `radial_offset`. The point of
  `radial_offset` is that scaling a centre-relative point is a genuine lift along
  the atom's normal, while scaling a world point is not -- the three ring
  z-fight factors (1.008 / 1.012 / 1.014) now run in the local frame. Ports
  previously had no lift at all and z-fought; they now get the 1.008 lift.
- **`analyze_mesh_geometry`** takes an optional `sphere_radius` override so the
  authoritative per-atom radius drives the pitch, matching the viewer.
- **Multi-atom note**: each analysed atom has its own `AtomSphereData` zone, so
  each gets its own centre, radius and δθ (C ~0.413 Å vs H ~0.28 Å). Selecting
  `-a N` resolves that atom's zone metadata; nothing is shared between atoms.
- **Verification**:
  - ethene3 C1, electron density condensed: χ = 2 valid (17 Max / 22 Min /
    37 Sad), 12 clusters, 7 multi-CP, Σχ = 2, ring scale `k·δθ` = 4.46° (was
    1.41°). Cluster angular diameters now 4-20°, i.e. physically sensible.
  - Pd_20K: `verify_step6_toggles.py` "ALL ASSERTIONS PASSED", χ = 2 in every
    toggle combination, first ECP position unchanged -- zero regression, as
    expected for an atom already at the origin.
  - Headless layout build passes for both files. Suite: 91 passed.
- **A/B discipline**: `test_naive_origin_radius_is_inflated_for_offset_atom`,
  `test_radial_offset_about_world_origin_is_the_bug`,
  `test_boundary_ring_from_world_coordinates_is_the_bug` and
  `test_pipeline_is_translation_invariant_after_centring` all assert the *old*
  behaviour is wrong, so reverting the fix fails the suite rather than passing
  silently. The measured buggy ring floats entirely outside the atom (all points
  ~2.2 R from the nucleus, height swinging 0.29 R around the loop) -- it does not
  pierce the sphere, which is why the symptom reads as "standing on edge".
- **UI Addition (num_contours slider)**: added an interactive slider (range 3 to 50,
  step 1) with dynamic count label in the "Condensed Field & Mesh" drawer card beneath
  the "Isocontour Lines" switch, allowing live tuning of the number of contour levels
  rendered on the atom surface.
- **Open next**:
  1. Glyph and tube radii (0.045, 0.065, 0.010 Å) are absolute and tuned for
     Pd's R ≈ 1.04 Å. On a 0.41 Å carbon sphere they are ~2.5x oversized.
     Recommend scaling by `R / R_reference` as its own atomic unit.
  2. The orientation-axes widget stays at the world origin while the atom renders
     offset, so it appears far from the sphere. Consider translating it to the
     nucleus.
  3. `polarity_detector.evaluate_spherical_radial_slope` still accepts a
     `sphere_radius` argument it never uses.

---

### [Entry 018] Step 1 Basin Locator: Basin Center Management & Interactive UI (`basin_center.py`, `sphere_viewer.py`)

- **Objective**: Implement Step 1 of the incremental Basin Locator pipeline to extract, structure, and manage basin centers (canonical raw extrema and reduced effective extrema) with an interactive 3D highlight halo and selection inspector in `sphere_viewer.py`.
- **Changes**:
  - `gba_topology2/src/basin_center.py`:
    - Created `BasinCenter` dataclass representing a basin center with fields: `center_id`, `polarity` (`"maximum"` vs `"minimum"`), `position`, `value`, `is_effective`, `vertex_id`, `origin_id`, `net_index`, and `member_count`.
    - Implemented `extract_basin_centers`: filters out hidden raw CPs that were absorbed into effective entities, ingests visible raw maxima and minima, ingests effective extrema (`E-MAX` and `E-MIN`), and returns a sorted list descending by scalar value.
    - Implemented `find_basin_center_by_id`: fast lookup helper by unique string ID.
  - `tests/test_basin_center.py`:
    - Unit tests covering `BasinCenter` validation, mixed raw + effective CP extraction, saddle/hidden exclusion, descending value ordering, and ID lookup.
  - `gba_topology2/src/sphere_viewer.py`:
    - Integrated `extract_basin_centers` into the topology update pipeline.
    - Added dedicated VTK gold wireframe sphere actor (`basin_center_actor`, color pure gold `(1.0, 0.84, 0.0)`, line width `3.5`) to highlight the active basin center.
    - Added drawer card **Basins: Step 1 Centers** with a candidate basin centers table displaying ID, effective badge, polarity chip (MAX in red, MIN in blue), and scalar value.
    - Added **Basin Center Inspector Card** displaying center origin/vertex details, polarity, exact coordinates in molecular frame, scalar value, and member counts.
    - Wired selection/deselection handlers (`select_basin_center`, `select_basin_center_from_list`, `clear_basin_center_selection`) and linked into `reset_defaults`.
- **Validation**:
  - Unit tests: 93 passed in `pytest`.
  - Format, lint, & typecheck: `ruff check`, `ruff format`, `mypy` passing cleanly with 0 errors across all 30 source files.
  - Pre-commit hook suite (`uv run pre-commit run --all-files`): all 4 hooks passed cleanly.

---

### [Entry 019] Basin Center Frame Fix: Off-Sphere Halos on Off-Origin Atoms (`basin_center.py`)

- **Symptom** (reported on `ethene3.plt`, C1): selecting a basin center from the
  "Basins: Step 1 Centers" table placed the gold halo a long way off the sphere.
- **Measured** (`gba_topology2/inspect_basin_centers.py`, read-only diagnostic
  reporting `||p - center|| / R`, which must be ~1.0 for a point on the sphere):

  | case | min d/R | median d/R | max d/R |
  |---|---|---|---|
  | raw centers, nucleus omitted (the bug) | 2.0622 | **2.6357** | 4.0617 |
  | raw centers, nucleus passed (fixed) | 0.9949 | **0.9972** | 1.0050 |
  | effective `E-MAX` center (world-frame dict) | - | 0.9973 | - |

- **Root cause**: the frame contract was broken at the extraction boundary.
  `CriticalPoint.position` is **centred-frame** (nucleus at the origin), while
  the `effective_cps` dicts are built by the viewer's UI layer and are already
  **world-frame**. `extract_basin_centers` emitted raw positions verbatim, so on
  an atom offset by 1.2618 A the raw centers were drawn relative to the world
  origin -- up to 4.06 R away. `Pd_20K.plt` sits at the origin, so the two
  frames coincide and the bug is invisible there (d/R = 1.0000 before and after).
- **Fix**: `extract_basin_centers(..., center=None)` maps raw CP positions with
  `sphere_frame.to_global` and returns **every** center in the world frame,
  matching the existing convention for `all_cps_list` / `clusters_list` /
  `catastrophes_list`. `center=None` keeps the centred behaviour for
  origin-centred synthetic meshes. The `BasinCenter.position` docstring now
  states the frame explicitly.
- **Note on the report**: the measured defect is on the **raw** `MAX_*` /
  `MIN_*` rows. The `E-MAX_MC*` rows were already world-frame and verified
  on-sphere (d/R = 0.9973), because `effective_cp_to_ui` receives `center`.
- **Tests**: `tests/test_basin_center.py` +2 tests -- raw centers mapped out of
  the centred frame (exact world coordinates + on-sphere distance, and the
  pre-fix pass-through pinned as the origin-centred-only case), and the mixed
  case where a world-frame effective dict and centred-frame raw CPs must land on
  the same sphere. Suite: 95 passed. `ruff` / `mypy` clean.

---

### [Entry 020] Step 2 Basin Locator: Saddle Launch Vectors & UI Visual Deliverable (`saddle_launch.py`, `sphere_viewer.py`)

- **Objective**: Implement Step 2 of the incremental Basin Locator pipeline to extract separatrix launch branches from saddle critical points and render directional ascent/descent launch rays on the sphere in `sphere_viewer.py`.
- **Mathematical Approach**:
  - Rejected the standard surface Hessian approach because higher-order saddles (e.g. $k=3$ monkey saddles where $f \sim x^3 - 3xy^2$) have degenerate second derivatives where Hessian eigenvectors are ill-defined.
  - Implemented the cyclic-link sector scan: orders the 1-ring neighbours cyclically around vertex $v$, locates continuous sign crossings where $f(u) - f(v) = 0$, partitions the cycle into $2k$ sectors ($k$ ascent/ridge sectors and $k$ descent/valley sectors), and computes sector bisectors on the sphere tangent plane.
  - Applies geodesic exponential map stepping: $\vec{p}_{\text{launch}} = R \cdot (\cos\theta\,\hat{n} + \sin\theta\,\hat{v})$ to place launch seed coordinates strictly on the sphere surface.
  - Verifies sector count against discrete Morse multiplicity ($\mu = k - 1$) and surfaces any index mismatch.
- **Changes**:
  - `gba_topology2/src/saddle_launch.py`:
    - Created `LaunchBranch` and `SaddleLaunchSeed` dataclasses.
    - Implemented `build_cyclic_link`, `extract_saddle_launch_seed`, and `compute_all_saddle_launch_seeds`.
  - `tests/test_saddle_launch.py`:
    - 11 unit tests covering cyclic link cycle validation, simple saddle 2-ascent/2-descent branches, $k=3$ monkey saddle 3-ascent/3-descent branches ($120^\circ$ orthogonality on tangent plane), on-sphere geodesic radii, normal orthogonality, scale factor, and non-saddle/error handling.
  - `gba_topology2/src/sphere_viewer.py`:
    - Added VTK pipelines for launch rays: Orange-Red tubes for ascent (towards maxima) and Cyan tubes for descent (towards minima).
    - Added drawer card **Basins: Step 2 Launch Seeds** with toggle switch, launch offset slider (`launch_seed_pitch`), branch count summary chips (Ascent, Descent, Max $k$), and index mismatch warnings.
    - Added `show_launch_seeds` and `launch_seed_pitch` to `DEFAULT_UI_STATE` and reactive state handler.
- **Validation**:
  - Unit tests: 106 passed in `pytest`.
  - Formatting, linting, and type safety: `ruff check`, `ruff format`, and `mypy` passed with 0 errors across 33 source files.
  - Pre-commit hook suite passed cleanly.

---

### [Entry 021] Directional Support Radius for Corridor Bridging (`string_aggregator.py`)

- **Symptom**: on the curvedness field of `Pd_20K` at `k_pitch=1.6`, MC18 (21 CPs,
  chi = -1, AR = 14.4) reduced correctly to a single effective saddle with
  bridging off. With bridging on it absorbed MC131, a **lone minimum sitting at
  90 degrees to the corridor axis and visibly displaced from it**, giving
  MS12 = [MC18, MC131], 22 CPs, chi = 0 -> annihilation. A genuine saddle
  vanished and its ring turned ghost grey.
- **Root cause**: the `bridge_all` gap test subtracted each cluster's scalar
  `semi_major_rad` from the centroid separation. That subtraction is
  **isotropic**, so a long thin corridor is treated as a disk of radius `a`
  rather than the ellipse it actually fitted. Measured on the target pair:

  | quantity | value |
  |---|---|
  | MC18 semi-major / semi-minor | 16.76 deg / 2.77 deg |
  | centroid separation MC18-MC131 | 17.53 deg (10.91 dth) |
  | bearing of MC131 from MC18 axis | **90.0 deg** |
  | old isotropic gap | **-0.46 dth** (limit 3.0) -> linked |
  | new directional gap | **8.26 dth** -> rejected |

  The axis-collinearity test could not help: `_axis_pairwise_angle` returns
  `None` when either cluster is compact (a point has no meaningful axis), so
  `collinear` was `True` vacuously. The pooled aspect-ratio gate also passed,
  since one extra point barely dents the AR of a 21-CP string. The gap test was
  the only gate in play, and it was measuring the wrong shape.
- **Fix**: `cluster_directional_support_radius(cluster, target_unit)` evaluates
  the polar form of the cluster's fitted boundary,
  $r(\phi) = \sqrt{(a\cos\phi)^2 + (b\sin\phi)^2}$, at the bearing of the
  neighbour measured on the tangent plane at the cluster centroid, with
  $\phi$ folded to [0, 90] degrees because the principal axis is an undirected
  line. The gap becomes
  $\text{gap} = \theta_{ij} - r_i(\phi_i) - r_j(\phi_j)$.
  Full length along the axis, narrow width sideways. A compact cluster has
  $a == b$ and returns $a$ unchanged, so isotropic fragments are untouched.
  Falls back to the semi-major on ambiguous geometry (target at the centroid or
  antipodal, null axis) -- the conservative choice, never shrinking reach on
  geometry the code cannot resolve.
- **Why this form**: it reuses the exact polar expression
  `generate_elliptical_boundary_ring` already uses to draw the ring, so the
  reach the bridging test grants is the boundary the user sees on screen. No new
  parameter, no new threshold, symmetric in the two clusters.
- **Result (Pd_20K curvedness, k_pitch=1.6, k_bridge=3.0)**:

  | rule | MC18 unit | units | bridged | sum chi |
  |---|---|---|---|---|
  | old isotropic | MS12 [MC18, MC131] 22CP chi=0 | 96 | 94 | 2 |
  | new directional | MC18 21CP chi=-1 | 98 | 90 | 2 |

  The saddle survives; MC131 stays a standalone +1. Index sum preserved (2 == 2).
- **Lesson on A/B harnesses**: the first A/B run on the real field printed
  nothing and was read as "the bug does not reproduce". The field array name
  contains a non-ASCII prefix (`Ï curvedness (condensed)`), the lookup raised
  `AttributeError` inside `numpy_support.vtk_to_numpy`, and piping through
  `grep` swallowed the traceback. A/B runs must assert on a positive marker,
  not on the absence of output. Passing the field name through a shell `-c`
  string also mangles it; use a script file.
- **Tests**: `tests/test_corridor_aggregation.py` -- 8 new tests: support radius
  along axis, transverse, axis-sign folding, 45-degree polar value,
  unnormalised target, compact isotropy, degenerate fallbacks, plus the two
  end-to-end cases (perpendicular compact fragment rejected, on-axis tip
  fragment still bridged). The perpendicular test asserts the old isotropic gap
  *would* have linked, pinning the regression. Suite: 115 passed.
- **Open next**: the collinearity skip for compact clusters is still a
  deliberate blind spot -- a compact fragment at the *tip* of a corridor joins
  from any direction, which is correct, and one sitting inside the corridor's
  own bounding ellipse joins too. The quadrupolar prescreen (2 Max longitudinal
  + 2 Min transverse) from Entry 017 remains the type-aware route to corridors.

---


### [Entry 022] "Save Parameters" Diagnostic JSON Export (`sphere_viewer.py`)
- **Objective**: One-click capture of the full viewer GUI state so issues
  observed on screen can be reproduced in headless diagnostic scripts without
  manually re-specifying every toggle, slider and field selection.
- **Changes**:
  - `save_params_to_dict(state_like, *, plt_path, atom_info, atom_num)` --
    pure, duck-typed snapshot builder. Emits `schema_version` (=1), `tool`,
    ISO-8601 UTC timestamp, `params` split into `bools` (16 switches),
    `floats` (8 sliders, ints widened), `strings` (`selected_field`), plus
    `context` (abs .plt path, atom, discovered `field_items`),
    `derived_metrics` (geom_info, morse_counts, display_counts,
    reduction_stats, launch_stats -- deep-copied via json round-trip) and
    `selection` (CP/cluster/catastrophe/basin-center ids only).
  - `load_params_from_dict(payload)` -- validates schema_version, requires
    every control key, coerces strictly (bools reject ints, sliders reject
    bools); returns a flat dict safe for `state.update()` plus passthrough
    diagnostic sections. Metrics are outputs, never inputs.
  - `default_params_path(plt_path, atom_num)` -- `<stem>_atom<N>_params.json`
    next to the .plt.
  - Type tables `PARAM_BOOL_KEYS`/`PARAM_FLOAT_KEYS`/`PARAM_STRING_KEYS` are
    derived from `DEFAULT_UI_STATE` at import and checked to partition it
    exactly, so a new control is covered by construction.
  - UI: toolbar "Save Parameters" button -> `@ctrl.add("save_parameters")`
    writes the JSON (`indent=2, sort_keys=True`), prints the abs path, and
    shows a `params_last_saved_path` confirmation chip in the drawer. That
    key is deliberately outside `DEFAULT_UI_STATE` and the `@state.change`
    decorator: it triggers no recompute and survives "Reset to Defaults".
- **Verification**: `gba_topology2/verify_params_export.py` fires the export
  headlessly on ethene4.plt atom 1 at default values (995 nodes, chi=2, 8 Max
  / 6 Min / 12 Sad); the written JSON round-trips through
  `load_params_from_dict` back to all 24 controls + `selected_field`.
  Generated `*_params.json` files are gitignored.
- **Tests**: `tests/test_sphere_viewer_params_export.py` -- 14 tests: control
  coverage parsed from the `@state.change` decorator source (same guard as
  Entry 020's defaults test), widget-type fidelity, json round-trip, context/
  metric/selection capture, metric deep-copy, missing-key and type-mismatch
  rejection on both export and load, int-widening, path naming, table
  partition. Suite: 129 passed.

---

### [Entry 023] `diagnose_params.py`: Headless Stage Runner over Saved Params
- **Objective**: Turn a "Save Parameters" JSON (Entry 022) into a reproducible
  headless run: re-execute every analysis subroutine the GUI controls trigger,
  in the viewer's own order, and answer "does this parameter set reproduce the
  observation?" with an exit code.
- **Changes**: `gba_topology2/diagnose_params.py`.
  - `RunContext` dataclass carries the mesh (pts/triangles/f_vals, centred
    frame), the loaded control dict, and every stage's accumulated results.
  - Nine stage functions in `update_topology()` order: geometry, morse,
    clusters, units, catastrophes, ports, reduction, basins, launch -- each
    with the viewer's exact arguments (tie_break 1e-15, margin 1.5 deg, 48
    ring points, catastrophes on raw clusters not bridged units). `--only
    <stage>` stops after the named stage; consumers guard with
    `require_morse()` so a hand-built context fails loudly.
  - `stage_reduction` replicates the viewer's inline Step 6 block
    (catastrophe collapse, polarity/index decision, annulment/irreducible,
    nudge, display counts) -- the same duplication `verify_step6_toggles.py`
    carries. The clean fix is extracting a shared `run_effective_reduction()`
    subroutine; deferred as its own atomic unit.
  - `diff_metrics` walks the saved `derived_metrics` leaf-wise (float
    tolerance, strict bools, element-wise list leaves) and skips sections the
    run did not reach, so `--only` stops stay comparable. Exit 1 on any drift.
  - `--set key=value` applies typed overrides (bools true/false, floats, the
    field selector) for parameter sweeps; `--out file.json` dumps
    per-entity results (clusters, catastrophes, effective CPs, basin centers,
    launch seeds) for programmatic A/B diffing. `--plt/--atom/--field`
    override the recorded context when the .plt moved.
- **Verification**: on the live-saved `ethene4_atom1_params.json` (viewer
  session, atom C1, Electron Density): full run reproduces **35/35 saved
  metric leaves** (geom, morse 6/8/12 chi=2, 12 clusters, 1 split k=3
  catastrophe 3V+3R, 14 basin centers, 12 saddles -> 24+24 branches), exit 0.
  `--only clusters` compares 14/14; `--set reduce_multi_clusters=true`
  correctly reports 10 drift leaves and exits 1.
- **Tests**: `tests/test_diagnose_params.py` -- 24 tests: leaf comparison
  semantics (bool strictness, tolerance, shape mismatch), recursive diff
  (drift isolation, missing-section skip, list leaves), typed overrides,
  stage order, early stop, morse-first guard, and an end-to-end quadrupole
  octahedron run (2 Max/2 Min/2 Sad, chi=2 preserved through reduction and
  launch). Suite: 154 passed.

---

### [Entry 024] Troubleshooting Protocol in AGENTS.md; Fix Displayed-Count Bug
- **Objective**: Make "reproduce in the viewer, export params, point the agent
  at the JSON" the standard troubleshooting entry point -- and let the
  reproduce check earn its keep by fixing a real runner bug it caught.
- **Changes**:
  - `gba_topology2/AGENTS.md` gained a mandatory **Section 3: Issue
    Troubleshooting Workflow**: the four-step protocol (user reproduces ->
    clicks "Save Parameters" -> describes the issue and points at the JSON ->
    agent investigates with `diagnose_params.py`), the CLI surface with
    worked examples, the stage list, the exit-code contract, and the
    expectation that the runner itself may be extended to gather the
    information a given investigation needs. It also states that a
    reproduce-check mismatch is itself a finding, and that display-only
    controls are inert headlessly.
  - `gba_topology2/diagnose_params.py` -- **bug fix**: `stage_reduction`
    computed the displayed maxima/minima/saddles counts from the visible raw
    CPs only. The viewer's glyph bookkeeping also adds one glyph per
    effective CP (E-MAX -> maxima, E-SAD -> saddles), so with the reduction
    toggles on the runner under-reported both the counts and the index sum.
    The counts are now built exactly as the viewer builds them.
- **How the bug was found**: the user re-saved `ethene4_atom1_params.json`
  from a live viewer session with `reduce_catastrophes` and
  `reduce_multi_clusters` enabled and `cat_barrier_tol = 0.001`. The runner
  reported 2 drifting leaves (`display_counts.maxima` 2->0,
  `display_counts.saddles` 3->1) on a *reproduction* run -- the reproduce
  check flagged the runner, not the data. This is the workflow working as
  intended: the same JSON that reproduces cleanly under the old default
  toggles exposed a divergence the earlier, all-defaults run could not see.
- **Verification**: after the fix, the same saved file reproduces **35/35**
  leaves with exit 0 (reduction on: 4 effective CPs, 21 raw CPs hidden).
  The all-defaults case still reproduces 35/35.
- **Tests**: two new regression guards in `tests/test_diagnose_params.py` --
  displayed counts equal raw-visible + effective-CP glyphs, and the index
  sum stays 2 across the collapse; plus the reduction-off case asserting the
  raw-only counts. Suite: 156 passed.



---

### [Entry 025] Fix: Split Catastrophes Were Collapsed, Hiding Real Extrema
- **Objective**: Resolve the user-reported defect on `ethene4_atom1_params.json`
  (atom C1, Electron Density): a clear maximum had **no maximum glyph**, a
  spurious saddle glyph appeared nearby, and clicking where the maximum should
  be selected MC3 ("2 max, 1 sad, net index +1"), which did not match what was
  drawn.
- **Diagnosis** (via `diagnose_params.py`, per the Section 3 protocol):
  - Full run reproduced the saved observation **35/35, exit 0** — the runner
    and the viewer agreed, so the defect was in the pipeline, not in the
    bookkeeping.
  - `--only catastrophes` showed **1 split** constellation (`CAT_1`,
    `physical_symmetry_breaking`), i.e. `is_fused == False`.
  - `--out` per-entity dump showed `CAT_1`'s constituents were
    **`MC3, MC12, MC11, MC10`** — i.e. **MC3 was absorbed by the catastrophe**.
  - `--set reduce_catastrophes=false` was the decisive bisect: displayed
    counts flipped `maxima 2 -> 3`, `saddles 3 -> 5`, and
    `n_catastrophes_reduced 1 -> 0`.
- **Root cause (two compounding defects)**:
  1. **The spurious/physical distinction was computed and then discarded.**
     `classify_catastrophes` classifies a constellation as `spurious_unfolding`
     (numerical artefact, safe to fuse) or `physical_symmetry_breaking`
     (genuine, distinct poles), and its docstring states a split constellation
     is *"Preserved as distinct physical simple saddles"*. But the viewer's
     Step 6 block collapsed **every** catastrophe into one `E-SAD` at the
     constellation centroid, ignoring `is_fused` entirely. `is_fused` was
     computed, exported, and displayed in the sidebar table — and never acted
     upon.
  2. **CAT_1 is a false-positive constellation.** `classify_catastrophes`
     accepts a constellation on *index arithmetic alone*
     (`chi_core + k*(-1) == 1-k`) with no geometric validation of the k-fold
     pattern. CAT_1's three "flanking saddles" lie **79.7° / 83.5° / 84.7°**
     from the core — across the sphere, not ringing it. The classifier
     computes the azimuthal symmetry deviation σ (43.09° against an ideal
     120° spacing) inside `evaluate_constellation_metrics` and then **discards
     it** (`_ang_std`).
- **Why the maximum vanished**: the core cluster MC3 is a *composite* set
  (2 Max + 1 Sad, χ=+1) that holds the field's **global maximum**
  (v = 0.0648, twice the next highest). Collapsing CAT_1 hid all six of its
  members and replaced them with one synthetic `E-SAD` at a centroid that is
  **not a critical point at all** — the "spurious saddle glyph". The maximum
  glyph was gone because it was a member of a hidden set. Clicking the empty
  spot selected MC3 because MC3's centroid is the nearest cluster, which is
  exactly the mismatch the user reported.
- **Fix**: added `fusable_catastrophes(catastrophes)` to `sphere_viewer.py`,
  returning only constellations with `is_fused == True`. The viewer's Step 6
  collapse now gates on `reduce_cats and cat.is_fused`, and
  `catastrophe_claimed_cluster_ids` claims only fusable constellations'
  constituents (so a split constellation's clusters stay eligible for
  standalone reduction instead of being stranded). Mirrored in
  `diagnose_params.py` (`stage_reduction`) and `verify_step6_toggles.py` so the
  three implementations cannot drift.
- **Result**: `E-SAD_CAT_1` is gone; MC3 now reduces to **`E-MAX_MC3`** (χ=+1),
  so the global maximum is visible again. Displayed counts `3 Max / 4 Min /
  5 Sad`, and the displayed **χ stays 2** — the index invariant holds.
- **Runner instrumentation**: the `catastrophes` stage now reports
  `n_fusable` alongside `n_fused`/`n_split`, so a future investigation can see
  at a glance how many constellations the reduction stage may collapse.
- **Open follow-up (not fixed here)**: CAT_1 is still *detected* as a
  catastrophe — it is simply no longer collapsed. The underlying false
  positive remains in `classify_catastrophes`, which accepts a constellation
  on index arithmetic without validating the k-fold geometry and discards the
  azimuthal symmetry metric it already computes. A symmetry/geometry gate on
  detection (reject σ far above the ideal 360/k spacing) is the natural next
  unit of work. Empirically, genuine k=4 constellations on `ethene4` show
  σ ≈ 2.5–14.8°, while the false positive shows 43.1°; note that `ethene3`
  (mesh pitch 2.23°) produces legitimate constellations with σ up to 98.8°,
  so the gate must be scaled by δθ_mesh, not fixed.
- **Tests**: 4 regression tests in `tests/test_sphere_viewer_reduction.py` —
  only fused catastrophes are fusable; an all-split set yields none; a split
  constellation does not claim its clusters; a fused one still does. Suite:
  160 passed.

---

### [Entry 020] Non-Monotonic "Bridge Radius" Response (ethene4 C1, curvedness)

- **Symptom (reported)**: raising the "Bridge Radius" slider (`k_bridge`)
  from 3.0 to 3.25 *increased* the number of visible effective critical
  points (2 → 4) instead of decreasing it. The expectation is that a wider
  bridge merges more fragments, so fewer effective CPs should survive.
- **Reproduction**: two saved parameter snapshots
  (`ethene4_atom1_params a.json` at k_bridge=3.0 and
  `ethene4_atom1_params b.json` at k_bridge=3.25). Both reproduce exactly
  with `diagnose_params.py` (35/35 match), so the divergence is in the
  pipeline, not in the runner.
- **The bridging rule is two-stage**:
  1. A graph edge is created between two micro-clusters when the uncovered
     gap is within `k_bridge`·δθ_mesh **and** their principal axes are within
     30°.
  2. Each connected component of that graph is gated by its **pooled**
     tangent-plane aspect ratio (≥ 2.0). A group failing the gate is
     rejected **whole** and its members revert to standalone units.
- **Root cause**: the stage-2 gate is a *pooled* property and therefore **not
  decomposable**. One off-axis satellite joining a valid corridor widens the
  pooled minor axis enough to sink the whole group below the threshold — and
  rejecting the group annihilates the corridor as well.

  | k_bridge | component | pooled AR | gate | result |
  |---|---|---|---|---|
  | 3.00 (thr 21.26°) | MC1+MC2+MC3 (17 CP) | **2.477** | pass | one corridor `E-MIN_MS1` |
  | 3.25 (thr 23.03°) | MC1+MC2+MC3+MC13+MC14 (19 CP) | **1.543** | fail | corridor destroyed |

  The two extra links are MC2–MC13 (gap 21.55°) and MC2–MC14 (gap 21.57°),
  both just inside the 23.03° threshold. MC13/MC14 are lone saddles hanging
  off MC2 at a bearing roughly perpendicular to the corridor; they widen the
  pooled minor extent from 13.7° to 45.3° and collapse the aspect ratio.
- **Verdict: this was incorrect behaviour, not a design choice.** Widening the
  bridge radius must not destroy structure a narrower radius had already
  resolved. The invariant that was *not* broken is the displayed Euler
  characteristic, which stayed at 2 for every k_bridge — only the *number* of
  glyphs was wrong.
- **Fix — make the gate decomposable** (`string_aggregator.py`): new
  subroutine `select_elongated_core()`. When a component fails the pooled
  gate, the member whose removal most restores the anisotropy is peeled off
  as a **satellite** and the remainder is retested, so the largest
  still-qualifying sub-corridor survives. A peel is only accepted when the
  remainder stays **connected** in the bridging graph, so a genuine corridor
  is never split into two halves that each look elongated. The peeled
  satellites return as standalone clusters, so their index is preserved and
  they remain eligible for standalone reduction.
- **Latent degeneracy found and fixed on the way**: a group of fewer than
  three points has a rank-deficient tangent-plane covariance, so its minor
  eigenvalue is zero and the aspect ratio is a division-by-zero artefact
  (observed: 7.1e7) rather than a measurement. Such a group trivially
  "qualified" as a corridor. `is_degenerate_group()` now marks it, and a
  degenerate group never qualifies on the pooled ratio.
- **New subroutines** (`string_aggregator.py`):
  - `compute_bridge_links()` — pure, per-pair verdicts with a human-readable
    `reason`; the aggregator now consumes its `is_linked` flags instead of
    re-implementing the rule inline.
  - `compute_group_geometry()` — pooled centroid, principal axis and aspect
    ratio of a set of clusters.
  - `select_elongated_core()` — the decomposable gate.
  - `is_degenerate_group()` — the <3-point shape guard.
- **Runner instrumentation** (`diagnose_params.py`): new `bridge` stage,
  placed between `clusters` and `units`, reporting the per-pair verdicts and
  the per-component gate outcome (pooled AR, whether it passed, the core and
  the satellites). The per-entity dump now carries `bridge_links` and
  `bridge_gate`. This is what makes the mechanism auditable:
  `--only bridge --out results.json` shows exactly which link widened the
  component and which member was peeled.
- **Result**: the response is now monotone. Displayed glyphs over
  k_bridge = 2.0 → 6.0: 12, 10, 10, 10, 10, 10, 9, 9, 8, 8 (non-increasing),
  with the displayed χ = 2 throughout. k_bridge = 3.25 now reproduces the
  k_bridge = 3.0 corridor instead of dissolving it. The saved snapshot for
  k_bridge = 3.0 still reproduces 35/35; the 3.25 snapshot now drifts by
  design, because it recorded the buggy behaviour.
- **Tests**: `tests/test_bridge_decomposition.py` (16 tests) — the link
  evaluator's verdicts and reasons, the group geometry, the core selection
  (peeling, index preservation, connectivity, blob rejection), and four
  regression tests on the real `ethene4.plt` data pinning the dilution, the
  peel, the monotone response and the index sum. Suite: 176 passed.
- **Open follow-up**: the pooled-AR threshold (2.0) is still hardcoded and has
  no viewer control, so the "Bridge Radius" slider is not the only thing that
  governs whether a corridor forms. Exposing it — or deriving it from δθ_mesh
  the way the other tolerances are — would make the corridor decision
  auditable from the UI alone.

### [Entry 026] "Load Parameters": Snapshot Restore, Camera Persistence & Runtime Reload
- **Objective**: Close the loop on the diagnostic snapshot that Entry 022
  introduced. Saving a snapshot was write-only: it could be read by the
  headless runner, but the viewer could not read it back. This entry adds
  the reverse direction -- a "LOAD PARAMETERS" button next to "SAVE
  PARAMETERS", a `--params` command-line flag, and the ability to restore
  the *file*, the *atom*, the *field*, the controls and the **camera**.
- **Why the camera matters**: the snapshot is meant to reproduce an
  observation. Reproducing an observation of a 3D scene means reproducing
  the *view*, not just the numbers: which way the camera looked and how far
  it was zoomed in are part of what the operator saw. Without them, a
  restored snapshot shows the right topology from an arbitrary viewpoint,
  which is not the same observation.
- **New subroutine** (`camera_state.py`) -- owns the serialisable form of
  the camera:
  - `capture_camera_state()` reads a `vtkCamera` into an immutable
    `CameraState` (position, focal point, view up, projection mode,
    parallel scale, view angle).
  - `camera_section_to_dict()` writes it to JSON *together with the nucleus
    it was captured around* (`atom_center`).
  - `camera_state_from_dict()` reads it back with strict validation: every
    field required, types checked, zoom and field-of-view range-checked,
    non-finite values rejected.
  - `apply_camera_state()` writes a snapshot back onto a camera.
  - `rebase_camera_state()` translates a snapshot from one nucleus to
    another, moving the position and the focal point while leaving the
    viewing direction, the roll and the zoom untouched.
- **Why the camera is stored relative to the nucleus**: the camera is
  captured in world coordinates, but the viewer draws in the world frame
  while the algorithms work in the atom's own frame. Recording the
  reference nucleus makes the camera *portable*: restoring a snapshot
  against a different atom (or a different file whose atom sits elsewhere
  in the molecule) moves the camera by the same offset as the nucleus, so
  it still looks **at** the atom instead of at empty space beside it.
  Without the reference point, a camera restored onto another atom would
  silently point at nothing.
- **New module** (`params_schema.py`) -- the snapshot format, extracted
  from `sphere_viewer.py` so the viewer, the headless runner
  (`diagnose_params.py`) and the tests share **one** definition instead of
  three copies. The `camera` section is **optional**: it was added without
  a schema-version bump because files written before it still load (the
  section is simply absent), and files with it are recognised by the
  presence of the key. The two sample snapshots already in the repository
  were written at schema 1 and still load unchanged.
- **New module** (`params_loader.py`) -- turns a loaded snapshot into an
  explicit, inert `RestorePlan`: which `.plt` to open, which atom to
  isolate, which field to select, the control values to push into the
  state, and the camera to apply. The plan resolves *what* to restore; the
  viewer decides *how*. That split keeps the interesting logic (path
  resolution, field fallback, camera re-basing) testable without a browser
  or a render window.
- **Runtime reload** (`sphere_viewer.py`): the mesh-bound state used to be
  a dozen closure variables captured at startup, so a different file could
  not be opened without restarting. Those variables are now grouped in a
  `ViewerSession` dataclass, and the pipeline reads from it. A reload
  replaces the session's contents in place, re-points the flood/contour/
  wireframe mappers at the new mesh, and rebuilds the pipeline. The
  re-entrancy latch (`suppress_param_change`) is reused so the batched
  state write rebuilds the pipeline exactly once, not once per key.
- **The LOAD button**: a toolbar button opens a dialog with a file input
  (`.json`). The file is picked in the browser and uploaded to the server,
  so a snapshot can be restored without the server knowing where it lives
  on disk. The dialog reports failures in an alert and in a drawer chip
  rather than raising into the trame event loop, and the loaded file name
  is shown in a confirmation chip.
- **`--params` CLI flag**: `uv run python gba_topology2/src/sphere_viewer.py
  <file>.plt --params <snapshot>.json` restores a snapshot at startup. The
  snapshot wins over the positional arguments, because it is the more
  specific description of the state to reproduce: it may name its own
  `.plt`, atom and field.
- **Camera capture source**: trame mirrors the browser's camera onto the
  server-side camera on every interaction, so the server's `vtkCamera` is a
  faithful mirror of what the operator is looking at. Capturing it is
  therefore a read of the authoritative camera, not a guess.
- **Tests** (50 new, 226 total, all passing):
  - `tests/test_camera_state.py` (18) -- capture, serialisation, strict
    validation, application, re-basing, and the property that actually
    matters: the restored camera still looks at the atom.
  - `tests/test_params_loader.py` (19) -- the camera in the JSON, the
    optional section, the plan's file/atom/field/control resolution, the
    field fallback, and the end-to-end round trip on real data.
  - `tests/test_sphere_viewer_params_load.py` (13) -- the session, the
    uploaded-file decoding, the CLI flag, and the UI wiring.
- **Open follow-up**: the dialog's file input uploads the snapshot to the
  server, but a snapshot that names a *different* `.plt` still requires
  that `.plt` to exist on the server's filesystem. Uploading the `.plt`
  itself is the natural next step, and would need the mesh extraction to
  accept an in-memory buffer rather than a path.




### [Entry 027] Pair Scale Distribution Diagnostic (`pair_scales.py`, `inspect_pair_scales.py`)

- **Goal**: replace guessed thresholds with measured ones. The pipeline stacks
  three length scales -- micro-clustering ($k_{\text{pitch}}$), bridging
  ($k_{\text{bridge}}$), and whatever macro-clustering range basins need -- and
  all three are currently set by eye. The question is whether the fields
  actually separate these scales, i.e. whether the distribution of pairwise
  separations between consolidated entities has a trough between the meso peak
  (fragments of one feature) and the macro peak (distinct basins). If it does,
  the bottom of the trough is the objective cutoff.
- **New modules**:
  - `gba_topology2/src/pair_scales.py` -- pure math, no viewer/VTK/Trame import.
    `PairwiseClearance` records, `compute_pairwise_clearances`, `histogram_bins`,
    `render_ascii_histogram`, `render_intermediate_registry`,
    `write_svg_histogram`. Consumes the `MicroCluster` interface, so
    `ReductionUnit.as_cluster` is measured exactly like a lone cluster.
  - `gba_topology2/inspect_pair_scales.py` -- CLI. Per field: Morse CPs ->
    micro-clusters -> bridged units -> **measure** -> ASCII histograms +
    intermediate registry (+ optional SVG).
- **Design decisions**:
  - **Measure before catastrophe reduction.** Catastrophe classification is an
    interpretation step; measuring after it would report the shape the
    constellation templates imposed, not the shape of the field. It also keeps
    the diagnostic aligned with `classify_catastrophes`, which itself consumes
    raw clusters.
  - **Two measures, not one.** Centroid separation is dominated by the
    half-lengths of elongated entities, so tip-to-tip fragments look far apart
    and crossing corridors look close. The directional edge clearance
    $\theta_{ij} - r_i(\phi_i) - r_j(\phi_j)$, reusing
    `cluster_directional_support_radius` from Entry 021, puts touching entities
    at zero at any orientation and sharpens the trough.
  - **Normalise by $\Delta\theta_{\text{mesh}}$.** Makes Pd ($R = 1.04$ A) and
    ethene C ($R = 0.41$ A) share one axis, so fields and datasets plot together.
  - **Zero-dependency SVG.** matplotlib is not a project dependency; the chart
    is emitted as standalone XML so it opens in VS Code beside the viewer.
- **Result -- intermediate pairs in the 2.5-6.0 dtheta transition zone**:

  | field | ethene3 C1 | Pd_20K Pd1 |
  |---|---|---|
  | Electron Density | **empty** | **empty** |
  | V (volume) | 2 | 56 |
  | shape index | **empty** | 72 |
  | curvedness | 8 | 4 |
  | Willmore energy | **empty** | 176 |
  | modified Willmore | 13 | 19 |

- **Findings**:
  1. **Density is the cleanest field on both datasets** -- zero intermediate
     pairs, so its entities are strictly near or strictly far. If basin
     identification wants a hard-coded cutoff, density is where it is safest.
  2. **The transition zone is field-specific, not a constant.** Willmore on Pd
     has 176 pairs there; curvedness on the same atom has 4. A single global
     macro-clustering radius cannot be correct for all fields, which argues for
     per-field (or per-field-class) parameters rather than one slider.
  3. **The ethene3 curvedness registry is entirely compensating dipoles** -- all
     8 pairs have $\chi_{\text{sum}} = 0$, and they fall into four
     symmetry-related pairs of two (MC7 and MC8 each facing MC22 at 3.61
     dtheta; MC10 and MC11 each facing MC20 at 4.44). These are the pairs a
     macro rule must decide on, and their index balance says merging them would
     annihilate.
  4. **The directional measure beats the centroid measure as designed.** On
     ethene3 curvedness the centroid distribution is empty across the whole
     2.5-5.5 band while the clearance distribution shows structure at 3.5-6.0 --
     the structure bridging already acts on. The centroid view would have
     reported "clean separation" and hidden the real decisions.
- **Rendering lesson**: scaling bars to the global peak made the charts
  unreadable. On a sphere most pairs are distant, so the overflow bin (164 of
  190 on ethene3 curvedness) dwarfs everything and flattens the meso trough --
  the one feature the chart exists to show. Bars now scale to the peak of the
  **finite** bins, with the overflow bin clamped and labelled off-scale.
- **Tooling lesson**: the volume field is stored as plain `V`, so the natural
  pattern `volume` matches nothing and the short pattern `v` matches
  "cur**v**edness" and "distance". `_field_matches` treats single-letter
  patterns as exact base names (name minus `(condensed)` and non-ASCII prefix)
  and longer patterns as substrings.
- **Tests**: `tests/test_pair_scales.py` -- 26 tests. Collinear pair clearance
  equals the analytic tip gap (0.015 rad); perpendicular pair measures the
  transverse width (0.32 rad) where the isotropic measure reports 0.03;
  crossing corridors report negative clearance; $\binom{N}{2}$ pair count with
  no self-pairs; bin conservation including the negative underflow bin; SVG
  parses as XML and escapes non-ASCII field names. Two tests failed while the
  code was wrong (empty-meso marker, overflow scaling), confirming both are
  load-bearing. Suite: 141 passed.
- **Open next**: (a) run the diagnostic across the remaining ethene variants and
  more atoms to see whether the per-field ordering in the table is stable;
  (b) if it is, derive per-field macro-clustering cutoffs from the measured
  troughs; (c) the density result suggests starting macro-clustering experiments
  on density, where the separation is cleanest.

---

### [Entry 028] Viewer Sidebar Card Background & Contrast Harmonization (`sphere_viewer.py`)

- **Objective**: Standardize drawer card styling and contrast across all steps in `sphere_viewer.py` to ensure slider labels, ticks, counts, badges, and card contents remain legible against dark mode backgrounds.
- **Changes**:
  - `gba_topology2/src/sphere_viewer.py`:
    - Updated scoped `<style>` block in `SinglePageWithDrawerLayout.drawer`:
      - Set `.v-navigation-drawer .v-card` background to `#3e434c` with foreground `#f0f2f5`.
      - Adjusted text medium emphasis color to `#d8dce2` for crisper readability.
      - Styled sub-surfaces (`.surface`, `.v-table`) to `#30343b` and table headers to `#272a30`.
      - Increased contrast on slider tracks (`#202328`, opacity 0.85), ticks (`#ffffff`, opacity 0.8), and slider typography (`#ffffff`, opacity 0.95) including `.v-slider-thumb__label`.
    - Standardized Step card containers (`Field Controls`, `Step 2: Micro-Clusters`) to use `color="surface-variant"` uniformly with the rest of the step cards.
- **Validation**:
  - `uv run pytest tests/test_sphere_viewer_*.py`: 43 passed.
  - `uv run ruff check gba_topology2/src/sphere_viewer.py tests/`: All checks passed.
  - `uv run ruff format --check gba_topology2/src/sphere_viewer.py tests/`: 23 files formatted.
  - `uv run mypy gba_topology2/src/sphere_viewer.py`: Success: no issues found.
  - Headless UI smoke test compiling and initializing viewer layout: passed cleanly.
  - Full test suite (`uv run pytest`): 253 passed.

---


### [Entry 029] High-Contrast Sidebar: Real Vuetify Theme Replaces the Force-Dark CSS (`sphere_viewer.py`)

- **Objective**: Kill the dark-grey-on-light-grey text in the sphere viewer sidebar. Target: WCAG AA (4.5:1) for all body text and chip labels.
- **Root cause -- two dead mechanisms stacked on top of each other**:
  1. **`state.trame__theme = "dark"` was a no-op.** The string `trame__theme` appears **zero times** in `trame_server`, `trame_client`, `trame_vuetify`, the built `vuetify3.js`, or the trame `index-*.js` bundle. The app has always rendered in Vuetify's **default light theme** (`defaultTheme:"light"`).
  2. **`html.Style` inside a Vue template never reaches the document.** Vue strips `<style>` from component templates. Verified two ways: a minimal Vue 3 repro (template containing `<style>` -> `document.querySelectorAll('style').length === 0`, rule not applied, no warning), and the live app (the drawer's rules were absent from `document.styleSheets`). So the Entry 028 block that set `#3e434c` cards and `#f0f2f5` text **has never executed**.
  - The real grey-on-grey was therefore the **stock light theme**. `color="surface-variant"` in the light theme is a `#424242` panel with `#EEEEEE` text -- a dark-theme pair pasted into a light one -- and Vuetify paints emphasis text as `on-background` (black) at the stock **0.6** opacity. Black at 60% over `#424242` composites to `#1A1A1A`: **1.73:1**. Measured on the card:
    - `text-medium-emphasis` (the ~50 field labels): **1.73:1**
    - `text-high-emphasis` / card body text: 1.98:1
    - `.text-teal` `#009688`: 2.74:1, `.text-grey` `#9e9e9e`: 3.75:1, `.text-cyan` `#00bcd4`: 4.38:1
    - trame's own footer links `.text-grey-lighten-1` `#bdbdbd` on the white footer: 1.88:1
- **The mechanism that does work**: `SinglePageWithDrawerLayout(server, vuetify_config=...)` -> `server.state.trame__vuetify3_config` -> `createVuetify(config)`. Vuetify generates `.text-{color}` / `.bg-{color}` rules from `themes.light.colors` into `#vuetify-theme-stylesheet`, injected into `<head>` **after** the static CSS and with `!important` at equal specificity, so **a palette entry beats the static Material rule**. Naming the accents in the palette fixes all 22 drawer tokens at once -- no per-widget edits.
- **Changes** (`gba_topology2/src/sphere_viewer.py`):
  - Added `VUE3_THEME_CONFIG`: black-on-white neutrals, semantic accents, the Material accents the cards name, explicit `-darken-N` entries, and `medium-emphasis-opacity` raised 0.6 -> 0.80.
  - Wired it via `vuetify_config=`; deleted the dead `trame__theme` assignment and the dead `html.Style` block (48 lines).
  - Card borders come from `defaults: {VCard: {border: True}}` -- set centrally, colored by the theme's `border-color` / `border-opacity` variables.
  - Added `grey-lighten-1` to the palette: **trame's own layout footer** hardcodes `text-grey-lighten-1` on the "Powered by trame" links, which is 1.88:1 on a white footer.
- **Findings**:
  1. **`VChip` defaults to `variant="tonal"`, not filled.** The label is the accent and the underlay is `currentColor` at `--v-activated-opacity` (0.12), so an accent must clear AA **against its own 12% tint** as well as against the bare surface. Choosing chip colors for a solid fill with a white label is the wrong model -- that is what made the first palette draft pass on paper and fail in the browser.
  2. **Vuetify derives `on-{color}` by perceptual (CIE) distance, not luminance** (`dt()` in the bundle). A mid-tone like `#C77800` is dark enough for black text by luminance but gets a **white** label, landing at 3.43:1.
  3. **Sliders are self-tinting**: fill and thumb are the accent at full strength, the track is the same accent at 0.38 over the surface. Fill-vs-track is 3.56-4.16:1 (passes the 3:1 non-text floor); the track-vs-surface groove is ~1.9:1 **by Material design** and is not a WCAG target -- the value is carried by the thumb, not the groove.
  4. **`surface-variant` is load-bearing beyond cards**: Vuetify also paints the slider track and ticks from `--v-theme-surface-variant`, so setting it near-white for cards has effects outside the card.
- **Validation** (Chrome 155, real `vuetify3.js` + `vue.global.js` from the installed packages):
  - Isolated harness mounting the real Vuetify with `VUE3_THEME_CONFIG`: 40/40 elements >= 4.5:1.
  - **Live app** (`ethene4.plt`, atom C1) full-page sweep of every visible text node, with the WCAG large-text exemption applied: **175 elements checked, 0 failures**. Dialog sweep: 6 checked, 0 failures.
  - Toolbar tonal buttons measured 7.28-8.13:1; table text 16.1:1; emphasis labels 11.84:1; card titles 15.8:1.
  - `uv run pytest tests/test_sphere_viewer_theme.py`: 63 passed. Mutation-checked the guards: reverting a chip fill to `#C77800`, adding an unregistered `color="lime-darken-2"`, and restoring `state.trame__theme` each fail the suite.
  - `uv run ruff check` / `ruff format --check`: clean. `uv run mypy gba_topology2 tests`: no issues in 48 files. Full suite: **316 passed**.
- **Deliberately out of scope**: the 3D viewport keeps its dark renderer background and light-on-dark actor palette (white contours, yellow highlights, gold basin centers read 8-16:1 on dark and would vanish on white). A white sidebar beside a dark viewport is the normal ParaView arrangement.
- **Open next**: (a) the same `color="surface-variant"` + light-theme trap exists in `BondalyzerParaView/trame_viewer.py` and `gba_topology/stage0_sphere_view.py`, which have no theme config at all; (b) any future `html.Style` in a trame layout is dead code -- if real CSS is ever needed it has to be a served stylesheet registered through `server.enable_module({"styles": [...], "serve": {...}})` (the client reads `trame__styles`), never a widget inside a template.

---

### [Entry 030] Local IDW Interpolation via a Kd-Tree Locator (`excision_boundary.py`, `point_locator.py`)

- **Objective**: Make per-ring field sampling cheap enough to profile an excision boundary for **every displayed saddle**, not just the ~8 catastrophe constellations. This is the enabling unit for the excision-based Launch Seeds rewrite (plan: `Excision-Based Launch Seeds`).
- **Why it is needed**: `interpolate_scalar_at_points` builds a full `(M, N)` `arccos` matrix -- one angular distance per query point per mesh vertex. At `Pd_20K` scale (20174 nodes) that is ~1e6 transcendental evaluations per 48-sample ring. Profiling ~200 rings, each at up to 12 expansion radii, costs seconds inside `update_topology()`, which already rebuilds Steps 1-9 on every toggle.
- **Changes**:
  - `gba_topology2/src/point_locator.py` (new): `KdTreeNearestFinder` wraps `vtkKdTreePointLocator` (in `vtkCommonDataModel`, **not** `vtkFiltersCore`) behind a callable, built once per mesh and reused for every query. `make_kdtree_nearest_finder(poly)` is the factory.
  - `excision_boundary.py`: added the `KNearestFinder` type alias and `interpolate_scalar_at_points_local`, which evaluates the **same** angular IDW kernel on the `k` candidates the index returns instead of on all N vertices.
  - `sphere_viewer.py`: `ViewerSession` carries a `locator` field, built in `build_viewer_session` over `sphere_poly` (the centred frame every algorithm already works in). The `apply_restore_plan` in-place mesh swap now also swaps `locator` -- a stale tree indexes a discarded sphere and returns neighbours of the **old** mesh, which is a silent correctness bug, not a crash.
- **Design decision -- the seam is a callable, not a vtk object**: `excision_boundary` advertises a pure NumPy/stdlib contract and must stay testable without a VTK build. Injecting `find_k_nearest` keeps that contract intact and lets tests pass a brute-force lambda. The adapter therefore lives in its own module.
- **Design decision -- k-nearest, not radius search**: the first draft took a chord search radius with widen-then-global fallback. It is unnecessary: mesh vertices and ring samples both lie on the sphere of radius R, so chord distance `2 R sin(theta / 2)` is a strictly monotone function of the angular distance the kernel uses, which makes the k nearest by chord **provably** the k nearest by angle. Verified empirically over 200 random queries (zero set mismatches) and pinned as `test_chord_k_nearest_equals_angular_k_nearest_on_sphere`. Dropping the radius removed the widening loop, the fallback path, and ~2x of the runtime.
- **Pre-existing bug fixed**: `interpolate_scalar_at_points` raised `ValueError: kth(=50) out of bounds (12)` from `np.argpartition` whenever `k_neighbors > N`. The local variant clamps k to the mesh size, so the two would have diverged on small meshes; the reference kernel now clamps too, preserving the drop-in parity contract.
- **Performance**: the per-query Python loop over 4-element arrays dominated, so the kernel is vectorised over the whole `(M, k)` candidate block (padded with `+inf` distances so short rows drop out of the weighting). Measured on a 20174-node sphere, 200 rings x 48 samples: **6.01s global -> 0.205s local (29.3x)**, max deviation **7.4e-15**. One ring at 12 expansion radii: **21ms**.
- **Validation**:
  - `tests/test_point_locator.py` (new, 14 tests): parity vs the global kernel on ring samples, parity through the real VTK locator, the chord/angle order-equivalence proof, exact-vertex snap, k-clamping above mesh size, zero-norm query, non-positive k rejection, single-query `(3,)` widening, locator-vs-brute-force set equality, closest-first ordering, empty-dataset rejection, and a runtime guard that the rewrite stays faster than the scan it replaces.
  - `uv run pytest`: **330 passed** (316 baseline + 14 new).
  - `uv run ruff format` / `ruff check`: clean. `uv run mypy gba_topology2/ tests/`: no issues in 50 files.
- **Environment note**: `uv run mypy .` is broken at baseline -- `BondalyzerParaView/plt_gba_to_vtm.py` is discovered twice under different module names, and those 2 errors abort the run before checking the rest. Confirmed pre-existing by stashing. Use `uv run mypy gba_topology2/ tests/` for this package.
- **Open next**: parabolic port azimuth refinement (unit 2), then the pure `excision_launch.py` subroutine that consumes this locator.

---

### [Entry 031] Sub-Sample Port Azimuth Refinement (`excision_boundary.py`, `micro_cluster.py`)

- **Objective**: Remove the azimuth quantization that makes a boundary port less angularly precise than the mesh it is derived from, so a port is a trustworthy launch point for separatrix integration.
- **The problem, measured**: ring samples sit at a fixed pitch of `2*pi/48 = 7.5 deg`, and `extract_boundary_ports` snaps every port to a sample index. That is a worst-case error of **3.75 deg** -- **4.7x coarser than the 1.606 deg mesh pitch** the port is meant to resolve. The cyclic-link bisector it is replacing is *more* precise than this, so shipping the ring substrate unrefined would have been a regression in launch direction.
- **Changes**:
  - `extract_boundary_ports` / `profile_boundary_loop` take `refine_azimuth: bool = False`. When on, each accepted extremum is located between samples by fitting a parabola to the three smoothed points (`_refine_extremum`) and its position is re-evaluated **on the loop** at the refined azimuth, via the ring's own frame.
  - `micro_cluster.ring_frame` (new, public): the analytic inverse of `generate_boundary_ring`, recovering `(c_unit, t1, t2, angular_radius)` from ring samples. This was the private `_ring_frame` inside `harmonic_nudge`; it is moved to `micro_cluster` because (a) it belongs beside the generator it inverts, and (b) `excision_boundary` cannot import from `harmonic_nudge` -- that is a cycle, since `harmonic_nudge` already imports from `excision_boundary`. Pure relocation, no behaviour change: the 330-test suite passed unchanged.
  - The four duplicated `BoundaryPort(...)` construction blocks in `extract_boundary_ports` are consolidated into one `_port` builder.
- **Design decisions**:
  - **The prominence test stays on the discrete profile.** Refinement moves an accepted port between its bracketing samples; it must never change *which* ports are accepted. Pinning this as `test_refinement_preserves_port_counts_and_types` for k = 2, 3, 4.
  - **The frame is recovered once per call and reused**, not re-derived per port.
  - **`scalar_value` is not refined.** This routine sees only the profile, so a refined azimuth does not imply a re-interpolated scalar; callers that need it re-interpolate (as `reposition_ports_on_ring` already does).
  - **Clamped to one sample step**, with a flat-stencil guard (`den ~ 0`) and non-finite guards, so a degenerate stencil keeps the sample.
- **Empirical validation of the accuracy claim** (this is the part worth keeping):
  - **Isolated, analytic** `cos(3(phi - phase))`, 48 samples, no mesh: mean max-error **3.750 deg -> 0.000-0.019 deg** across phases. The parabola is near-exact on a smooth profile, as expected.
  - **Real 20174-node mesh, quadrupole, phase-swept** through 277 non-grid-aligned orientations (the fair test -- a sweep step equal to the 7.5 deg sample pitch makes every trial grid-aligned and shows snapped as *perfect*, which is an artifact, not a result):

    | ring radius | snapped | refined | gain |
    |---|---|---|---|
    | 1.606 deg (1 dtheta) | 9.314 | 8.618 | 1.1x |
    | 3.20 deg | 5.640 | 5.066 | 1.1x |
    | 8.00 deg | 2.619 | 1.363 | 1.9x |
    | 20.0 deg | 1.941 | 0.493 | 3.9x |

  - **Refinement wins at every radius and every synthetic noise level tested** (0-10% of span). It never loses.
- **The finding that matters for the expansion loop**: at a tight ring the gain is only 1.1x because **IDW interpolation noise dominates the quantization it removes**. Measured against the analytic field on a 20174-node sphere, IDW error is **7.3% of the field span at r = 1.6 deg**, and the high-frequency content of the *interpolated* profile is **3.6x that of the true profile** (1.38e-4 vs 3.79e-5) -- the interpolant is noisier than the signal it samples. At r = 20 deg the IDW error is 0.5% of span.
  - This is independent evidence for the ring-expansion design: a ring at ~1 dtheta is inside the interpolation-noise regime, and expanding it buys real accuracy, not just topological cleanliness. It also bounds what refinement can do -- the residual at tight radii is an **interpolation** limit, not a sampling limit, so raising `ring_samples` will not fix it.
  - Checked and rejected the obvious alternative: **barycentric interpolation on the mesh triangles is worse**, not better (32% of span vs 12.6% at r = 1.6 deg on a structured test mesh, and its profile noise tracks the triangle crossing pattern). The mesh triangles are long and thin near the poles, so linear interpolation across them is a poor fit for a smooth field.
- **Validation**:
  - `tests/test_excision_boundary.py`: 9 new tests (15 total) -- sub-degree phase recovery on an analytic field, refined strictly beats snapped, port counts/types/ordering preserved, refined positions lie on the small circle, reported azimuth consistent with reported position in the ring's own frame, default-off is byte-identical, flat-stencil and clamp guards, and the flag threaded through `profile_boundary_loop` end-to-end on a real mesh.
  - `uv run pytest`: **339 passed** (330 + 9).
  - `uv run ruff format` / `ruff check`: clean. `uv run mypy gba_topology2/ tests/`: no issues in 50 files.
- **Open next**: `excision_launch.py` (unit 3), the pure ring-to-seed subroutine, which will call `profile_boundary_loop(..., refine_azimuth=True)`.

---

### [Entry 032] Excision-Based Launch Seeds: the Ring-to-Seed Subroutine (`excision_launch.py`)

- **Objective**: Produce separatrix launch branches from a saddle's **excision boundary**, so seeds can be computed for effective critical points, which have no mesh vertex to scan.
- **Why the cyclic link cannot do this**: `saddle_launch.extract_saddle_launch_seed` needs a `vertex_id` and walks the incident-triangle link graph. An effective CP's position is a mean of member directions, optionally nudged -- `effective_cp_to_ui` hardcodes `"vertex_id": -1`. No link, nothing to scan. Step 9 iterated `morse_res.saddles` and silently omitted every `E-SAD`, contradicting its own comment. The excision loop is defined for **any** point on the sphere, which is the property that makes it the substrate for the reduction stage.
- **Changes**:
  - `gba_topology2/src/excision_launch.py` (new, pure -- no VTK, no viewer): `extract_excision_launch_seed(ring_points, mesh_pts, f_vals, saddle_id, saddle_position, saddle_value, expected_k, ...)` -> the existing `SaddleLaunchSeed` / `LaunchBranch` dataclasses, so `_emit_branch`, the tube filters, the actors and the visibility gate in the viewer all survive unchanged.
  - `SaddleLaunchSeed` gains two defaulted fields: `ring_radius_rad` (the accepted radius, for expansion bookkeeping) and `failure_reason` (why a ring produced nothing). Both zero/`None` for the link path, so existing construction sites are untouched.
  - `profile_boundary_loop` gains `find_k_nearest`, threading the kd-tree locator from Entry 030 into the profiling entry point.
- **Mapping from link scan to ring** (the semantics carry over cleanly):
  - link sector with `f(u) > f(v)` -> **ridge port** (local max of `f|_loop`) -> `kind="ascent"`, unstable manifold.
  - link sector with `f(u) < f(v)` -> **valley port** -> `kind="descent"`, stable manifold.
  - link sign crossing -> profile crossing of `saddle_value`, by chord interpolation (`_crossing_azimuths`), giving the 2k separatrix directions.
  - link sector bisector -> the **port itself**, refined to sub-sample azimuth (Entry 031).
  - `fold_order = len(ascent)` -- unchanged formula, now measured on the loop.
- **Launch direction**: the separatrix runs centre -> port -> outward, so the tangent is the geodesic direction at the port pointing away from the ring centre: `t = normalize((p.c) p - c)`. It is tangent to the sphere by construction (`t.p = 0`) and outward (`t.t1 = +cos(a) > 0`, and moving along it decreases `p.c`, i.e. increases angular distance from the centre).
- **Bug caught by the tests**: the first draft used `c - (p.c) p`, which is the **inward** tangent -- the direction back toward the degenerate core. `test_tangent_points_away_from_the_ring_centre` failed on it immediately. The sign matters: launched inward, every Step 3 path would run into the singularity it is meant to escape.
- **Failure taxonomy** (`FAILURE_REASONS`, the routing keys for the expansion loop):
  | Reason | Meaning |
  |---|---|
  | `empty_ring` / `too_few_samples` | no loop to profile |
  | `no_ports` | structureless profile (e.g. a constant field) |
  | `unbalanced` | valleys != ridges -- still inside the degenerate core, expand |
  | `below_min_k` | balanced 1 + 1 -- a ring around an **extremum**, not a saddle |
- **Design decision -- this subroutine measures, it does not judge**: `index_agrees = (fold_order == expected_k)` is reported, never enforced. Rejecting a ring whose `k` disagrees is the expansion loop's job (unit 4). Keeping the two apart preserves the ring as an **independent** check on the reduction's `chi = 1 - k`; if the loop searched for `expected_k` and this function coerced to it, the audit would be circular.
- **The extremum case, and why it is not fully handled here**: a ring around a maximum has a nearly flat profile, so IDW noise alone produces many balanced "ports" -- measured at `k = 17` on a synthetic maximum. The `k >= 2` gate catches the 1 + 1 case, and a noise-generated 17 is caught by `index_agrees = False`. Both are the expansion loop's `k == expected_k` acceptance test rejecting the ring, which is exactly the division of labour above. Pinned as `test_extremum_ring_is_not_accepted_as_a_saddle`.
- **Validation**:
  - `tests/test_excision_launch.py` (new, 23 tests): k = 2 / 3 / 4 branch counts and crossing counts (4 / 6 / 8), ideal 360/k azimuth spacing, ascent/descent interleaving (6 cyclic sign flips for k = 3), launch positions on the sphere **and** on the loop's small circle, tangents unit-length and perpendicular to the radial, tangents pointing outward, dense counter-clockwise `sector_index`, reported ring radius, index agreement and mismatch (measured `k` never coerced), all five failure reasons, `unbalanced` and `below_min_k` produced by real tilted fields, off-pole centres (frame recovered from the ring, not assumed), local-vs-global sampling parity to 1e-12, and identity-field passthrough.
  - `uv run pytest`: **362 passed** (339 + 23).
  - `uv run ruff format` / `ruff check`: clean. `uv run mypy gba_topology2/ tests/`: no issues in 52 files.
- **Open next**: the ring-expansion loop (unit 4) that routes on `FAILURE_REASONS` and the `k == expected_k` audit, then the Step 9 rewire.

---

### [Entry 033] Self-Calibrating Excision Ring Search (`excision_launch_seeds.py`)

- **Objective**: Choose the excision radius for each saddle, so a ring that starts inside the degenerate core expands until its port signature is pure.
- **Why the radius must be searched**: the excision-radius rule puts a lone saddle's ring at `max(0 + 1.5deg, k_pitch*dtheta/2)` = **exactly 1 dtheta** on `Pd_20K`. Entry 031 measured that regime: IDW error is 7.3% of field span there, and the interpolated profile carries 3.6x the high-frequency content of the true one. A ring that tight has no clean alternating signature, and the tell is topological -- valley and ridge counts disagree. So **balance is used as a convergence test**.
- **Changes**: `gba_topology2/src/excision_launch_seeds.py` (new, pure). `expand_excision_seed(SeedRequest, ...) -> ExpansionOutcome` walks radii from `a0` in `expand_pitch * dtheta` steps to `max_radius_pitch * dtheta`, accepting the first ring that is balanced, has `k == expected_k`, and has `k >= 2`. `summarize_outcomes` aggregates into the viewer's `launch_stats`.
- **Separation of concerns**: this module owns the *policy* (which radius to accept); `excision_launch` owns the *measurement* (what the ring says). Kept apart so the ring stays an independent check on the reduction's `chi = 1 - k`.
- **The finding that changed the design -- the overshoot early-exit needed a grace radius**: the plan specified stopping as soon as a balanced ring reported `k > expected_k`, on the assumption that port count grows monotonically with radius. **That assumption fails exactly where the ring is tightest, which is where the search starts.** Measured on a k = 3 monkey saddle:

  | a0 | with immediate exit | with sustained exit |
  |---|---|---|
  | 0.4 deg | expanded, k=3 | expanded, k=3 |
  | 0.8 deg | **k_overshoot, FAILED** | expanded, k=3 |
  | 1.2 deg | **k_overshoot, FAILED** | expanded, k=3 |
  | 2.0-20.0 deg | converged, k=3 | converged, k=3 |

  Interpolation noise inflates the port count at a tight ring, so a single radius reads k = 4 for a k = 3 saddle and the search terminates on a guaranteed-successful path. Requiring the overshoot at **two consecutive radii** fixes it: all starting radii now recover k = 3. A *sustained* overshoot is still a real signal, so the cost saving survives. Mutation-checked: reverting to the immediate exit fails 6 of the 18 tests.
- **Failure taxonomy** (`ExpansionOutcome.status`), and the diagnosis each one implies:
  | Status | Meaning |
  |---|---|
  | `converged_at_a0` | the radius rule was right |
  | `expanded` | the rule was too tight -- tune `min_boundary_margin_deg` |
  | `unbalanced` | never balanced at any radius -- degenerate core |
  | `k_mismatch` | balanced at the wrong k -- **the reduction's index is suspect** |
  | `k_overshoot` | k passed the target and stayed past it -- ceiling too loose |
  | `no_ring` | `expected_k < 2`, nothing to search for |
  The `k_mismatch` / `unbalanced` split is what preserves the audit that accepting on `target_k` would otherwise destroy: `observed_k` is recorded at the **tightest balanced radius**, so "the field disagrees with the reported chi" is distinguishable from "the ring never resolved". Mutation-checked: collapsing the two into one bucket fails 3 tests.
- **Ceiling semantics**: the ceiling bounds **expansion**, not `a0`. A constellation's ring is `span/2 + dtheta` and can legitimately exceed 12 dtheta, so `a0` is always profiled once. Pinning this explicitly, because "the radius never exceeds the ceiling" is false and a future reader will assume it.
- **Validation**:
  - `tests/test_excision_launch_seeds.py` (new, 18 tests): convergence at a0 for k = 2/3/4, expansion from a too-tight ring, all seven starting radii recovering the same k, `k_mismatch` and `unbalanced` diagnoses, ceiling bounds, `expand_pitch` controlling step count, `expected_k < 2` short-circuit, the caller's ring reused unmodified at a0, local-vs-global sampling parity to 1e-12, `num_samples` honoured, and the aggregate stats (converged/expanded/failed counts, failing entity ids with radius, empty-input well-formedness).
  - `uv run pytest`: **380 passed** (362 + 18).
  - `uv run ruff format` / `ruff check`: clean. `uv run mypy gba_topology2/ tests/`: no issues in 54 files.
- **Open next**: port colours (unit 5), the parameter schema (unit 6), then the Step 9 rewire that consumes all of this.
