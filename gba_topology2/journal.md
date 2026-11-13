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

---

### [Entry 034] Excision Ring Controls in the Parameter Schema (`params_schema.py`)

- **Objective**: Expose the excision ring geometry as live, savable controls, so the ring substrate can be tuned interactively and a tuned configuration can be captured in a snapshot.
- **Changes**:
  - `DEFAULT_UI_STATE` gains `ring_samples` (48), `ring_expand_pitch` (1.0), `ring_max_pitch` (12.0).
  - All three added to the viewer's `@state.change` handler so they trigger a rebuild.
  - `PARAMS_SCHEMA_VERSION` **1 -> 2**.
  - The two committed sample snapshots migrated to v2.
- **Why adding keys is breaking, not additive**: `load_params_from_dict` requires **every** control key to be present and raises `KeyError` naming the first missing one. A version 1 file lacks all three new keys, so without the bump it fails as a confusing `Missing parameter 'ring_samples' in section 'floats'` -- it looks like a corrupt file, not an old one. Bumping converts that into the clear `Unsupported params schema_version 1` ValueError. The version gate exists precisely to make stale files reject legibly.
- **The type tables need no edits**: `PARAM_BOOL_KEYS` / `PARAM_FLOAT_KEYS` are derived by `isinstance` from `DEFAULT_UI_STATE`, and the import-time coverage check (`PARAM_*_KEYS` must partition `PARAM_CONTROL_KEYS`) validates the derivation. Adding a key to the defaults is sufficient; the check is the safety net.
- **`ring_samples` is int-valued, exported as float**: it follows the `num_contours: 15` precedent -- `_coerce_float` widens every numeric control, so the round-trip yields `48.0` and consumers must cast with `int()`. Pinned by `test_ring_samples_loads_as_float_and_must_be_cast` so the cast is not later mistaken for a bug and removed.
- **Fixture migration**: the two `ethene4_atom1_params *.json` files are real viewer output used by `test_a_saved_file_without_a_camera_still_resolves`, whose point is that an **absent camera section** still loads -- not that v1 loads. Migrated in place: `schema_version` 1 -> 2, the three new floats added at their defaults, camera left absent so the test keeps covering what it was written to cover. Written with `ensure_ascii=True` and `sort_keys=True` to match the exporter's own output; the first attempt used `ensure_ascii=False` and rewrote every non-ASCII escape as raw UTF-8, a 22-line diff where 5 was correct.
- **`launch_seed_pitch` deliberately retained** in this unit. It is still read by Step 9 and by `diagnose_params.stage_launch`; removing it here would break the viewer for one commit. It is retired in the Step 9 rewire, which is the unit that stops needing it.
- **Validation**:
  - `tests/test_sphere_viewer_params_export.py`: `test_load_rejects_unknown_schema_version` extended to include version **1** in the rejected set -- the real-world stale case, previously untested. Two new tests: the ring controls round-trip with overridden values, and the int-as-float coercion contract.
  - `uv run pytest`: **390 passed** (388 + 2). Params suites specifically: 76 passed, and the fixture test verified to PASS rather than skip.
  - `uv run ruff format` / `ruff check`: clean. `uv run mypy gba_topology2/ tests/`: no issues in 55 files.
- **Open next**: the Step 9 rewire (unit 7), which consumes `ring_samples` / `ring_expand_pitch` / `ring_max_pitch` and retires `launch_seed_pitch`.

---

### [Entry 035] Step 9 Rewired to Excision-Boundary Seeds (`sphere_viewer.py`, `diagnose_params.py`)

- **Objective**: Seed every **displayed** saddle from its excision boundary, so effective saddles -- which the cyclic-link scan structurally cannot serve -- get launch seeds.
- **Changes**:
  - `sphere_viewer.build_launch_requests` (new, module-level, pure): the seed-set rule. Raw saddles not in `hidden_vertex_ids`, plus every `E-SAD` from `effective_cps_ui`. Extracted from the `update_topology` closure so the rule is unit-testable without a viewer, per AGENTS.md Principle 1.
  - Step 9 now builds `SeedRequest`s, runs `expand_excision_seed` over each, and takes `state.launch_stats` straight from `summarize_outcomes`. `compute_all_saddle_launch_seeds` is no longer imported by the viewer.
  - `eff_ring_radius` records each effective CP's initial radius (constellation `angular_radius_rad`, or the unit's cluster radius), keyed by `id_label`.
  - `diagnose_params`: `RunContext` gains `locator` and `eff_ring_radius`; `stage_launch` uses the same rule, so the headless harness and the viewer cannot drift.
- **Effective saddles are centred correctly, including under the nudge**: the ring is generated fresh at `eff_dict["position"]`, which is the **nudged** centre, rather than reusing the constellation's stored ring, which is centred on the pre-nudge centroid. UI positions are world frame, so they are mapped back with `to_local` before the search.
- **The finding that changed the failure taxonomy**: the first run on `Pd_20K` reported 36 `k_overshoot` failures. Diagnosing one (`SAD_5`, multiplicity 1, so expected k = 2) showed the ring reading **k = 3 stably from 3.2 deg all the way to 19.3 deg** -- not interpolation noise, and not a loose ceiling. That is a genuine disagreement with the index the Morse detector reported, which is exactly what `k_mismatch` means.
  - So `k_overshoot` was **removed as a status**. A *sustained* overshoot is an index mismatch by definition; the two classes were the same event wearing different names. The search now terminates when **two consecutive radii agree** on the same inflated `k`, and returns `k_mismatch`.
  - The two-consecutive-radii rule is still required, for the reason found in Entry 033: a *single* inflated reading at a tight ring is noise, and a k = 3 monkey saddle read at 0.8 deg recovers to k = 3 one step later. Verified: the stable path exits at `radii_tried = 2` of ~11 available, so the cost saving survives.
  - Reclassifying also **raised the yield**: 164 -> 185 of 200 saddles seeded, because radii whose readings differ now continue expanding instead of terminating.
- **`Pd_20K` (`V (condensed)`, 20174 nodes, dtheta = 1.606 deg)**:

  | toggle set | reqs | seeded | conv@a0 | expanded | k_mismatch | mean radii | time |
  |---|---|---|---|---|---|---|---|
  | none | 200 raw | 185 | 118 | 67 | 15 | 1.73 | 0.84s |
  | reduce_catastrophes | 194 raw + 2 E-SAD | 181 | 114 | 67 | 15 | 1.74 | 0.85s |
  | cats + clusters | 18 raw + 2 E-SAD | 20 | 15 | 5 | 0 | 1.25 | 0.06s |

  - **The 2 E-SADs converge at a0 with k = 3, 3 ascent + 3 descent, and `vertex_id = -1`** -- the case the rewrite exists for, and one the link scan silently dropped.
  - 8 catastrophes detected, 2 fused; the 6 split ones are physical symmetry breaking and correctly stay as raw saddles.
  - Cost is **0.84s for 200 rings** with the locator, against the 6.0s the global kernel alone would have taken for the same sampling.
  - `n_rings_unbalanced = 0` everywhere: the tight-ring balance failure predicted in the plan did not materialise, because the expansion loop resolves it.
- **The 15 `k_mismatch` findings are the audit working as designed.** Under the plan's `target_k` acceptance the ring constructs the index rather than measuring it; recording the stable measurement and reporting the disagreement is what preserved the check. These 15 are simple saddles whose excision loop consistently shows 3 ridge ports, and they are now visible in `launch_stats.failures` with id, expected/observed k, radii tried and final radius.
- **Validation**:
  - `tests/test_sphere_viewer_reduction.py`: 11 new tests on the seed-set rule -- visible raw saddles requested, hidden ones excluded, E-SADs included with `vertex_id = -1` and the constellation radius, E-MAX/E-MIN not seeded, `expected_k` from multiplicity for raw and `fold_order` for effective, the lone-CP radius rule (both the pitch floor and the 1.5 deg margin branches), `k_pitch` reaching the rule, world-to-centred mapping of effective centres, the fallback when no radius is recorded, and stable raw-then-effective ordering.
  - `tests/test_excision_launch_seeds.py`: 3 new tests pinning the reclassified taxonomy -- a single noisy overshoot must not terminate, a stable one is `k_mismatch`, and `n_rings_k_overshoot` is gone.
  - `tests/test_diagnose_params.py`: fixture gains the ring params and a real locator. The octahedron end-to-end still asserts `n_saddles_seeded == 2` and now yields 2 saddles x k = 2 = 4 ascent + 4 descent, converged at a0 -- a meaningful check on a 6-vertex, 90 deg-pitch mesh.
  - `uv run pytest`: **403 passed**. `ruff format` / `ruff check` / `mypy gba_topology2/ tests/`: clean.
- **Open next**: the sidebar (unit 8) -- `launch_seed_pitch` is now dead in the pipeline and must be replaced by the ring sliders and the failure list.

---

### [Entry 036] Launch Seeds Sidebar: Ring Controls and Failure List (`sphere_viewer.py`)

- **Objective**: Expose the excision ring controls, the expansion outcome taxonomy, and the unresolved-ring list, so the substrate is tunable and a flagged ring is findable in the scene.
- **Changes**:
  - Replaced the Launch Offset slider (`launch_seed_pitch`) with three controls: **Ring Samples** (24-96, step 8), **Expand Step** (0.25-3.0 x dtheta), **Radius Ceiling** (2-24 x dtheta, shown with its degree equivalent).
  - Second stats row: **Ring @ a0** (green) and **Expanded** (amber), plus "Mean N radii profiled (max M)". The split is the tuning signal -- a high expanded count says the excision-radius rule is too tight.
  - Separate lines for `n_rings_k_mismatch` ("ring k != reduction k") and `n_rings_unbalanced` ("never balanced (degenerate core)"), which the old single "Index mismatch (k != mu+1)" line conflated.
  - **Unresolved rings** list: id, status, observed vs expected k, radii tried, final radius.
  - Retired `launch_seed_pitch` from `DEFAULT_UI_STATE` and the `@state.change` handler. No consumers remained after the Step 9 rewire.
- **Retiring a control key needs no fixture migration**: `load_params_from_dict` iterates the *schema's* keys and reads them from the payload, so an extra key in an old file is simply ignored. Only *adding* a key breaks old files, which is why Entry 034's version bump was required and this one is not.
- **Verified live** (headless server on `ethene4.plt`, atom C1, driven through the browser):
  - Layout builds; all three sliders render with correct values and derived labels ("48 pts (7.5 deg pitch)", "1.00x dtheta", "12.0x dtheta (85.0 deg)").
  - **Sliders are reactive**: moving Ring Samples 48 -> 88 recomputed the pipeline live -- seeded 10 -> 9, expanded 0 -> 6, mean radii 1.33 -> 3.33, mismatches 2 -> 3, and the failure list repopulated with new ids.
  - Failure rows render with the id and diagnosis separated ("SAD_444 k_mismatch . k=3 != 2 . 3 radii . @21.3 deg"). The first draft concatenated them ("SAD_444k_mismatch") because inline elements collapse inter-element whitespace; fixed with an `ml-1` margin class, not a leading space, which HTML would also collapse.
  - Viewport check: ridge (orange-red) and valley (cyan) ports render on the purple catastrophe ring, matching the ascent/descent seed stubs from Entry 035.
- **The tuning finding, and it contradicts the obvious intuition**: raising Ring Samples from 48 to 88 made the search **worse** -- fewer saddles seeded (10 -> 9), more expansions (0 -> 6), more mismatches (2 -> 3). Finer azimuthal sampling resolves IDW interpolation noise as additional spurious extrema, inflating the measured k. This is the direct consequence of the Entry 031 measurement: at a tight ring the interpolated profile carries **3.6x the high-frequency content of the true field**, so sampling it more densely samples more noise.
  - The practical rule: **the residual error is an interpolation limit, not a sampling limit.** Raising `ring_samples` cannot buy accuracy; expanding the radius can, because the IDW error falls from 7.3% of field span at 1.6 deg to 0.5% at 20 deg. 48 is a reasonable default and the slider is for exploring the ceiling, not for beating it.
- **Validation**: `uv run pytest`: **403 passed**. `ruff format` / `ruff check` / `mypy gba_topology2/ tests/`: clean.

---

### [Entry 037] Removing the Expansion Early-Exit; Ring Geometry Fixed (3 sliders -> 0)

- **Objective**: Fix the reported defect (E-SAD_MC22 seeds only at expand step >= 1.25, not at 0.75) and reduce the number of adjustable parameters needed to find the correct topology, per the project goal.
- **The bug, and it was mine**: the `k_overshoot` -> `k_mismatch` early-exit added in Entry 033 terminates the search when two **consecutive ladder rungs** agree on an inflated `k`. "Consecutive" counts rungs, and rung spacing **is** the expand step (`step = expand_pitch * dtheta`). IDW noise is spatially correlated, so two closely spaced radii agree almost by construction. A fine, thorough scan was punished with premature termination; a coarse one survived. **The expand step was choosing which saddles seeded**, which is exactly what the goal forbids.
- **Reproduced on the real data before fixing** (Pd_20K, Electron Density (condensed), old code, step 0.75/48 samples):

  | saddle | 0.25 | 0.5 | 0.75 | 1.0 | 1.25 | 2.0 |
  |---|---|---|---|---|---|---|
  | E-SAD_MC22 | FAIL | FAIL | **FAIL** | FAIL | OK | OK |
  | E-SAD_MC4 | OK | OK | OK | OK | OK | OK |

  Exactly the reported symptom, including MC4 being unaffected. **Coding problem, not logic** -- the user's logic and the predecessor's were both right.
- **The fix**: delete the early-exit. The sweep now profiles every radius on the ladder and returns only on `k == expected_k`, which is precisely the predecessor's `compute_adaptive_excision_regions` (gba_topology/topology_engine.py:1477): sweep `r_sweep`, `break` on success, no early exit. That is why the predecessor never showed non-convergence. `best_k_at_balance` and the final `k_mismatch`/`unbalanced` diagnosis are kept -- that is the chi = 1 - k audit.
- **Verified after fixing**: MC22 and MC4 both converge at **every** step in {0.25, 0.5, 0.75, 1.0, 1.25, 2.0}, all at k = 2.
- **Parameters reduced 3 -> 0.** All three sliders removed; the values are module constants in `excision_launch_seeds`, matching the predecessor, which hardcoded them (3.0/24.0/0.5 deg, 64 samples):

  | Constant | Value | Predecessor |
  |---|---|---|
  | `RING_SAMPLES` | 16 | 64 |
  | `EXPAND_PITCH` | 0.5 dtheta | 0.5 deg |
  | `MAX_RADIUS_PITCH` | 16 dtheta (25.7 deg on Pd_20K) | 24 deg |

  They remain function arguments (defaults = the constants) so tests can vary them; the viewer and `diagnose_params` pass none, so the harness and the viewer cannot drift.
- **`RING_SAMPLES = 16` validated on real data**, and the user's recollection that 48 was overkill is confirmed. Yield is 100% at every count; the difference is *where* it converges:

  | dataset | samples | seeded | conv@a0 | expanded | mean radii | time |
  |---|---|---|---|---|---|---|
  | Pd_20K / Electron Density | 16 | 12/12 | 6 | 6 | 1.50 | 0.03s |
  | Pd_20K / Electron Density | 48 | 12/12 | **0** | **12** | **4.33** | 0.13s |
  | Pd_20K / V | 16 | 20/20 | 20 | 0 | 1.00 | 0.03s |
  | Pd_20K / V | 48 | 20/20 | 15 | 5 | 1.25 | 0.06s |
  | ethene4 / Electron Density | 16 | 5/5 | 5 | 0 | 1.00 | 0.01s |

  48 samples converges at a0 far less often and costs 4x, consistent with Entry 031: finer azimuthal sampling resolves interpolation noise as spurious extrema. 12 also works, so 16 has margin.
- **Two jobs, two constants**: `ring_samples` had been driving BOTH the measurement profile and the DRAWING of the visible purple/black boundary rings. Profiling now uses 16; drawing keeps `num_boundary_points=48` in `compute_micro_clusters`, so displayed rings are not 16-gons.
- **Ceiling semantics, simplified**: with no early exit, a tight ceiling just means fewer rungs -- it can no longer trap a ring. The earlier "the floor is load-bearing" caution from the plan is retired as the user directed.
- **Pd_20K Electron Density, 172 raw saddles, fixed geometry**: 127 converged at a0, 39 expanded, 6 `k_mismatch`, 0 unbalanced. Counters sum to 172 exactly. **0.67s total, 3.9ms per saddle.** The 6 failures each ran the full 31-rung ladder to the 25.7 deg ceiling reading k = 3 against an expected k = 2 -- genuine index disagreements, surfaced with ids in the sidebar.
- **Validation**:
  - `tests/test_excision_launch_seeds.py`: 4 new tests. The regression is pinned by a **scripted stub** that replaces the measurement with `k(radius)`, isolating the search policy from the numerics -- the first attempt used a synthetic noisy field and failed for the right reason (it modelled the opposite shape, k inflated at large radius), which is itself evidence the stub is the honest test. `test_search_is_invariant_to_the_expand_step` reproduces the MC22 ladder (inflated k below 4 deg, true k above) and asserts convergence at all six step sizes; `test_finer_step_is_at_least_as_thorough` asserts success is monotone in step size; `test_no_early_exit_scans_the_whole_ladder` asserts a permanently inflated k is diagnosed only after the full sweep.
  - **Mutation-checked**: restoring the early-exit fails 2 of the new tests, and reproduces MC22's exact failure pattern on real data.
  - `tests/test_sphere_viewer_params_export.py`: the two ring-control tests replaced by `test_ring_geometry_is_not_a_control` (the keys cannot return to `DEFAULT_UI_STATE`) and `test_snapshot_with_retired_ring_keys_still_loads` (old files carrying the keys load fine -- removing a key needs no version bump, only adding one does).
  - `tests/test_diagnose_params.py`: fixture drops the three keys.
  - Live check (Pd_20K, port 8921): card renders with the three sliders gone, stats grid, mismatch line and unresolved-ring list intact.
  - `uv run pytest`: **407 passed**. `ruff format` / `ruff check` / `mypy gba_topology2/ tests/`: clean.
- **Open next**: the 6 `k_mismatch` saddles on electron density are worth inspecting -- they are simple saddles whose loop consistently shows 3 ridge ports, which is a finding about the Morse detector, not the rings.

---

### [Entry 038] Provisional Launch Seeds: Rendering the k_mismatch Geometry

- **Objective**: Show launch points for saddles whose excision ring balanced cleanly at a fold order that **disagrees** with the index the reduction reports, while still flagging them as unresolved.
- **The motivating case (user-reported, reproduced on `Pd_20K.plt` / Curvedness)**: `E-SAD_MC21` reduces as a **-2** saddle (`expected_k = 3`, six ports), but the site is genuinely fourfold -- a nearby saddle and minimum merged into it, leaving an index-0 region adjacent. The ring reads **k = 4 at every radius out to the ceiling**, i.e. eight ports. The user predicted eight launch points; the code produced exactly eight.
- **The bug was a one-line filter, not missing data.** `expand_excision_seed` already returns a fully populated seed for `k_mismatch` -- all 8 branches intact. It was dropped at `seeds = [o.seed for o in outcomes if o.converged]` (`sphere_viewer.py:2320`) and by the identical filter in `summarize_outcomes`.
- **The strongest argument for the change: `n_index_mismatch` was a dead statistic.** It was computed over converged seeds only, and converged *means* `fold_order == expected_k`, so `index_agrees` is `True` for every seed it inspects -- the counter was **structurally always 0**, and a test asserted `== 0` with the comment "no seed emitted, so nothing to audit". The `chi = 1 - k` audit that `best_k_at_balance` exists to preserve was computed and thrown away. Counting over all renderable seeds makes it live.
- **Changes**:
  - `ExpansionOutcome.renderable` -- has branches to draw (converged, or `k_mismatch`). `unverified` -- renderable and not converged. `unbalanced` and `no_ring` seeds have **empty** branch lists, so "has branches" selects exactly the `k_mismatch` cases with no status test.
  - `sphere_viewer._make_seed_actor(renderer, rgb, opacity=1.0)` (new, module-level): builds a seed tube pipeline. Replaces ~35 duplicated lines and is called four times -- verified ascent/descent at full opacity, provisional ascent/descent at `UNVERIFIED_OPACITY = 0.45`.
  - **Colours are now structural, not asserted.** All six manifold-drawing actors (2 ports + 4 seeds) take their hue from the shared `ASCENT_RGB` / `DESCENT_RGB` constants. The provisional pair uses the same hues faded, not a third colour: a port and the stub growing out of it are the same manifold, and a new hue would add a legend entry.
  - `summarize_outcomes`: added `n_saddles_seeded_unverified`, `n_ascent_unverified`, `n_descent_unverified`; `max_fold_order` spans both sets; `n_index_mismatch` now counts over all renderable seeds; failure records carry `renderable`, `n_ascent`, `n_descent`.
  - **Seed/radius consistency fix**: the retained failure seed was the *ceiling* ring's, while `observed_k` was recorded at the *tightest balanced* radius -- the drawn geometry and the reported measurement disagreed. Now the seed, its radius, and `observed_k` all come from the tightest balanced ring (new `observed_radius_rad` field). On MC21 this moved the accepted radius from 25.2 deg (the ceiling) to **8.3 deg** -- a far more local, more trustworthy measurement.
  - **Visibility gate fixed**: it read `n_ascent`/`n_descent`, which excluded provisional seeds, so a populated actor could stay hidden. Now gates on the sum; each actor's own visibility uses its own count.
  - `reset_defaults` hides the two new actors. `apply_restore_plan` needs no change -- it ends in `update_field()` -> `update_topology()`, which recomputes visibility.
  - Sidebar: a "Provisional (faded in scene)" line, and failure rows now append "N seeds shown faded".
  - `failures` is sorted renderable-first before the `[:40]` cap, so a saddle with geometry in the scene is not pushed off the list.
- **Validation on `Pd_20K.plt` / Curvedness, atom 1, reduce_multi_clusters on (30 saddles)**:
  - verified: 29 seeded, 68 ascent + 68 descent; provisional: **1 seeded, 4 ascent + 4 descent = 8**.
  - `n_index_mismatch = 1` (was structurally 0).
  - `E-SAD_MC21`: `status=k_mismatch`, `converged=False`, `renderable=True`, `unverified=True`, `expected_k=3`, `observed_k=4`, seed `fold_order=4`, 4+4 branches, 22 radii, accepted R=8.32 deg.
  - **Live browser check**: sidebar reads "Provisional (faded in scene): 1 saddle(s), 4 ascent + 4 descent", "Index mismatch (ring k != reduction k): 1", and the failure row "E-SAD_MC21 k_mismatch . k=4 != 3 . 22 radii . @8.3 deg . 8 seeds shown faded".
- **Complications considered (the user asked)**:
  1. **Visibility gate coupling** -- real, fixed above.
  2. **`n_saddles_seeded` semantics** -- kept meaning "verified/certified"; provisional is counted separately so the sidebar never mixes them.
  3. **Step 3 must not inherit these silently** -- convergence is the certification that a seed set is consistent with the reduction. The two lists stay separate (`seeds` / `unverified_seeds`), so a future integrator opts in deliberately. A k=4 seed set on a -2 saddle traces separatrices that do not close against the reduction's index.
  4. **Visual ambiguity** -- addressed by opacity, not hue.
- **Validation**:
  - `tests/test_excision_launch_seeds.py`: 9 new tests -- `renderable`/`unverified` truth tables across all five statuses, retained seed matches `observed_k` and its radius, verified/provisional counts stay separate, `n_index_mismatch` is live, failure records carry branch counts, renderable failures sort first.
  - `tests/test_sphere_viewer_seed_actors.py` (new, 5 tests): the factory registers with the renderer, starts hidden, applies the given hue, is opaque by default, fades only when asked, and returns independent pipelines per call (sharing a polydata would mix verified and provisional geometry).
  - `tests/test_sphere_viewer_port_colors.py`: rewritten to assert every manifold actor is wired to a shared constant, plus a guard that none hardcodes a literal RGB. **Mutation-checked**: reverting a port to literal cyan fails 3 tests.
  - `uv run pytest`: **425 passed**. `ruff format` / `ruff check` / `mypy gba_topology2/ tests/`: clean.
- **Pre-existing defects found, NOT fixed (out of scope, unrelated to this work)**: two toolbar/table actions are referenced with `ctrl.<name>` but never defined with `@ctrl.add`, so clicking them raises `FunctionNotImplementedError` -- `select_cp_from_list` (2 refs) and `reset_camera` (2 refs). Both confirmed present at baseline `bee00a5` with 2 references and 0 definitions.
- **Open next**: MC21 is a real reduction defect -- a fourfold site collapsed to -2 by an adjacent saddle+minimum merge. The ring is right and the reduction is wrong, which is what the provisional seeds now make visible.

---

## 2026-10-31 -- Pairwise monkey saddles and the bimodal-quad gate

- **Symptom** (reported on `Pd.plt`, field `V (condensed)`): MC 41/42/43/44 --
  four simple saddles -- were absorbed into a single catastrophe whose purple
  ring enclosed all four. The user expected two monkey saddles, MC 44+42 and
  MC 43+41, each with its own ring.
- **Measured geometry**: the four form two pairs separated by **12.24°**
  (41+43, 42+44) with **63.1-64.5°** between the pairs. All four carry the same
  value (0.0202954, spread 8.9e-7 relative). There is **no core at the centre of
  the four**: the nearest positive cluster, MC25, is 29.3-34.6° from each pair
  midpoint, i.e. 2.4-2.8 pair-widths away.
- **Root cause (two compounding defects)**:
  1. **The coreless monkey saddle was not modelled.** A k=3 monkey saddle
     (chi = -2) unfolds two ways: on a smooth extremum into a core (chi = +1)
     plus three saddles, and on a flat ridge/valley into **two** simple saddles
     whose indices sum to -2 with no core at all. `classify_catastrophes`
     searched only from `local_euler_index > 0` cores, so the second channel was
     invisible. MC25 reached into the 12° search window, grabbed three of the
     four saddles, and reported a bogus constellation.
  2. **k=4 was matched by first fit, not tightest fit.** With four saddles in
     range the quad loop took the first index-balancing combination, so two
     disjoint pairs were merged into one octupole spanning 64.5° -> ring radius
     35.6°.
- **Fix** (`src/catastrophe_classifier.py`):
  - `find_saddle_pairs` (new): bound saddle pairs with `chi = -1 + -1 = -2`.
     Gates: separation <= `pair_max_pitch` (5.0) pitches, relative value barrier
     <= `barrier_tol`, no extremum on the geodesic between them
     (`has_intervening_extremum`), and **coreless** -- nearest positive core
     farther than `coreless_min_ratio` (1.8) x pair separation. Greedy,
     tightest-first, each saddle consumed once.
  - `split_quad_into_pairs` / `is_bimodal_quad` (new): the min-sum perfect
     matching of four saddles into two pairs. A true octupole has four
     comparable gaps; two independent monkey saddles have a small intra-pair
     angle and a large inter-pair gap. A quad is refused as k=4 when
     `min_inter >= bimodal_ratio` (2.5) x `max_intra` and `max_intra` is within
     the pair window.
  - `constellation_span_deg` (new) + combo search rewrite: within each fold
     order the **tightest** index-balancing combination now wins.
  - `evaluate_constellation_metrics` and `build_catastrophe` accept
     `core=None`; `net_euler_index` for a pair is the summed member index (-2),
     which equals `1 - k` at k=3.
- **Why `coreless_min_ratio` is load-bearing**: without it the new channel
  *stole* flanking saddles from legitimate cores. Measured before/after on 10
  field/file combinations, the guard is the difference between a fix and a
  regression:

  | field | pairs found | coverage lost |
  |---|---|---|
  | Pd.plt negative mean curvature | 10 (ratio 0.62-0.96) | 10 clusters |
  | Pd.plt Willmore energy | 3 (ratio 0.86) | 3 clusters |
  | ethene.plt Electron Density | 2 (ratio 0.72) | 6 clusters |
  | ethene3.plt mc sign-change arc | 6 (ratio 0.86-1.40) | 4 clusters |
  | **Pd.plt V (condensed)** | **2 (ratio 2.40, 2.79)** | **0** |

  Every spurious pair sits at ratio <= 1.40; the two real ones at >= 2.40. The
  threshold 1.8 separates them with margin, and it is scale-free -- a ratio of
  angles, so it needs no per-field tuning.
- **Result on `Pd.plt` / `V (condensed)`**: 1 catastrophe (E-4SAD, span 64.5°,
  ring 35.6°, 5 clusters) -> **2 catastrophes** (E-MSAD, span 12.2°, ring
  **9.4°**, 2 clusters each): `['MC42','MC44']` and `['MC41','MC43']`. MC25 is
  released back to standalone reduction.
- **Independent physical confirmation**: profiling the new 9.4° rings through
  `profile_boundary_loop` yields **3 valley + 3 ridge ports, topologically
  balanced** on both -- the exact 6-port signature of a k=3 monkey saddle. The
  geometry was right before any threshold was tuned for it.
- **Blast radius**: across Pd.plt, Pd_20K.plt, ethene/2/3.plt and all 13 fields,
  cluster coverage is unchanged or improved (ethene Electron Density 92 -> 96,
  Gaussian curvature 63 -> 65, Pd Willmore 59 -> 61 as the tightest-fit search
  replaced first-fit). No field loses a cluster.
- **Validation**: 5 new tests in `tests/test_catastrophe_classifier.py` --
  bimodal detection on the real Pd geometry, true-octupole acceptance, the
  four-cluster precondition, the end-to-end two-monkey-saddle outcome with
  per-pair ring radii, and an A/B proving the coreless guard protects core
  constellations. `uv run pytest`: **431 passed**. `ruff format` / `ruff check`
  / `mypy gba_topology2/ tests/`: clean.
- **Open next**: the three new thresholds (`pair_max_pitch`, `bimodal_ratio`,
  `coreless_min_ratio`) are module defaults, not yet exposed as sidebar
  controls. `cat_barrier_tol` and `cat_angular_tol_mult` already are; these
  should join them so the pairwise channel can be swept live on the Pd fields.

### [Entry 039] Geodesic Primitives Promoted to `sphere_frame.py`

- **Objective**: Give the new maximum-connector a single, public home for
  great-circle math, instead of a third private copy.
- **Changes**: `sphere_frame.py` gains `unit`, `tangent_project`,
  `rotate_towards` (the geodesic exponential map, moved from
  `saddle_launch._rotate_towards`) and `angular_distance` (clipped `acos` of the
  normalised dot, so coincident/antipodal points never yield NaN). The
  on-sphere formulas belong in the module that owns the centred frame: they need
  `p/||p||` to be the outward normal, exactly like every angular computation
  already documented there. `saddle_launch` keeps thin aliases
  (`_unit = unit`, etc.) so its call sites and any historical import read the
  same.
- **Validation**: 6 new tests in `tests/test_sphere_frame.py` -- normalisation
  and the degenerate zero, tangency, on-sphere + monotone advance, the
  non-positive-angle identity, the basic 0 / pi/2 / pi cases, and a
  chained-vs-single step test proving the map is a true geodesic (30 steps match
  one step to 1e-9).

### [Entry 040] Maximum Connection Subroutine (Basin Locator Step 3)

- **Objective**: For each ascent launch branch of a visible saddle, extrapolate a
  great-circle ray and claim the maximum it points at.
- **New module `src/max_connection.py`**:
  - `connect_seed_to_maxima(seeds, maxima, sphere_radius, delta_theta_mesh_rad,
    center, max_arc_pitch, direction_mode)` -> one `MaxConnection` per
    (seed, ascent branch).
  - **Matching rule = cross-track minimum, capped arc.** In the frame
    `(o, d, n)` with `n = o x d`, a target scores `along = atan2(t.d, t.o)`
    (signed progress) and `cross = asin(t.n)` (perpendicular deviation). The
    winner is the smallest `|cross|` among `0 < along <= cap`. Ties break on
    `along` then `max_id`, and the candidate list is pre-sorted by id, so the
    result is a pure function of the input SET, not its order.
  - **Why not plain proximity**: "nearest maximum to the launch point" ignores
    direction. On a dense field the nearest maximum is almost always off to the
    side of where the branch points, and two opposite branches of one saddle
    collapse onto the same extremum. `test_cross_track_beats_plain_proximity`
    pins a fixture where the nearer maximum is 90 deg off-ray and must lose.
  - `mark_mutual` flags the tightest claim on each maximum (one per maximum,
    connected only). Reported, never gating -- a non-mutual claim is the crossing
    separatrix signal.
  - **Frame contract**: seeds are centred, `BasinCenter` positions are world.
    `center` maps the latter in, the whole computation runs centred, arcs are
    emitted world-frame. This is the `extract_basin_centers` fix pattern; Pd_20K
    sits at the origin so a frame bug is invisible there --
    `test_world_frame_and_centred_runs_agree` runs identical geometry offset and
    requires identical metrics, and asserts the offset arc is the centred arc
    translated by the nucleus.
- **The two ray definitions are the same great circle (measured).** `chord`
  starts at the saddle centre aiming at the port; `port_tangent` starts at the
  port along `_outward_tangent`. `_outward_tangent` builds the port direction as
  `radial*port - centre`, which is the parallel transport of the centre->port
  geodesic, so both rays lie in the plane spanned by centre and port and share
  its pole. They differ only in the arc's start: `cross_track` (the quantity the
  rule minimises) is identical, `along_track` shifts by the ring radius.
  **Measured on the three saved Pd_20K snapshots** (V 24, curvedness 72, shape
  index 96 branches): identical claim sets, cross-track equal to 1e-11 deg.
  `test_both_modes_share_the_same_great_circle` pins the pole dot = 1 and the
  along-shift = ring radius.
- **Consequence for the UI**: because the mode cannot change an answer, it is an
  API parameter (it will matter to the Step 4 integrator, which starts at the
  port), NOT a viewer control. AGENTS.md Principle 4: a knob that cannot change a
  measurement does not ship.
- **Arc cap `MAX_ARC_PITCH = 40.0`** (~64 deg at Pd_20K pitch 1.6058). Measured
  max saddle->claimed-maximum separation across the three snapshots: 57.1 / 55.2
  / 45.6 deg, so the cap admits every observed connection while excluding the far
  side of the sphere where cross-track stops discriminating. A ceiling, not a
  tuned threshold.
- **Validation**: 24 tests in `tests/test_max_connection.py` -- ray bearing,
  unit/tangency, the two modes' distinct starts, the closed-form
  `along = atan2(sc,c)` / `cross = asin(s*sin az)` decomposition (cross is NOT
  the azimuth offset; it shrinks with sin(colatitude), which is the correct
  great-circle distance -- the first draft of the test asserted the naive
  azimuth reading and failed), cross-track-beats-proximity, opposite branches
  claim opposite maxima, arc cap -> `no_target`, rear hemisphere never claimed,
  ascent-only, minima filtered, mutual uniqueness, arc points on the sphere, arc
  is a great circle (zero torsion), world/centred agreement, order independence
  (8 shuffles), degenerate centre, empty inputs, summary finiteness, verified
  flag.

### [Entry 041] Params Schema v3 + Tolerant Legacy Load; `connect` Stage

- **Objective**: Make the new observation reproducible headlessly, and keep the
  existing v2 snapshots loadable.
- **The trap**: adding a control to `DEFAULT_UI_STATE` is breaking --
  `load_params_from_dict` requires every control key, so the three attached
  `Pd_20K_atom1_params_*.json` (v2) would fail with a confusing "Missing
  parameter 'show_max_saddle_arcs'". The version-2 comment in `params_schema`
  predicted exactly this failure mode.
- **Fix**: `PARAMS_SCHEMA_VERSION = 3` and a new
  `PARAM_LEGACY_SCHEMA_VERSIONS = frozenset({2})`. A legacy snapshot fills absent
  controls from `DEFAULT_UI_STATE` (a control invented after a file was written
  had exactly one value on screen: its default). The **current** version stays
  strict -- a truncated v3 file is a bug to surface, not a gap to paper over.
  `test_current_version_stays_strict_about_missing_controls` pins the asymmetry.
- `PARAM_METRIC_KEYS` gains `connection_stats`. `PARAM_STRING_KEYS` is now
  derived from the str-valued defaults (empty today; every selector is
  data-dependent) so the first str widget self-registers.
- **`diagnose_params.py`**: new `connect` stage after `launch` (consumes
  `ctx.seeds + ctx.unverified_seeds` and the `extract_basin_centers` output,
  mirroring the viewer). `RunContext` gains `unverified_seeds`, `connections`,
  `connection_stats`; `computed_metrics` and `dump_results` carry them.
- **`inspect_max_connections.py`** (new probe, `inspect_basin_centers.py`
  pattern): prints the per-branch table (saddle, branch, maximum, cross, along,
  sep, mutual/verified flags), sortable, with `--mode` / `--max-arc` overrides
  that vary the module constants without touching the saved controls.
- **Validation on the three snapshots**: reproduce check **47/47 match, exit 0**
  on all three (the differ walks the SAVED structure, so a new metric key is
  reported as new-only and cannot cause false drift). V: 24 branches, 24
  connected, 6/6 maxima, cross mean 4.36 deg, max 7.82. Curvedness: 72/72, 12
  maxima, mean cross 0.38. Shape index: 96/96, 26 maxima, mean cross 0.50. Every
  ascent branch claims a maximum; cross-track stays under 8 deg everywhere.

### [Entry 042] Viewer Overlay: Arcs From the Selected Maximum

- **Objective**: Draw great-circle arcs from a *selected* maximum to the saddles
  that claim it.
- **Changes** (`sphere_viewer.py`):
  - Actor built by the shared `_make_seed_actor(renderer, MAX_ARC_RGB,
    opacity=MAX_ARC_OPACITY)` factory. Gold (the basin-center hue) so an arc
    reads as belonging to the selected extremum, semi-transparent because the
    arcs span tens of degrees straight through the seed field.
    `test_max_arc_actor_uses_the_shared_gold_constant` forbids a hardcoded RGB.
  - `update_topology` stage 10 fills `runtime_max_connections`,
    `state.max_connections_list` and `state.connection_stats` for ALL branches
    (cheap, and the sidebar lists it). The arcs are selection-driven, so
    `_emit_max_arcs(basin_id)` is called from `select_basin_center`, not the
    rebuild -- a click is a render-only change. `update_topology` re-invokes
    `select_basin_center` at the end so a rebuild refreshes the overlay against
    the new table.
  - The arc points arrive world-frame; the 1.020 lift is applied about the
    nucleus (`to_local` -> `radial_offset` -> `to_global`), the correct order per
    the origin-centred audit.
  - `show_max_saddle_arcs` toggle in `DEFAULT_UI_STATE` (auto-registers in
    `PARAM_BOOL_KEYS`), added to `@state.change`, hidden in `reset_defaults`.
  - Sidebar: a "Basins: Step 3 Max Connections" card at drawer indent 16 (the
    Step 2 card's indent), with counters, a cross-track/separation readout, and a
    per-arc table for the selected maximum. Selecting a minimum, or nothing,
    shows a hint instead -- minima have no ascent connections.
  - `test_provisional_seeds_share_the_hue_and_differ_only_in_opacity` counted
    every solid `_make_seed_actor` call and broke at 3; narrowed to the
    ASCENT/DESCENT hues, since the gold overlay is not a seed colour.
- **Validation**: headless smoke (the `s.start` stub recipe) reaches
  `server.start`, so the full VTK + Vuetify layout constructs. `pytest`: 466
  passed. `ruff` / `mypy gba_topology2 tests` / `pre-commit --all-files`: clean.
- **Open next**:
  1. The connector is a straight great circle. Step 4 replaces it with the
     traced separatrix; the geodesic is the baseline to diff the integrator
     against.
  2. `direction_mode` is dormant (both modes agree on the excision substrate).
     It becomes load-bearing the moment the integrator starts at the port.
  3. Cross-track is reported, not gated. On the V field the spread is 0-8 deg;
     a cone gate is deferred until the traced path shows what a "real" cross-track
     looks like.
  4. Two ascent branches from one saddle can claim the same maximum (both are
     distinct gateways). Kept; `n_branches` per pair is reported. Dedupe only if
     the traced version proves them redundant.

### [Entry 043] Step 3 Rule Fixed: First-Hit Along the Ray, Not Best-Aim

- **Objective**: The connector claimed maxima a ray passed *through* on the way.
  Add the constraint that a connector cannot cross another critical point.
- **The bug, in one line**: the rule minimised `|cross_track|` with `along` as a
  *tie-break*. That scores **aim** and never checks **reach**. A ray passing
  through a near maximum toward a far one has `cross ~ 0` for BOTH (they are on
  the great circle by construction), so cross-track is *provably* unable to
  separate them and the tie-break was decided by float noise at 1e-11 rad.
  Three contributing defects: the pool was maxima-only (so "passes through a
  critical point" was inexpressible), the 40-pitch cap was the only distance
  gate, and the drawn arc is the saddle->winner geodesic, so the renderer
  faithfully drew the occlusion the matcher ignored -- which is what made it
  visible.
- **My validation missed it**: I checked aggregate counters and schema plumbing
  on the earlier snapshots and never asserted the first-hit property or eyeballed
  one arc against the glyph field. Worse, `test_cross_track_beats_plain_proximity`
  actively *pinned* the wrong behaviour. It tested "direction beats proximity"
  and was blind to "proximity beats direction when direction is degenerate".
- **The fix**: `connect_seed_to_maxima` now minimises **along-track** among
  **all visible critical points** inside the forward sector
  (`0 < along <= cap`, `|cross| <= cone`). If the first hit is a maximum the
  branch CONNECTS; if it is a minimum or saddle the branch is **OCCLUDED** and
  the blocker's id + polarity are recorded. Ties break on `|cross|` then id.
- **Both requested parameters collapsed into one, and the second vanished.**
  The score-then-test design needed a cone *and* an on-ray tolerance `eps`.
  First-hit needs neither tolerance: a maximum exactly on the ray is both the
  nearest hit and its own occluder, so the two rules agree by construction.
  `eps` is gone. The cone remains as one constant.
- **`sector_width_rad` measured UNUSABLE as the cone** (my proposed
  parameter-free option, and I was wrong to propose it). It is the gap between
  bounding separatrix crossings and degenerates whenever a sector contains all
  of them: on Curvedness **40 of 72 branches report 0.0 deg** and several report
  **355-357 deg**; on V four report 352.6 deg. So the cone is the pitch-relative
  `CONE_PITCH = 6.0` (9.63 deg at Pd_20K pitch 1.6058), chosen as the smallest
  pitch multiple admitting every cross-track observed on the four snapshots
  (max 7.82 deg = 4.9 pitches, on V).
- **New pool builder `build_connection_pool`**: basin centers carry extrema only,
  so the visible saddles are added from the two places they live -- raw Morse
  saddles not in `hidden_vertex_ids` (CENTRED frame) and `E-SAD` entries of the
  effective-CP dicts (WORLD frame, no vertex to scan). The frames are reconciled
  with `to_global` inside the builder. This is the recurring effective-CP trap;
  the test asserts the offset is applied.
- **Validation on `Pd_20K_atom1_params_Shape2.json`** (shape index, k_pitch 2.2,
  k_bridge 2.75, 96 ascent branches, pool 98 visible CPs = 26 max + 24 min +
  48 sad). Id mapping confirmed: **MC139=SAD_384, MC140=SAD_385, MC143=SAD_388,
  MC144=SAD_389** (all chi=-1 single-saddle clusters).
  - **24 of 96 branches changed claim.**
  - **`E-MAX_MC109` goes from exactly 2 connectors to 4** -- the count the user
    reported as missing. It gains `SAD_384` (MC139) and `SAD_385` (MC140).
  - The over-claiming maxima collapse from 8 connectors to 4: MC7, MC8, MC111,
    MC112, MC113, MC114. Mean separation 31.3 -> **20.1 deg**, max 57.1 ->
    **33.0 deg**. Connections became local, which is what a separatrix is.
  - Reproduce check **56/59**: the 3 drifts are exactly
    `mean/max_separation_deg` and `mean_cross_track_deg`. Every other metric
    matches, i.e. the change is confined to Step 3. This is the tool working --
    a rule change that did NOT move these numbers would be the surprise.
  - **Occlusion fires on Curvedness: 20 of 72 branches blocked, all by minima**
    (0 by saddles). On V and both shape-index snapshots, 0 occluded.
- **The specific expectation did NOT survive measurement, and the trace says
  why.** The report was that MC143/MC144's uphill directions walk through MC109.
  Measured: **MC109 is 90.0 deg from MC143 and 89.95 deg from MC144** -- it is
  perpendicular, not behind. `--trace SAD_388` shows MC109 at
  `along = -94, cross = +90` on both branches: behind the ray and 90 deg off it,
  unreachable by any forward-sector rule. What MC143/MC144 actually claim is
  `E-MAX_MC110` (along 15.3/16.1) and `MAX_4`/`MAX_5` (along 19.5). MC109 and
  MC110 are 90.08 deg apart, so the two are the symmetric partners of a quartet,
  and the fix moved MC109's connectors from {MC139, MC140} to
  {MC139, MC140, E-SAD_MC73, E-SAD_MC77}. The count the user wanted is now
  correct; the attribution is to MC139/MC140, not MC143/MC144.
- **`--trace` added to `inspect_max_connections.py`**: scores every pool CP
  against each branch of one saddle, in-sector first, with the verdict column
  showing the blocker. This is what turned "the rule is wrong" into a number.
  It had its own instance of the frame bug -- it scored world-frame pool
  positions against centred-frame rays, invisible on origin-centred Pd_20K --
  caught on review and fixed by passing `center` and mapping with `to_local`.
  Verified: the trace's top in-sector entry equals the rule's claim on every
  branch of SAD_384/388/389.
- **UI**: the Step 3 card chip now reads "N/M connect", gains an occlusion line
  (blocked-by-minimum / -saddle / no-target counts), the per-arc table gains an
  **Along** column (the primary key), and the explanatory text describes
  first-hit. `MaxConnection` gains `blocker_id` / `blocker_polarity`;
  `mark_mutual` switched to `dataclasses.replace` so it cannot drop fields when
  the dataclass grows (it would have silently lost the blocker on the first
  version of this change).
- **Validation**: 34 tests in `tests/test_max_connection.py` (was 24) --
  first-hit-beats-farther-on-axis (the MC109/MC8 inversion of the old proximity
  test), minimum-on-ray occludes, saddle-on-ray occludes, source saddle is never
  its own target, cone excludes an off-sector nearer target, 5 pool-construction
  tests (merge, hidden exclusion, E-SAD inclusion, world-frame mapping, dedup),
  mark_mutual preserves blockers, 3-status summary. `pytest`: **477 passed**.
  `ruff` / `mypy gba_topology2 tests` / `pre-commit --all-files` clean. Headless
  viewer smoke reaches `server.start`.
- **Open next**:
  1. `CONE_PITCH = 6.0` is calibrated on Pd_20K only. The V field's 7.82 deg
     max cross-track is 4.9 pitches, so 6.0 has ~20% headroom there and is
     untested on a coarser mesh.
  2. 0 occlusions on V/shape-index is a real result, not a gap: with 26 maxima
     and 48 saddles in a 9.6 deg sector, the first hit is nearly always a
     maximum. The mechanism is proven on Curvedness.
  3. The straight great circle is still the connector. Step 4's integrator
     replaces it; `direction_mode` becomes load-bearing when it starts at the
     port.

### [Entry 044] Step 3: Launch From the Port, Claim the Disc, Not the Centre

- **Objective**: Monkey saddles on the Curvedness field lost connections (MC18
  had no arc to MC38). The user's model: propagate from the point where the
  launch direction crosses the excision boundary, and treat a maximum as reached
  when the arc meets its boundary.
- **The user's diagnosis was correct on both counts, and I had it wrong twice.**
  My Entry 043 answer ("the geodesic clips a minimum the true ridge goes around,
  only Step 4 fixes it") was wrong. Two real defects, both mine:
  1. **The ray started at the saddle CENTRE, not the port.** `direction_mode`
     defaulted to `"chord"`. The disc inside the excision boundary is the
     degenerate core whose field is meaningless -- the entire reason the boundary
     exists -- and launching from the centre means the connector's first ~ring
     radius traverses it, so any CP the ring swallowed sits on the ray and
     occludes. From the port, every CP inside the boundary has `along < 0` and is
     excluded by construction.
  2. **The rule scored the CP's CENTRE, not its disc.** A minimum at
     `along 7.5, cross 7.4` with a 1.6 deg disc blocked MC18's branch 0 from
     MC38 at `along 45, cross 0.4` with a 2.9 deg disc. The ray passes 7.4 deg
     clear of a 1.6 deg object -- it is not an obstacle. It was a candidate only
     because the 9.63 deg cone is six times wider than its disc, and it won
     because its centre is nearer.
- **The fix**: `disc_entry_angle(along, cross, rho)` returns the arc-length where
  the great circle first enters the disc, `along - acos(cos rho / cos cross)`,
  defined exactly when `|cross| <= rho`. The rule minimises ENTRY, and a CP whose
  disc the ray misses is not a candidate at all. Default mode is now
  `"port_tangent"`. The arc is two-part: leg 1 along the ray to the boundary
  contact, leg 2 the geodesic from contact to centre (the user's drawing rule).
- **Measured on `Pd_20K_atom1_params_curv.json`** (curvedness, k_pitch 2.0,
  k_bridge 1.75, 72 ascent branches, pool 73 visible CPs = 12 max + 31 min +
  30 sad, disc radii 1.61-16.76 deg, median 2.91):
  - **occluded 20 -> 0, connected 52 -> 72.** Every branch now claims a maximum.
  - **MC18 branch 0 -> E-MAX_MC38**, entry 33.79, cross -0.39, disc 2.91. The
    minima that blocked it (MIN_19165/19161 at cross 7.4, disc 1.61) now read
    `miss` -- the great circle passes outside their discs.
  - MC18 is k=4 (ring 8.33 deg), 4 ascent branches, all propagate. The "only 4-5
    of 6" symptom was the connection layer dropping branches, never the seed
    layer: `n_ascent = 68` over 29 converged seeds solves to 10 k=3 + 19 k=2,
    i.e. every port emitted.
  - Reproduce check **53/62**: the 9 drifts are all `connection_stats` leaves
    (direction_mode, the 4 counts, 3 angular stats, n_unverified 0->4 from the
    provisional MC21 seed now being connected). Every non-Step-3 metric matches.
- **The cone is now provably redundant, and I am keeping it.** Cone 9.63 deg and
  144 deg give identical verdicts on this snapshot: every pool target carries a
  real disc radius (0 of 73 are points), so the per-object `|cross| <= rho` test
  subsumes a global gate. Same for the cap (max entry ~42 deg, cap 64 deg). Both
  stay as the `no_target` mechanism for genuinely empty sectors and to bound the
  search, and as the fallback if a target ever has radius 0. Recorded so the next
  tuning pass knows they are not load-bearing here.
- **Frame safety**: `build_connection_pool` takes `ring_radius_by_vertex` (raw
  CPs, keyed by mesh vertex, from `MicroCluster.angular_radius_rad`) and
  `ring_radius_by_id` (effective CPs, the viewer's `eff_ring_radius`). Verified
  on **ethene4 atom 1, an off-origin atom** -- reproduce 35/35, so the centred
  raw / world effective merge is correct. Pd_20K sits at the origin and would
  have hidden this.
- **A latent bug caught in the probe**: `for *_, s in scored` silently bound `s`
  to the newly appended 6th tuple field (disc radius) instead of the in-sector
  flag. Star-unpacking is positionally fragile; the trace now indexes `r[4]`.
  `--trace` also had its own instance of the frame bug (world pool vs centred
  rays), fixed in Entry 043 and re-verified.
- **Validation**: 33 tests in `tests/test_max_connection.py` -- `disc_entry_angle`
  on-axis / tangent / miss / zero-radius-degenerate; port-origin starts at the
  port and puts an in-ring CP at `along < 0`; first-hit on the disc; minimum and
  saddle discs occlude; **the MC18 off-ray-minimum case**; source-saddle skip;
  cone gate; cap; rear hemisphere; two-part arc endpoints (port first, centre
  last, all on the sphere); 5 pool tests now asserting disc radii; world/centred
  agreement; order independence. `pytest`: **476 passed**. `ruff` /
  `mypy gba_topology2 tests` / `pre-commit --all-files` clean. Headless viewer
  smoke reaches `server.start`.
- **Open next**:
  1. The straight great circle is still the connector. The 20 occlusions it used
     to report were artefacts of the centre-point test, NOT real curvature
     effects -- so the "geodesic vs separatrix diverge" case is now UNMEASURED.
     Step 4's integrator is the baseline test for that.
  2. `CONE_PITCH` / `MAX_ARC_PITCH` are non-binding on curvedness. Untested on a
     field with sparse maxima, where a wide cone could admit a genuine off-ray
     claim.
  3. The arc's leg 2 (contact -> centre) is a drawing convention. Once Step 4
     traces the separatrix, the whole arc should come from the integrator.

### [Entry 045] Click a Glyph, Light the Row: Selection-to-Table Resolvers

- **Objective**: Restore the capability where clicking a CP / effective CP in the
  3D scene highlights the corresponding cluster and its designation in the
  sidebar. Clicking a maximum should say *which* maximum it is, and with the arc
  overlay on, reveal its saddle connections.
- **The gap was narrower than it looked.** `select_cp` / `select_cluster` /
  `select_basin_center` all set `state.selected_*`, and those drive the
  **inspector cards** below the tables. The table **rows** carried a static
  `classes="cursor-pointer"` with no binding to any selection state, so a glyph
  click opened the card and never lit the row. Nothing was lost to restore --
  the row highlight was never wired.
- **Three pure resolvers** at module level, so the id-space joins are testable
  without trame, a render window, or a browser:
  - `basin_center_id_for_cp` -- a raw extremum joins on `MAX_<v>` / `MIN_<v>`
    from `vertex_id`; an effective CP joins on its own `id_label`. A saddle, a
    hidden CP, and an E-SAD have no basin center and resolve to None.
  - `cluster_id_for_cp` -- a raw CP scans `members[].vertex_id`; an effective CP
    matches the row `id` against its `origin_id`. A CAT-prefixed origin lives in
    the catastrophe table, so it resolves to None here.
  - `basin_center_id_for_cluster` -- tries `E-MAX_<unit_id>` / `E-MIN_<unit_id>`
    first (the visible glyph), then falls back to the members' raw `MAX_<v>`.
- **The insight that made it work: wire the CLUSTER path, not just the CP path.**
  `on_scene_click` tests cluster centroids *before* CP glyphs, so clicking a
  maximum glyph usually lands on `select_cluster`, not `select_cp`. Wiring only
  `select_cp` -- the obvious reading of "clicking on a cp" -- would have fixed
  the rare case and missed the common one. Both now resolve a basin center.
- **The arc reveal is free.** `select_basin_center` already calls
  `_emit_max_arcs`, which gates on `state.show_max_saddle_arcs`. So driving the
  basin-center selection from a glyph click gives exactly the requested
  behaviour: toggle on -> the maximum's row lights AND its saddle arcs appear;
  toggle off -> the row lights and the micro-cluster is highlighted, no arcs.
  No new branching was needed, and none was added.
- **One-way edge, guarded.** `select_cp` -> `select_basin_center` and
  `select_cluster` -> `select_basin_center`; `select_basin_center` calls neither,
  so there is no cycle. Each compares the resolved id against the current
  selection first, so re-clicking the same maximum does not rebuild the arcs.
- **No custom CSS is possible.** Vue strips `<style>` from component templates
  (documented at the theme block: that is why the drawer has a Vuetify theme
  instead of CSS overrides). The highlight is therefore a Vuetify utility class,
  `bg-amber-lighten-4`, bound with `:class`. Trame emits a dynamic binding only
  for a **tuple** attribute value -- a plain string is emitted verbatim as a
  static `class=` -- so each use site is `classes=(EXPR,)`. The three JS
  expressions are module constants (`CLUSTER_ROW_CLASSES`,
  `EFFECTIVE_CP_ROW_CLASSES`, `BASIN_CENTER_ROW_CLASSES`) shared by the rows.
- **A separate `highlighted_cluster_id` key, not `selected_cluster`.** A CP click
  should light the cluster *row* without opening the cluster *inspector card*
  (which would cover the CP card the operator just asked for). The id key drives
  the row; `selected_cluster` keeps driving the card.
- **Validation on `Pd_20K_atom1_params_curv.json`** (43 basin centers, 74 unit
  rows, 972 CP dicts):
  - all **12 of 12** E-MAX CPs resolve to their own `id_label`, 0 failures;
  - **972 / 972** CPs resolve to an owning cluster row;
  - **43 / 43** basin centers are reachable from some CP, and 43 of 74 unit rows
    resolve to a basin center. The split is exactly the reduction decision:
    every resolved row has `local_euler_index == +1` (an extremum set), and all
    31 unresolved rows have chi in {-3, -2, -1, 0} -- saddle sets and annihilated
    chi=0 sets, which correctly light no extremum row. The 266 maxima members
    inside unresolved rows are absorbed into a saddle/corridor whose visible
    glyph is the E-SAD, not a basin center;
  - `E-MAX_MC38` resolves in both directions (CP -> basin, cluster -> basin).
- **Validation**: 30 tests in `tests/test_sphere_viewer_selection.py` -- the
  three resolvers over raw/effective/saddle/hidden/absent inputs, falsy and
  non-dict guards (matching the `_selection_id` idiom in `params_schema`), plus
  structural asserts that `select_cp` and `select_cluster` call the resolvers and
  that each table binds its row with a tuple. `pytest`: **506 passed**. `ruff` /
  `mypy gba_topology2 tests` / `pre-commit --all-files` clean. Headless viewer
  smoke reaches `server.start`, so the three re-bound rows construct.
- **Open next**:
  1. There is no raw-CP table -- only effective CPs are tabulated
     (`all_cps_list.filter(c => c.is_effective)`). Clicking a RAW maximum lights
     its Extrema row and its cluster row, which answers "which maximum is this",
     but there is no CP row to light.
  2. A multi-member cluster containing several extrema resolves to the first
     basin-center match, i.e. the highest-value one, since the list is sorted by
     value descending. Arbitrary but stable; revisit if it misleads.
  3. `select_catastrophe` does not resolve a basin center. A catastrophe is a
     constellation, not an extremum, so there is nothing to light -- confirm that
     is the desired behaviour rather than an omission.

### [Entry 046] Step 3: Cone Captures the Target, Disc Blocks Only Obstacles

- **Objective**: On Pd_20K shape index, maxima surrounded by three saddles claimed
  only 1-2 connections. MAX_7 (3) and MAX_4 (1) are symmetric partners, so the
  pattern demanded identical answers and did not get them.
- **Root cause: the disc gate applied to the TARGET was tuned backwards.** Entry
  044 made the rule "claim the first CP whose excision disc the ray enters", and
  applied that to maxima as well as obstacles. Every raw maximum's excision
  radius is exactly ONE MESH PITCH (1.61 deg at pitch 1.6058), while the
  legitimate rays into a 3-saddle maximum arrive at cross-tracks
  `0.0, 1.35, 1.42` (MAX_7) and `0.0, 2.00, 1.98` (MAX_4). A gate at 1.61 sits
  INSIDE that band, so it accepted all three of MAX_7's rays and rejected two of
  MAX_4's. Identical geometry, different answers, decided by measurement noise.
  Connector counts across the eight raw maxima, MAX_0..MAX_7: **1, 2, 2, 3,
  1, 2, 2, 3** -- two identical symmetry classes of four, each spanning 1 to 3.
  After the fix: **3, 3, 3, 3, 3, 3, 3, 3**.
- **The fix separates the two roles.** Walk the candidates in increasing
  along-track and stop at the first decisive one:
  - a **maximum** ends the walk and the branch CONNECTS. Captured by the CONE,
    which is the directional criterion the sector already encodes. The arc still
    stops at the excision boundary where the ray crosses it, and at the centre
    when the ray passes outside -- the drawing rule from Entry 044 is preserved.
  - a **minimum or saddle** ends the walk as an OCCLUSION only if the great
    circle enters its disc (`|cross| <= rho`). The disc test is retained exactly
    where Entry 044 needed it: MC18's near-miss minimum at cross 7.4 with a
    1.61 deg disc is not an obstacle.
- **A second, independent bug surfaced on the way, and the repo had already
  solved it once.** The 8 occlusions on this snapshot were corridor MS13
  blocking rays at cross 3.4. An excision boundary around an elongated set is a
  fitted ELLIPSE: MS13 has a = 11.35, b = 2.43, aspect 10.6. Using the semi-major
  as a disc inflates the transverse reach by the aspect ratio, so the corridor
  occluded a ray passing 3.58 deg off its axis (3.58 < 11.35, but 3.58 > 2.78).
  `string_aggregator.cluster_directional_support_radius` exists for exactly this
  and its docstring names the failure: an isotropic radius "turns a long thin
  corridor into a disk of radius a and lets it swallow compact fragments sitting
  far off to the side". Extracted the ellipse math to
  `max_connection.directional_support_radius` (pure, no MicroCluster dependency)
  and added `ConnectionTarget.radius_toward()`.
- **The pool now carries the ellipse, not a scalar.** `DiscGeometry`
  (angular/semi-major/semi-minor/principal axis) replaces the two
  `ring_radius_by_*` maps with `disc_by_vertex` / `disc_by_id`, built by
  `disc_geometry_from_cluster`. Catastrophes are isotropic rings, so their
  geometry round-trips with a == b. `basin_centers_to_targets` is the single
  implementation of the conversion; `build_connection_pool` delegates to it.
- **Validation on `Pd_20K_atom1_params_shape.json`** (shape index, k_pitch 2.0,
  k_bridge 2.25, 96 ascent branches, pool 98 visible CPs):
  - **all 8 raw maxima now claim exactly 3 connectors** -- MAX_0..MAX_7 = 3 each,
    from 1, 2, 2, 3, 1, 2, 2, 3. Spread over all 26 maxima: 3-4.
  - verdicts 96 connected, 0 occluded, 0 no target (was 88 / 8 / 0).
  - mean separation 20.07 deg, max 32.99 -- connections stay local.
- **Regressions held**: curvedness stays 72 connected / 0 occluded and MC18
  branch 0 still claims E-MAX_MC38 (entry 33.79, cross -0.39). ethene4 atom 1,
  the off-origin case, reproduces 35/35.
- **Validation**: 40 tests in `tests/test_max_connection.py` (was 33) --
  `directional_support_radius` along-axis vs transverse; a target captured at
  cross 2.0 with a 1.61 deg disc; the symmetry fixture driving rays at cross
  0.0 / 1.4 / 2.0 and asserting all three connect (verified NON-VACUOUS: the old
  disc gate rejects the 2.0 case); an obstacle still needing its disc entered;
  arc stops at the boundary when hit and at the centre when missed; pool
  carrying DiscGeometry. `pytest`: **513 passed**. `ruff` /
  `mypy gba_topology2 tests` / `pre-commit --all-files` clean. Headless viewer
  smoke reaches `server.start`.
- **Process note**: I diagnosed this twice before measuring. Entry 044's
  "occlusion fires as designed" was the design being wrong, and my first fix
  (directional radius) was correct but did not move a single connector count --
  caught only by re-running the counts, not the occlusion total. The metric that
  found it was per-maximum connector counts across a symmetric set, which is the
  test for any rule with a geometric threshold: **symmetry is a free oracle.**
- **Open next**:
  1. `CONE_PITCH = 6.0` is now the sole directional gate for targets, so it is
     load-bearing where the disc used to share the job. Calibrated on Pd_20K
     (max observed cross-track 7.82 deg = 4.9 pitches). Untested on a coarser
     mesh or a sparse-maxima field.
  2. The 1.61 deg = one-pitch excision radius for raw maxima is a property of
     `compute_micro_clusters`, not of this rule. Worth checking it is intended.
  3. Step 4's integrator replaces the straight ray; the cone becomes the
     separatrix's own sector.

### [Entry 047] Step 3 Final Connector: One Great Circle, Centre to Centre

- **Objective**: The connectors bent sharply where they met the maximum's
  excision boundary. The search had found the two endpoints it needed; the drawn
  path should be the single geodesic between them, not the three-segment route
  the search travelled.
- **The change**: `arc_points` is now `_arc_between(centre_unit, hit.position, ...)`,
  the shortest geodesic from the saddle cluster centre to the maximum cluster
  centre. The previous `_arc_to_disc` (leg 1 along the launch ray to the boundary
  contact, leg 2 a geodesic from contact to centre) is deleted; it had exactly
  one caller. Gold is unchanged -- the viewer already colours the arc with
  `MAX_ARC_RGB` and iterates `arc_points` without inspecting its structure, so
  the viewer needed no change at all.
- **The reframing, which is the real content of this entry**: the port, the
  launch bearing, the boundary crossing and the cone are a SOLVER whose job is to
  find two endpoints. They are not part of the connector. `along_track_deg` keeps
  recording the boundary entry as the ranking key and as "how far the ray
  travelled before it reached the basin", but it is no longer a vertex of the
  drawn line. Everything before this was the means to find the endpoints.
- **Narrowing `centre_unit`**: it is `unit(seed.saddle_position)` and mypy saw a
  `None`. It is provably non-None at the call site -- `connection_ray` returns
  `None` exactly when the centre is at the nucleus, and the `ray is None` branch
  already continues -- so the guard is now `if ray is None or centre_unit is
  None`, which is the same degenerate seed, not a new condition. A guard rather
  than an `assert`, so a degenerate seed degrades to `no_target` as it did before.
- **Validation that the RULE is untouched (the load-bearing check).** A/B against
  HEAD by stashing the change and re-running the connect stage on
  `Pd_20K_atom1_params_shape.json`:
  - `connection_stats` drift caused by this change: **NONE**.
  - all **96 of 96** claims identical in target, status, `along_track_deg` and
    `separation_deg`.
  This is a rendering change. If any statistic had moved, the search would have
  been disturbed.
- **Validation that the GEOMETRY did change (the positive marker).** A clean
  refactor can be a silent no-op, so the arcs were measured directly:
  - **every one of the 96 arcs starts at the saddle centre**: 64 bit-exact to
    1e-9 deg, the remaining 32 within **1.2e-6 deg**, which is float64
    round-trip noise through `acos` near 1.0. The offsets are spread across both
    raw (14) and effective (18) saddles, so they are not a systematic frame
    error -- a real offset would be one-sided and far larger.
  - **worst torsion over all 96 arcs: 1.21e-13 A** -- every sample lies in one
    plane through the nucleus, i.e. each is a true great circle.
  - **worst kink over all 96 arcs: 1.375 deg**, which is exactly the uniform
    sampling step (span / `ARC_SAMPLES`), so there is no bend anywhere.
  - The old two-leg path on the same fixture kinks **26.84 deg**. The new
    no-kink test's threshold is 1.875 deg, so it is strongly non-vacuous.
- **Tests** (`tests/test_max_connection.py`, 42, was 40):
  - `test_arc_starts_at_the_port_and_ends_at_the_centre` ->
    `test_arc_is_the_single_geodesic_between_the_two_centres` (first point is the
    pole, not the port).
  - **`test_arc_is_a_great_circle` re-added.** I dropped it in Entry 046 because
    the two-leg path legitimately was not a single great circle; the property is
    back and is now the direct assertion of this feature.
  - **`test_arc_has_no_kink_at_the_maximum_boundary` added** -- the MC20 symptom
    pinned as a regression, by measuring consecutive segment directions.
  - `test_arc_stops_at_the_boundary_...` / `..._closest_approach_...` renamed to
    `test_ranking_key_is_the_boundary_entry_...` / `..._is_the_centre_...`. They
    assert `along_track_deg`, which is unchanged; the old names claimed to
    describe arc geometry, which they no longer do.
- **Sidebar text corrected.** The Step 3 card still said the arc "runs along the
  launch direction to the boundary, then to the centre", and also still described
  the Entry 043 disc-captures-target rule. Both stale; it now states the cone
  captures targets, the disc blocks obstacles, and the connector is the single
  great circle centre to centre.
- **A note on the reported case.** `MC20` does not exist in
  `Pd_20K_atom1_params_shape.json`: its 26 visible maxima are `MAX_0..MAX_7`,
  `E-MAX_MC1..MC8` and `E-MAX_MC105..MC114`. The maxima with exactly four
  connectors are the 18 `E-MAX` sets. The sharp-turn symptom was therefore
  verified on `E-MAX_MC109` and `E-MAX_MC1` (4 connectors each, kink 0.5-0.67 deg
  = the sampling step) and across all 96 arcs, not on MC20 itself. If MC20 is
  from a different field or atom, re-check it in the viewer.
- **Validation**: `pytest` **515 passed**. `ruff` / `mypy gba_topology2 tests` /
  `pre-commit --all-files` clean. Headless viewer smoke reaches `server.start`.
  Curvedness and ethene4 reproduce as in Entry 046 (their drifts are Entry 046's,
  recorded in the snapshots written before it).
- **Open next**:
  1. `MaxConnection.arc_points` is now fully derivable from the two centres. Kept
     because it is the viewer's only geometry contract and keeps the viewer from
     needing the sphere radius and nucleus; revisit if the record grows again.
  2. Occluded branches draw nothing. A faint stub to the blocker would make "why
     is there no arc here" answerable on screen. Deferred to Step 4.
  3. Step 4 replaces the straight connector with the traced separatrix. The
     endpoints found here are its boundary conditions, and the solver's internal
     path (port -> boundary entry) is the natural seed for that integration.

### [Entry 048] Step 4: Gradient Ascent Fallback for Incomplete Branches

- **Objective**: The Step 3 connector is a straight ray, so it finds only a
  maximum FORWARD of the port. On the organic fields (ethene3) the separatrix
  leaves a saddle, curves around the sphere, and arrives at a maximum that sits
  BEHIND the launch direction. The user's diagnosis: designate launch points
  complete (a terminal maximum was found by the ray) or incomplete, and for the
  latter, follow the field gradient uphill to the maximum. Two saddles' uphill
  directions wrap all the way around to connect one maximum from two sides.
- **Is the two-stage approach sound?** Yes, and the reason is that the two stages
  answer different questions. The ray is a straight-line rule with a bounded
  reach (cap 89 deg, cone 13 deg); the gradient is the field's own statement of
  where the ridge goes. They agree wherever the ray succeeds (a dense metallic
  field: every branch resolves geodesically), and the gradient is consulted only
  where the ray reaches nothing. The plan was confirmed with the user before
  building.
- **New module `separatrix_tracer.py`** (Principle 1: a distinct algorithm gets a
  dedicated module). `trace_uphill` integrates the surface gradient by central
  differences of the IDW interpolant: at each point, an orthonormal tangent frame
  `(t1, t2)`, `g1 = f(p+h.t1) - f(p-h.t1)`, `g2` likewise, step along
  `unit(g1.t1 + g2.t2)` by one mesh pitch. Constants: `TRACE_STEP_PITCH = 1.0`,
  `MAX_TRACE_ARC_PITCH = 400.0` (~891 deg, more than a great circle, so a path
  that exhausts it is spiralling and the answer is "unresolved"),
  `STALL_PATIENCE = 6` consecutive non-rising steps. Statuses: arrived / stalled /
  budget / no_start.
- **The arrival-before-stall ordering is load-bearing.** The gradient vanishes at
  a maximum, so reaching the top and going flat is success, not a stall. The
  arrival test runs first in the loop. `centred_targets` maps the world-frame
  pool to centred unit positions before the tracer sees it, because
  `radius_toward` measures from the target's OWN position -- the recurring frame
  trap, which Pd_20K hides at the origin and ethene3 exposes at 1.2618 A.
- **Integration into `connect_seed_to_maxima`** (Phase C). The function gains
  optional `mesh_pts` / `f_vals` / `find_k_nearest`. For a branch the ray leaves
  as `no_target`, `_gradient_connection` climbs from `branch.launch_position`
  (the port, never the saddle centre -- the gradient is zero there). On arrival it
  emits `method="gradient"` with the traced polyline as `arc_points`. Occluded
  branches are NOT rescued: an occlusion is a resolved verdict, and climbing the
  field from the port would contradict the geodesic rule that produced it.
- **`method` and `failure_gate` on the record** (Phase A). `method` is
  geodesic|gradient. `failure_gate` names the test that rejected the maxima:
  behind / cone / cap / occluded / empty. This makes the two-stage design
  auditable -- "the ridge curves" (behind/cone) and "the ridge is merely long"
  (cap) are different statements about the field. `_classify_gate` scores every
  pool maximum against the ray directly (the pre-filtered candidate list cannot be
  used; the maxima it excludes are exactly the ones being diagnosed).
- **A field is not a control** (Principle 4). `mesh_pts`/`f_vals` are data the
  pipeline already has, not tunables. The fallback is inert when they are omitted,
  which is the unit-test contract and the Pd_20K no-op contract.
- **The regression invariant, verified byte-identical.** Pd_20K has zero
  incomplete branches, so the fallback must be a no-op there. A/B against HEAD
  (stash the change, re-run connect on both Pd_20K snapshots):
  - shape index: **96 of 96** connections identical in status, target,
    along/cross/separation. `n_gradient = 0`, `n_geodesic = 96`.
  - curvedness: **52 of 52** identical. `n_gradient = 0`.
  The connection dicts compare equal as tuples; the fallback changed nothing.
- **ethene3, the field the feature is for** (gradient fallback on):
  - Electron Density: 6 -> **10 of 10** connected (4 gradient, all gate=behind).
  - curvedness: 6 -> **10 of 18** connected (4 gradient reaching MAX_4864 at
    110-126 deg separation, traced along-track 115-124 deg -- past the 89 deg cap
    the ray obeys). The 4 remaining no_target are gate=behind with the climb
    stalling; the 4 occluded stay occluded (correct).
  - shape index: **0 of 4, unchanged, and this is the honest result.** The gate is
    cone (the single maximum MAX_4864 sits at cross +/-44.9 deg, outside the
    13.4 deg sector). The fallback runs and correctly stalls: the basin test (a
    prior measurement) showed 96/400 random sphere samples reach MAX_4864 but
    0/4 saddle ports do, at every stall threshold. The ports lie OUTSIDE the
    spike's ascent basin -- the four spikes are absorbed into a saddle corridor
    (MS1) and the field between them is a 10081-vertex plateau. This is a
    reduction/topology finding, not a connector bug: no connector rule can reach a
    maximum the field does not drain to from the saddle.
- **Viewer** (Phase D). A second actor `gradient_arc_poly/actor` in cyan
  (`GRADIENT_ARC_RGB`) draws the traced polylines; the gold actor draws the great
  circles. `_emit_max_arcs` splits by `conn.method` in one pass. The arcs table
  gained a "grad" chip and null guards on along/cross/sep (a gradient connector has
  no cross-track; `.toFixed` on null throws in the browser). A stats line reports
  "N by gradient ascent (cyan), M by great circle (gold)".
- **Probe** `inspect_max_connections.py`: `--no-gradient` A/B flag, a method
  column (G flag), and a "methods" summary line with the gate breakdown.
- **A `direction_mode` bug caught by the reproduce check.** The fallback first
  wrote `direction_mode="gradient"`, which overwrote the ray definition and made
  ethene3 shape index drift 61/62. `direction_mode` names the RAY (port_tangent /
  chord) and is carried through from the caller; `method` is what marks the
  fallback. After the fix: ethene3 shape index reproduces **62/62 PASS**.
- **Tests**: `tests/test_separatrix_tracer.py` (13) -- single peak, nearest of
  two, a curved ridge (two peaks, off-symmetry port; asserts the path bows >5 deg
  off the great circle), flat plateau stalls, tiny budget, degenerate start,
  visited non-maximum discs, world-frame emission about an offset nucleus,
  `centred_targets` map/drop/identity, constants, frozen outcome.
  `tests/test_max_connection.py` (47, +5) -- the behind-ray branch resolves by
  gradient with gate=behind, a stall keeps the gate, occluded is not rescued,
  no-field is inert, the summary counts the split.
- **Validation**: `pytest` **532 passed, 1 skipped**. `ruff` / `mypy gba_topology2
  tests` clean. Headless viewer smoke reaches `server.start` on ethene3.
- **Open next**:
  1. The shape-index basin is a reduction question, not a connector one: should a
     saddle corridor that swallows maxima expose its constituent spikes as
     connection targets? Out of scope for Step 4.
  2. The traced path is drawn at full fidelity; a long spiral could be decimated
     for the scene. Not needed at these budgets.
  3. Occluded branches still draw nothing (Entry 047 open item 2).

### [Entry 049] Step 5 Reverted: the Scope Is Two Stages, Not Three

- **The user's correction.** Entry 049 (connector refinement + topology audit,
  commit `cd99bbd`) was not the step we were on. The agreed scope is exactly two:
  great-circle identification, then gradient following for the INCOMPLETE saddle
  launch seeds -- climb from the excision boundary to the highest point and STOP.
  The coincidence of that endpoint with a topologically identified maximum is a
  concern to be recorded, not a defect to be corrected. Refinement is a separate
  future step, to be done a different way.
- **The revert.** `git revert --no-edit cd99bbd` -> `b48b6cf`. The refinement
  module, its 15 tests, the `refine_connectors` control, the promotion machinery
  and the Entry 049 journal text are all removed from the tree and all recoverable
  from `cd99bbd`. Revert, not reset: the record of what was tried, measured and
  deferred stays in the branch history, which is how this journal works.
- **What the measurements said, preserved for the future step.** Pd_20K shape
  index: 96/96 confirmed, byte-identical, mean endpoint deviation 2.05 deg.
  Pd_20K curvedness: 1 genuine reroute (E-SAD_MC29 br1 claimed E-MAX_MC42 at
  along 13.6 deg; the ridge drains 1.6 deg to MAX_7412, a raw maximum the
  reduction leaked inside the saddle's ring) and 2 corner-sharings. ethene3 shape
  index: 4/4 resolved by promoting the mirror pair {MAX_10052, MAX_5}. The
  tolerance finding stands: 2 pitches = step + disc, and the symmetry oracle
  rejects 1 pitch because it promotes one of a mirror pair.
- **Schema: v4 admitted as LEGACY, not removed.** Snapshots saved while the
  refinement control existed are on disk (the user's
  `ethene3_atom1_params_curv.json` is one). `PARAMS_SCHEMA_VERSION` stays 3;
  `PARAM_LEGACY_SCHEMA_VERSIONS = {2, 4}`. The loader reads the keys it knows and
  ignores `refine_connectors`, so an observation made with refinement on still
  opens and reproduces the pipeline it describes. `test_load_rejects_unknown_schema_version`
  now probes 99, not `PARAMS_SCHEMA_VERSION + 1`, because +1 is 4 and 4 is valid.

### [Entry 050] Stalled Climbs Keep Their Path; Select a Saddle to See It

- **Objective**: the user's report, on `ethene3_atom1_params_curv.json` -- "MC18
  remains incomplete. Every saddle launch point should show a connector that
  follows a gradient path uphill to its highest point and then terminates. It may
  be that I can't see these paths because they are not associated with a maximum."
  Both halves are correct, and they are one bug with two causes.
- **Cause 1: the path was thrown away.** Step 4's `_gradient_connection` returned
  `_no_target(...)` when the climb reached no maximum, discarding `outcome.path`.
  The record said `no_target` with an empty `arc_points`, so there was literally
  nothing to draw. The ascent had run, had travelled 70-120 deg of ridge, and its
  result was dropped on the floor.
- **Cause 2: the overlay is keyed by maximum.** `_emit_max_arcs(basin_id)` selects
  connections with `conn.max_id == basin_id`. A stalled climb has `max_id = None`,
  so it is invisible from EVERY maximum's view -- the operator cannot reach it by
  clicking anything. The connection table has two ends, and only one of them was
  selectable.
- **The fix, part 1.** A stalled climb's record now carries the ridge it travelled:
  `arc_points = outcome.path`, `stall_point = outcome.path[-1]` (new field, the
  highest point the ascent found), `along_track_deg = outcome.arc_length_deg`, and
  `method = "gradient"` -- the geometry was traced, so it draws in cyan. `method`
  names how the geometry was PRODUCED; `status` names what it FOUND. The two are
  independent, and conflating them is what made the record look like nothing
  happened.
- **The fix, part 2: select a saddle.** `_emit_max_arcs(basin_id, saddle_id=None)`
  filters by `conn.saddle_id` when a saddle is named, so the view shows every
  ascent branch LEAVING it, arriving and unresolved alike. `state.selected_saddle_id`
  is the new key, mutually exclusive with `selected_basin_center`: selecting a
  saddle clears the maximum (and vice versa), so the scene and the drawer table
  always agree about which end is being inspected.
- **The join is exact, not geometric.** `build_launch_requests` names a raw
  saddle's seed `SAD_<vertex_id>` and an effective saddle's seed its own
  `id_label` (`E-SAD_MC18`). The CP UI dicts carry `id_label` in the same space,
  so `saddle_seed_id_for_cp` is a set membership test against the seeded ids --
  no tolerance, no nearest-neighbour. `saddle_seed_id_for_cluster` covers the
  picker's other path (cluster centroids are tested before CP glyphs, so a saddle
  glyph click usually lands on `select_cluster`): `E-SAD_<cluster id>` first, then
  any saddle member's `SAD_<vertex>`. Both are module-level pure functions of the
  flattened UI dicts, so they are testable without trame, and both are exported.
- **The arcs table gained a Target column**: the claimed maximum, a `stall` chip
  for a climb that topped out on non-maximum ground, or an em-dash for a branch
  with no geometry at all. The hint text now names both ends of the overlay.
- **MC18, verified.** In the user's snapshot MC18 is the cluster containing vertex
  8487, seeded as `SAD_8487` (raw, un-reduced). Its branch 0 is gate=behind,
  status=no_target, method=gradient, stall_point set -- a cyan path from the port
  to the highest point reachable uphill, terminating in mid-sphere. Selecting
  `SAD_8487` (glyph, cluster row, or CP row) draws it. `SAD_8400` (MC17, the
  mirror partner) behaves identically: 2 stalled branches, symmetric.
- **Tests** (539, was 533):
  - `test_gradient_fallback_records_gate_when_it_stalls` REWRITTEN. It pinned the
    old contract (`arc_points == ()`), which is exactly the behaviour the user is
    reporting as a bug. It now asserts the path survives, the stall point is the
    last vertex and lies on the sphere, `method == "gradient"`, and the gate is
    preserved. The fixture uses a south-pole peak so the climb travels, making the
    path-length assertion non-vacuous.
  - 6 new resolver tests: raw/effective saddle resolve; extrema, hidden, unseeded
    and None resolve to None; cluster resolves via `E-SAD_<id>` and via a raw
    member; a maximum's cluster resolves to None.
- **Validation**:
  - ethene3 curvedness reproduces **70/70 PASS** against the user's v4 snapshot,
    which loads cleanly as legacy.
  - **Pd_20K byte-identical** on both fields (96/96 shape, 52/52 curvedness,
    `stall_point` included in the compared key). Zero stalled branches there, so
    the change is a provable no-op -- the same contract Steps 4 and 5 were gated on.
  - `pytest` 539 passed. ruff / mypy / pre-commit clean. Headless viewer smoke on
    ethene3 curvedness with the v4 snapshot restored.
- **Open next**:
  1. A stalled climb's endpoint is drawn and named, and the disagreement with the
     topology is visible. What to DO about it is the separate refinement step, by
     the user's design, not this one's.
  2. Occluded branches still draw nothing (Entry 047 open item 2).
  3. The saddle view shows ascent branches only. Descent branches are in the seed
     and are not connected to anything -- a minima-keyed view would be the mirror
     of this one.

### [Entry 051] The Ascent Transits an Excision Region; It May Not End Inside One

- **Objective**: the user's report, on `ethene3_atom1_params.json` -- "the uphill
  gradients from e.g. MS4 terminate inside the excision boundary of MC2. As this
  region is noisy, we should not allow termination inside an excision region. The
  gradient should continue through this region and pick up on the other side and
  continue uphill to near the MC12 maxima."
- **Confirmed by measurement, and MC12 is the spike.** MC12 is the cluster
  containing vertex 4864, i.e. `MAX_4864` -- the designated maximum is the
  cluster's own name, so the user's "near the MC12 maxima" is the spike exactly.
  All four stalled branches terminate inside an excision disc: MS3 br0 in MC1
  (6.6 deg from centre, ring 8.9), MS3 br1 in MC2 (4.5 / 6.1), MS4 br0 in MC1
  (5.0 / 8.9), MS4 br1 in MC2 (1.2 / 6.1). MC1 and MC2 are the two pole clusters,
  and they are ANNIHILATED (net chi 0): they reduce to nothing, are absent from
  the critical-point pool, and their interiors are the noisiest ground on the
  atom. The field along the great circle from MC2's centre toward the spike rises
  monotonically 0.0059 -> 0.0063 -> 0.0073 -> 0.0094 -> 0.0141 -> 0.0372 ->
  0.4001: the ridge is there, and the ascent stopped in the noise before it could
  read it.
- **The rule.** When the climb terminates inside an excision region, continue in a
  straight line THROUGH it -- along the great circle the path was already on, from
  the last point outside the region, until clear -- and resume the ascent from
  there. The transit is deliberately NOT a gradient step: inside the boundary the
  gradient IS the noise the boundary exists to exclude, so the only defensible
  direction is the one the path already had. This is the same principle as
  Entry 044's port launch, applied to the far end of the climb.
- **Two exclusions, both load-bearing.** The region containing the LAUNCH PORT is
  never transited: the port is ON the saddle's excision boundary by construction,
  and a saddle's boundary overlaps the clusters it was reduced from, so without
  the exclusion every branch transits at step 0 and none ever climbs. Regions
  already transited are excluded too, so the path cannot oscillate across the same
  disc. `TRACE_MAX_TRANSITS = 4` bounds the total.
- **The budget is per climb SEGMENT, not per trace.** The first cut shared one
  891 deg allowance across the whole trace, and the measurement caught it: all
  four branches transited correctly and arrived NOWHERE, because the pre-transit
  spiral exhausted the budget before the resumed climb started. A branch that
  spirals out its allowance, transits, and has nothing left to climb with makes
  the transit look useless. The budget bounds each climb (what it means to be a
  cap on how far an ascent may wander); the transit COUNT bounds the total work.
- **The regions are not the pool.** `excision_regions` is built from EVERY
  micro-cluster, including the annihilated ones. The pool answers "what can I
  connect to"; the regions answer "where may I not end". MC1/MC2 are invisible to
  the first question and essential to the second, which is why the builder takes
  clusters and not targets.
- **Frame contract.** Regions carry CENTRED unit-radial centres, the frame the
  ascent integrates in. `excision_regions(clusters, center)` maps world positions
  in with `to_local` and normalises; a zero-radius cluster (a lone vertex) has no
  interior to avoid and is dropped.
- **Measured result, ethene3 shape index: 0/4 -> 2/4 connected.** The two that
  arrive are `E-SAD_MS3 br1` and `E-SAD_MS4 br1` -- MIRROR PARTNERS, and they
  behave identically (transit MC2, resume, arrive at MAX_4864, arcs 982 and 987
  deg). The symmetry oracle passes. The two that do not transit MC1 and exit on
  the far side of the spike's basin: the field there genuinely does not drain to
  MAX_4864, and the rule reports that honestly rather than forcing a connection.
  All four record `transited_ids`, so the straight segment over untrustworthy
  ground is attributable.
- **The rule is a no-op where it should be.** Pd_20K: 0 transits on both fields,
  table byte-identical to HEAD (96/96 shape, 52/52 curvedness). ethene3 Electron
  Density: 10/10, 0 transits. ethene3 curvedness: 8/10, 0 transits -- its two
  stalls are on OPEN ground, not inside a disc, so the rule correctly leaves them
  alone. That is the discrimination the user asked for: the rule fires on noise,
  not on genuine flatness.
- **Plumbing.** `trace_uphill` gains `excision_regions` (default empty = the pure
  Step 4 ascent, which is what the existing 13 tracer tests pin) and
  `TraceOutcome.transited_ids`. `connect_seed_to_maxima` threads both through and
  records `MaxConnection.transited_ids`; `summarize_connections` counts
  `n_transited` and lists `transited_regions`. Viewer and diagnose pass
  `excision_regions(clusters)`; the probe gains `--no-transit` for the A/B and a
  `T` flag column. The arcs table shows a `transit` chip and the stats line names
  the regions carried across.
- **Tests** (544, was 539): the transit pair (same field, same port, same
  integrator, ONLY the rule differing -- so the two assertions together ARE the
  measurement of what the rule buys), the port's own region is never transited,
  the transit count is bounded on a field ringed by regions, the builder maps
  clusters to unit centres and drops degenerate ones, and the new outcome field
  defaults empty. The fixture uses two NARROW peaks (15 deg, 40 deg apart) so the
  bump is a genuine local maximum with a saddle between it and the pole -- the
  first version used wide peaks, they merged into one basin, and the test passed
  vacuously by arriving without the rule.
- **Validation**: `pytest` 544 passed. ruff / mypy / pre-commit clean. ethene3
  shape index reproduces 70/70 PASS. Headless viewer smoke on ethene3.
- **Open next**:
  1. The transit is a straight line, so the drawn connector has a segment that is
     not field-derived. It is recorded in `transited_ids`; a distinct rendering
     (dashed, or a different lift) would make it visible in the scene.
  2. The two MC1 branches exit on the far side of the spike's basin. Whether the
     transit heading should be aimed at the disc centre (shortest exit) versus
     kept along the path's heading (the choice made here, which is the
     continuation the user described) is worth an A/B on more fields.
  3. Occluded branches still draw nothing (Entry 047 open item 2).

### [Entry 052] Resume at the Boundary's Field Maximum, Not the Spiral's Heading

- **Objective**: the user's correction to Entry 051 -- "we took care of the problem
  on one side but evidently we did not take care of the general case. Now the
  uphill gradient from MS4 terminates inside the excision boundary of MC1. When
  entering an excision region we should look at the value of the field around its
  boundary and pick up the gradient at the point where the field achieves its
  largest value on the boundary. Much like we did for saddle seed points."
- **The user was right, and the diagnosis was precise.** Entry 051 resolved 2 of
  4 ethene3 shape-index branches: the two that transit MC2 arrived, the two that
  transit MC1 exited 84.6 deg from the spike, on the far side of its basin, and
  stalled. The rule worked where the geometry happened to cooperate.
- **Why the heading rule is wrong in principle, not just in outcome.** Entry 051
  exits along "the great circle the path was already on". But the path that
  terminated is a SPIRAL that ran out its budget across a plateau -- its final
  heading is set by the very noise the boundary exists to exclude. So the exit
  point is noise-determined, and the rule was using a noise-derived quantity to
  escape noise. The boundary's field maximum is a MEASUREMENT of trustworthy
  ground: the ring is the last reliable circle before the degenerate core, and its
  high side is the direction the region is draining from.
- **The rule is the saddle-seed method, applied to the far end of the climb.**
  Profile the ring, take the extremum, launch from there -- exactly
  `extract_boundary_ports` for launch seeds. The symmetry is the point: a saddle's
  ascent begins at a boundary port chosen from the field, and it should end at one
  chosen the same way.
- **`RING_SAMPLES = 16` is load-bearing, and the sweep proves it.** MC1's ring
  field spans 0.00027 -- 0.05 percent of the field's range -- so the argmax is
  picking a direction out of near-flat noise. Sampled at 16 the argmax reaches the
  spike; at 24, 32, 48, 72, 96 and 144 it does not (all land 83-84 deg away and
  spiral). MC2's ring is far better conditioned and arrives at every count. The
  coarse sample count is not a performance knob: it is the anti-noise filter, and
  `excision_launch_seeds` documents exactly that job -- "finer azimuthal sampling
  resolves IDW interpolation noise as spurious extrema and inflates the measured
  k". Reusing the constant reuses the reason for it.
- **Result: ethene3 shape index 2/4 -> 4/4 connected.** All four branches resume
  from a boundary argmax and arrive at MAX_4864. The two MC1 branches (E-SAD_MS3
  br0, E-SAD_MS4 br0) now behave like their MC2 partners, and the mirror pairs
  (MS3/MS4 br0, MS3/MS4 br1) give arcs of 990/989 and 979/976 deg -- agreement to
  1 deg, the free oracle passing on all four.
- **The bridge is drawn, and named.** The connector stays continuous: a geodesic
  from the stall point to the resume point, sampled at the step. It crosses the
  noisy interior, so it is a carried line, not a measured one, and
  `transited_ids` records which region it belongs to. An antipodal stall/resume
  pair has no unique great circle and the path teleports, which is honest about
  the gap.
- **No-ops, as required.** Pd_20K: 0 transits on both fields, table byte-identical
  to HEAD (96/96 shape, 52/52 curvedness). ethene3 Electron Density: 10/10, 0
  transits. ethene3 curvedness: 8/10, 0 transits -- its two stalls are on OPEN
  ground, not inside a disc, so the rule correctly leaves them alone.
- **Drift is confined to the connector layer.** Full-pipeline reproduce on ethene3
  shape index: 68/72, and the 4 drifts are exactly
  `connection_stats.{n_connected,n_gradient,n_no_target,n_unresolved}`, all the
  expected 2 -> 4. Geometry, morse, reduction and launch all match, so nothing
  upstream moved.
- **Tests** (546, was 544):
  - `test_resume_point_is_the_boundary_field_maximum` pins the MECHANISM, not just
    the outcome: the resumed path must pass through the ring argmax computed
    independently. Non-vacuity is asserted too -- the antipode of entry (where the
    heading rule resumed) is >20 deg from the ring's high side, so the two rules
    are demonstrably different and the test cannot pass for the old reason.
  - `test_ring_sampling_is_the_coarse_convention` pins `RING_SAMPLES == 16`,
    because the constant is part of the rule.
  - The Entry 051 transit pair (arrives with regions, not without) still passes
    unchanged: it asserts the outcome, which the new rule also achieves.
- **Validation**: `pytest` 546 passed. ruff / mypy / pre-commit clean. Headless
  viewer smoke on ethene3. Probe `--no-transit` A/B: 4 connected vs 0.
- **Open next**:
  1. The bridge segment is drawn like the measured path. A distinct rendering
     (dashed, or a lower lift) would make the carried line visible as such.
  2. The ring argmax is a single sample of a near-flat profile on MC1. A
     prominence test (the saddle-seed `min_prominence_frac` convention) would let
     the record say "the boundary had no high side" rather than picking one.
  3. Descent branches have the same latent issue and no connector to show it on.

### [Entry 053] The Regions Are the VISIBLE Boundaries, Not the Raw Clusters

- **Objective**: the user's report, on the re-saved `ethene3_atom1_params.json`
  (k_pitch 2.2, k_bridge 3.0) -- "the ascending gradient paths from MS3 and MS4
  still terminate in the excision region MS1. Why? Fix."
- **What the measurement showed.** The two br1 branches arrive at MAX_4864 (they
  resume from MC2's boundary). The two br0 branches transit **MC10** and spiral
  out their budget, ending 99.0 deg from the spike. So the rule fired and the
  ascent was carried -- out of one noisy disc and immediately back into the same
  one.
- **The cause is nesting.** MC10 is a raw micro-cluster (1 member, ring 3.48 deg)
  that the reduction absorbed into the corridor **MS1** (ring 8.94 deg). MC10's
  centre sits 7.10 deg from MS1's centre, well inside MS1's 8.94 deg boundary. The
  regions were built from `ctx.clusters` -- every raw cluster -- so the ascent was
  resumed at MC10's tiny ring, which is deep inside the corridor's noisy ground.
  It climbs a few steps, goes flat, and stalls. The boundary it was told to escape
  is a subset of the boundary it is still inside.
- **The fix is the region SOURCE, not the rule.** Build the regions from the
  reduction units' `as_cluster` views -- the boundaries the operator can SEE, and
  exactly the source the connection pool uses. `as_cluster` exists for this
  purpose; its docstring says "so subroutines written against clusters work on
  units". Resuming at MS1's boundary puts the ascent on ground the corridor has
  not swallowed, and it arrives.
- **A/B on the user's snapshot, same field, same ports, same integrator:**
  - `excision_regions(clusters)`: **2/4** connected, transiting `['MC10','MC2']`.
  - `excision_regions([u.as_cluster for u in units])`: **4/4** connected,
    transiting `['MS1','MC2']`.
  The mirror pairs agree to 2 deg (MS3/MS4 br0: 995/993; br1: 981/979) -- the free
  oracle passing on all four, where before it passed on two.
- **Why this is the right invariant, not a patch.** The rule is "do not terminate
  in ground the pipeline declared untrustworthy". The declaration is made by the
  REDUCTION, which is what decides a region is a degenerate core. A raw cluster
  the reduction absorbed has no standing: it is not drawn, not in the pool, and
  not a boundary anyone can see. Using raw clusters let a hidden entity define an
  escape hatch that escaped nowhere. The regions must be the visible boundaries,
  which is the same set the pool is built from.
- **The annihilated entities are still covered.** MS1 and MC2 are annihilated
  (net chi 0) and absent from the pool, so invisible to the arrival test, yet
  their interiors are exactly the ground the ascent must not stop in. They ARE
  reduction units, so they are in the region set. The pool answers "what can I
  connect to"; the regions answer "where may I not end" -- and both are now keyed
  to the same visible entity set.
- **No-ops, as required.** Pd_20K: 0 transits on both fields, table
  byte-identical to HEAD (96/96 shape, 52/52 curvedness). ethene3 Electron
  Density and curvedness: transit ON and OFF give IDENTICAL verdicts (9/10 and
  10/18) -- no branch on those fields terminates inside a visible boundary, so
  the rule correctly does nothing. (Those counts differ from Entry 052's because
  the user re-saved the snapshot with k_pitch 2.2 / k_bridge 3.0, which changes
  the reduction upstream; the on/off equality is the relevant control.)
- **Tests** (548, was 546):
  - `test_a_nested_inner_region_traps_the_ascent_the_outer_one_frees_it` pins the
    MECHANISM: same field, same port, same integrator, same trap -- only the
    offered disc differs (6 deg inner vs 18 deg outer). The inner one transits and
    stalls; the outer one transits and arrives. Non-vacuity is asserted on both
    sides: `transited_ids` is set in BOTH runs, so the rule fires in both and the
    outcome difference is attributable to the region set alone.
  - `test_excision_regions_names_regions_by_the_visible_entity` pins id
    preservation, which is what makes `transited_ids` readable as corridor names.
- **Validation**: `pytest` 548 passed. ruff / mypy / pre-commit clean. Headless
  viewer smoke on ethene3 with the re-saved snapshot.
- **Open next**:
  1. The bridge segment is drawn like the measured path (Entry 052 open 1).
  2. A prominence test on the ring profile (Entry 052 open 2) -- MC10's ring
     spans 0.000024, so its argmax is pure noise. With the corridor fix it is no
     longer consulted, but a near-flat OUTER ring would have the same issue.
  3. Descent branches have the same latent issue and no connector to show it on.

### [Entry 054] The Arc Cap Is an Angular Reach, Not a Mesh Multiple

- **Objective**: the user's question, on `ethene3_atom1_params_V.json` (field V,
  k_pitch 2.6) -- "the saddle MC14 ascending launch seeds are directly aligned
  with the maximum MC13. Why weren't these points connected by the great circle
  mechanism before defaulting to gradient path following? Why, and can this be
  corrected?"
- **The answer was in the snapshot's own stats**: `n_gate_cap: 2`,
  `max_separation_deg: 178.013`. Both branches were rejected by the ARC-LENGTH
  CAP, and the probe's `--trace` says it exactly:
  `MAX_4864  maximum  entry 176.19  cross -0.01  in-sector (blank)`. The
  alignment is as good as it gets -- cross-track -0.01 and +0.02 deg, dead centre
  on the great circle -- and the ray found its target. It was then thrown out for
  being 176 deg away.
- **The cause is a units error, not a threshold error.** `MAX_ARC_PITCH = 40.0`
  is in MESH PITCHES, so it means a different reach on every mesh: 40 x 1.6058 =
  64.2 deg on Pd_20K, 40 x 2.2278 = 89.11 deg on ethene3. The cone is legitimately
  pitch-relative -- it asks "can the mesh resolve this structure", which IS a
  mesh-scale question. Reach is not: it is an angular property of the sphere.
  Encoding it in pitches made a hemisphere the limit, and the hemisphere was
  calibrated on the metal.
- **The provenance, from Entry 040**: "Measured on the saved Pd_20K snapshots, the
  largest saddle-to-first-hit separation is 55.2 deg (V) and 45.0 deg
  (curvedness), so the cap admits every observed connection while excluding the
  far side of the sphere." That is exactly the failure mode this project keeps
  hitting -- a constant tuned on the dense metallic fields, where saddle/maximum
  pairs are 12-45 deg apart, silently rejecting the near-antipodal pairs that
  organic fields produce. MC14 and MC13 are 178.01 deg apart: an atom's two sides.
- **The fix**: `MAX_ARC_DEG = 180.0`. That is the natural domain of a great-circle
  ray -- every geodesic from a point reaches every point, uniquely up to the
  antipode. Under the first-hit rule the cap is a safety net, not the
  discriminator: the nearer hit wins on along-track, and the CONE is what makes
  the claim directional. Widening it does not make the rule permissive, because
  the rule was never using the cap to choose.
- **A/B across every saved field, cap 89 deg -> 180 deg. Strictly improving, zero
  regressions.**
  - ethene3 V: **1/2 -> 2/2**, and both branches now resolve by GEODESIC
    (`n_geodesic 0 -> 2`, `n_gradient 1 -> 0`). The gradient fallback is no longer
    invoked on a case the straight ray handles perfectly.
  - ethene3 curvedness: 9/14 -> 10/14. The two new claims are at cross-track 0.27
    and -0.24 deg, and one of them the gradient had INDEPENDENTLY reached at the
    same maximum -- the two methods agreeing is the corroboration.
  - ethene3 Electron Density 10/10, Pd_20K shape 96/96, Pd_20K curvedness 52/52:
    all unchanged, byte-identical.
  - Max separation over all connections at 180 deg: 178.0 (V), 151.8 (curvedness),
    83.4 (ED), 45.0 (Pd). The far-side claims are the organic fields' genuine
    geometry, not noise.
- **Result on the user's case**: MC14 (`SAD_4953`) connects to MC13 (`MAX_4864`)
  on BOTH branches by great circle, cross-track -0.01 / +0.02 deg, and the pair is
  flagged MUTUAL (`M`) -- the maximum points back at the saddle, which is the
  reciprocal reading the flag was built to report.
- **Plumbing**: `connect_seed_to_maxima(max_arc_pitch=...)` ->
  `max_arc_deg=...`; the probe's `--max-arc` now takes degrees and its sector line
  prints `cap 180.00 deg` with no pitch conversion; the schema comment names
  `MAX_ARC_DEG`. The tracer's `MAX_TRACE_ARC_PITCH` is deliberately NOT changed:
  that is an integration budget (how far a climb may wander before the answer is
  "unresolved"), which is genuinely about the mesh scale, and it is a different
  quantity from how far a ray may claim.
- **Tests** (549, was 548):
  - `test_arc_cap_produces_no_target` moved to `max_arc_deg=10.0`; the comment
    notes the test now means the same thing on every mesh.
  - `test_the_cap_is_angular_not_mesh_relative` added: the same geometry at a
    Pd_20K pitch and an ethene3 pitch must give the same verdict (both connect a
    150 deg maximum), and the old 89.11 deg reach must reject it. The second
    assertion is the non-vacuity check -- it pins the bug the test exists to
    prevent.
- **Validation**: `pytest` 549 passed. ruff / mypy / pre-commit clean. Drift on
  the V snapshot is 9 keys, ALL inside `connection_stats`
  (`n_connected`, `n_gate_cap`, `n_geodesic`, `n_gradient`, `n_mutual`,
  `n_no_target`, `n_unresolved`, mean/max cross-track), all the expected
  1->2 geodesic recovery; geometry, morse, reduction and launch all match, so
  nothing upstream moved. Headless viewer smoke on the V snapshot.
- **Open next**:
  1. The cap is now non-binding on every field measured. It is retained as the
     safety net its original docstring called it, and as the knob that makes the
     rule's reach explicit; deleting it would hide the assumption.
  2. The general lesson is worth a standing check: every pitch-relative constant
     should be asked "is this a mesh-scale question (cone: yes) or a sphere-scale
     one (reach: no)?" `CONE_PITCH` is the former and stays.
  3. Bridge segment rendering, ring prominence, descent branches (Entries
     052/053).

### [Entry 055] Antipodal Connectors Collapse to One Line; the Commitment Audit

- **Objective**: the user's report, on `ethene3_atom1_params_V.json` (field V,
  k_pitch 2.0) -- "MC14 (maxima) does not connect to both sides of MC15. They are
  antipodal and there is only one connection instead of two. Add a check to
  confirm that an uphill launch seed is never left uncommitted."
- **The user's eye was right and the table was lying.** Both branches WERE
  connected -- `n_connected: 2`, `n_no_target: 0`, cross-track -0.01 and +0.02 deg.
  Measured directly: the two drawn polylines are **identical point-for-point, max
  deviation 0.000e+00 deg**. The scene shows one line because the code produces one
  line. The connection table is complete; the drawing is not.
- **The cause is in the call site, and it is not antipodal-specific.**
  `arc_points=_arc_between(centre_unit, hit.position, ...)` takes **no branch
  argument**. Two branches of one saddle that claim one maximum have the same two
  endpoints, so they get the same polyline by construction. The reason it is
  invisible away from the antipode is that it cannot happen there: the cone is
  +/-13.37 deg and a k=2 saddle's two ascent ports are 180 deg apart, so two
  branches of one saddle can never both claim one maximum. The antipode is the
  single exception -- and it is also where the shortest geodesic stops being
  unique. Every great circle through the pair has the same length, so
  centre-to-target cannot say WHICH one the branch travelled, and the two
  genuinely different paths (one over each pole) merge.
- **The fix.** Inside a band of `ANTIPODAL_BAND_RING_MULTIPLE = 2.0` ring angles
  around 180 deg, build the arc on the great circle through the branch's OWN
  launch port -- the bearing the search used, well conditioned because the port is
  a full ring angle from the centre -- and sweep it in the port's direction. The
  band is 2 x ring because that is the smallest separation at which two branches'
  port bearings can differ by more than the ring. Measured: separation 178.0133,
  deficit 1.99 deg, band 2 x 2.23 = 4.46 deg, so the pair is inside it. The two
  arcs now separate by 38.16 deg.
- **The endpoint convention inside the band, and why it must differ.** The target
  is not on the branch's great circle, so the arc runs to the target's CLOSEST
  APPROACH on it. That point is at most a ring angle from the centre, i.e. inside
  the maximum's excision disc -- exactly where a connector is allowed to stop
  (Entry 044: the disc IS the basin). Forcing it onto the centre would need a
  perpendicular final leg and a right-angle kink, which is the geometry Entry 047
  removed.
- **The audit the user asked for.** `summarize_connections` gains
  `n_uncommitted` / `uncommitted_branches`: every branch must be claimed, occluded,
  or unresolved WITH A GATE, and a record that is none of those is a bug, not a
  finding. That check alone would NOT have caught this -- both records reported
  `connected`. So `degenerate_arcs()` is added: pairs of branches sharing a saddle,
  sharing a maximum, drawing the same polyline. It compares polylines, not
  endpoints, so it also catches a future connector that is branch-dependent by
  construction. `n_degenerate_arcs` / `degenerate_arcs` go in the summary; the
  probe prints `audit : OK (0 uncommitted, 0 degenerate arc pairs)`; the viewer
  shows a red AUDIT FAIL line when either fires.
- **`DEGENERATE_ARC_TOL_DEG = 1e-3` is absolute, the inverse of Entry 054.** The
  question is "are these two polylines numerically identical" -- a property of
  float arithmetic, not of the mesh. The first cut used 1e-6 deg and the
  non-vacuity check FAILED to fire: identical world-frame arcs differ by 1.48e-6
  deg from round-trip noise alone (an angle between two points 1.26 A from the
  nucleus loses precision to cancellation). 1e-3 sits three orders above that
  noise and five below the 38 deg separation of the real pair.
- **Four things that caught me, recorded because each is a reusable check.**
  1. My first fix anchored the arc on the RAY and swept the full centre-to-target
     span. The control killed it: at ordinary separations that path deviates
     44-176 deg from the centre-to-target geodesic. A fix must be a no-op where
     the bug is absent, and this one was not.
  2. The audit tolerance was too tight (above), found only because I ran the
     pre-fix case.
  3. My Pd_20K arc-equality gate was VACUOUS: the dump omits `arc_points`, so the
     comparison term was `() == ()` for every record. Replaced with an in-process
     check that counts how many arcs take the bearing path at all.
  4. My test fixture put the target on the meridian az=0, where a k=2 saddle's two
     antipodal ports share ONE great circle, so the arcs could not separate. The
     target must be off that circle (az=90).
- **The non-vacuity proof, which is the load-bearing result.** Rebuilding the
  pre-fix arcs (no bearing argument) on the live snapshot and running the audit:
  **PRE-FIX 1 degenerate pair `SAD_4953/0~1`, POST-FIX 0.** The audit fires on the
  bug it was written for and is silent after the fix.
- **No-ops, proven non-vacuously.** Pd_20K: **0 of 96 (shape) and 0 of 52
  (curvedness) arcs take the bearing path** -- max separation 32.99 and 45.04 deg
  against thresholds of 174.56 and 146.48 deg. Verdicts byte-identical to HEAD.
  ethene3 shape index 4/4, curvedness 10/18, Electron Density 9/10, all with 0
  degenerate pairs and 0 uncommitted. (A side probe reported "96 arcs off-sphere";
  that was my measurement bug -- I took norms about the world origin, and Pd_20K's
  centre is 1e-5 off it. Centred norms are exactly R.)
- **Tests** (556, was 549): the antipodal pair draws distinct arcs (>90 deg) that
  both start at the centre, end inside the disc, and share an endpoint; both are
  great circles with zero torsion; an ordinary separation is byte-identical to the
  centre-to-target geodesic and ends at the CENTRE; `degenerate_arcs` fires on
  forced-identical polylines and ignores different maxima; every branch is
  committed across three fixtures; the summary reports the count.
- **Validation**: `pytest` 556 passed. ruff / mypy / pre-commit clean. Headless
  viewer smoke on the V snapshot.
- **Open next**:
  1. Inside the band the connector ends at the disc, not the centre, so
     `separation_deg` (centre-to-centre) and the drawn extent differ by up to a
     ring angle. Honest and documented, but a `bearing_arc` flag on the record
     would let a reader tell the two conventions apart without the geometry.
  2. The audit is computed in the summary and shown in the UI; it is not asserted
     at the pipeline boundary. A hard check in `stage_connect`/the viewer would
     turn a silent regression into a loud one.
  3. Bridge segment rendering, ring prominence, descent branches (Entries
     052/053).

### [Entry 056] Capture: Complete an Ascent That Overshot Its Maximum

- **Objective**: the user's step -- "now we have all saddles uphill launch points
  terminating at or near a maxima. For those few that terminate NEAR the maxima we
  want to complete the gradient path and terminate them at the maxima. We can do
  this by continuing to follow the gradient path to the topologically identified
  maxima even if the gradient is not uphill. A good example is
  `ethene3_atom1_params.json` where the gradient path originating at MC17
  terminates just before reaching MC10."
- **The case, mapped.** MC17 = `SAD_8400`, MC10 = `MAX_4864` (curvedness, the
  re-saved snapshot at k_pitch 3.4). The branch orbits at 4.96 and 7.14 deg from
  the maximum, whose disc is 3.79 deg: it terminates **1.17 deg = 0.53 pitch**
  outside. The user's description is exact.
- **The mechanism is gradient OVERSHOOT, not a flat stall.** The path is a
  two-cycle: it alternates forever between two samples (4.96, 7.14, 4.96, 7.14 ...)
  until the budget exhausts. The field at those points is 0.075 and 0.174, and the
  peak is 0.481 -- the step lands on the far side of the summit, the gradient there
  points back OVER the top, and the integrator bounces. The terminal heading is
  0.9677 . to_max (aimed essentially straight at it) and the field along the
  geodesic rises monotonically 0.174 -> 0.481. The maximum is not in question.
- **One deliberate deviation from the instruction, and why.** The user proposed
  following the gradient "even if it is not uphill". Measured, the gradient at the
  terminal point is **165 deg away** from the maximum -- that is precisely the
  overshoot, and following it is what produces the orbit. Continuing to integrate
  cannot terminate: the path is already in its limit cycle and has been for 347
  samples. So the completion is CAPTURE BY PROXIMITY with a straight final leg,
  not continued integration. The outcome the user asked for is achieved -- the path
  terminates at MC10 and is associated with it -- by the only mechanism that
  terminates.
- **The rule.** When an ascent ends (stall or budget) and the nearest maximum's
  disc is within `CAPTURE_PITCH = 1.0` mesh pitches of the terminal sample, append
  a straight leg to that maximum's CENTRE and report `arrived`. One pitch is the
  natural bound because the step is the cause of the overshoot. Checked AFTER the
  excision-region test, so a stall in noise is transited, never captured.
- **The separation in the data is enormous, so the threshold is not sensitive.**
  Swept every branch on the snapshot: all ten genuine arrivals have a NEGATIVE gap
  (they entered the disc, -0.03 to -0.76 pitch); the orbiting branch is +0.53; the
  one branch that truly reaches no maximum (`SAD_2417 br0`, spiralling on the
  plateau) is **+24.09 pitch**. A threshold anywhere in (0.53, 24) gives the same
  answer.
- **The leg is a completion, not a measurement, and the record says so.**
  `TraceOutcome.completed_leg_deg` and `MaxConnection.completed_leg_deg` carry its
  length; `n_completed` is in the summary; the arcs table shows a `complete` chip;
  the stats line reads "N completed onto its maximum (last leg drawn, not
  measured)"; the probe prints a `completed` line. The field along the leg is not
  integrated, because the integrator has provably run out of step resolution there.
- **Capture is independent of the transit rule.** It is a property of the STEP,
  not of the boundary geometry, so it runs when no regions are supplied. The early
  return that gated it on `not excision_regions` was removed; a test pins it.
- **`capture_pitch` is a parameter, not a monkeypatched constant.** It joins
  `step_pitch` / `max_arc_pitch` / `stall_patience` as the per-call overrides the
  tests use. The first cut monkeypatched the module constant and the patch leaked
  into the second call in the same test, which is what made the non-vacuity
  assertion fail.
- **Result.** ethene3 curvedness **11/12 connected, 1 completed, 0 no_target**
  (was 10/12, 1 no_target). MC17's branch now terminates AT MC10 -- last sample
  0.000000 deg from the centre, centred radii exactly R.
- **No-ops, as required.** Pd_20K: 0 completions on both fields, table
  byte-identical to HEAD (96/96, 52/52). ethene3 shape index 4/4 with 0 completions
  (its branches transit and arrive inside the disc); Electron Density 10/10, 0
  completions.
- **Drift confined to the connector layer.** Full-pipeline reproduce on the
  snapshot: 71/76, and the 5 drifts are exactly
  `connection_stats.{n_connected,n_gradient,n_no_target,n_unresolved,mean_separation_deg}`,
  all the expected 10 -> 11. Geometry, morse, reduction and launch all match.
- **Tests** (561, was 556): the capture pair (same field, same port, same
  integrator -- `capture_pitch=0.0` gives `budget`/None, the default gives
  `arrived`/MAX_0 with a leg, so the arrival is attributable to the rule alone, and
  the path grows rather than being substituted); a far port proves capture does not
  reach across a real gap; capture runs with no regions supplied; a broad Gaussian
  arrives under its own steps with `completed_leg_deg is None` (the no-op
  contract); the constant is one pitch. The fixture needed a SHARP spike
  (scale 2.5 deg, disc 1.0) -- a broad Gaussian decelerates smoothly into the top
  and cannot overshoot, so the first six fixtures all arrived and the test would
  have been vacuous.
- **A palette gate caught a UI bug**: `light-blue` is not in
  `VUE3_THEME_CONFIG`, and `test_every_drawer_color_token_is_in_the_palette`
  failed. The completion is now `info`, distinct from cyan (gradient), orange
  (transit) and deep-orange (occluded).
- **Open next**:
  1. The 2-cycle is visible in the drawn path (347 samples of orbit before the
     leg). Decimating a repeated cycle would make the connector readable; the
     orbit is also evidence the step is too coarse for this peak, which is worth
     an A/B against `TRACE_STEP_PITCH = 0.5`.
  2. `SAD_2417 br0` remains unresolved at 24 pitch -- a genuine plateau spiral,
     correctly left alone.
  3. The audit is shown, not asserted at the pipeline boundary (Entry 055 open 2).

### [Entry 057] Capture Is TOTAL: an Invariant Cannot Be Implemented as a Threshold

- **Objective**: the user's report, on `ethene3_atom1_params1.json` (curvedness,
  k_pitch 2.8) -- "the uphill gradient path originating at MC20 terminates before
  reaching the maxima MC12 and so this gradient path is never associated with the
  maxima. I am having a difficult time understanding why this is a difficult task.
  Every uphill launch point should be committed and every one of these should
  terminate at one of the topologically identified maxima... There are two ways to
  complete this connection: 1) continue following the gradient to the nearest CP
  (even though this may not be uphill), or 2) connect the endpoint of the gradient
  path with a great circle to the centre of the topologically identified maxima."
- **The answer to "why is this difficult": it isn't, and the difficulty was mine.**
  Entries 051-056 each implemented the invariant as a PROXIMITY GATE tuned to the
  example in hand. 051 fired inside a disc (MC2's 6.1 deg). 054 widened the ray cap
  to 180 (MC14's 178.01). 056 captured within ONE MESH PITCH because the first
  example, MC17 -> MC10, overshoots by 0.53 pitch. MC20 -> MC12 overshoots by
  **1.80 pitch**, so the gate was silent. For any number chosen, the next snapshot
  is past it. A threshold cannot express an invariant: it silently drops the case
  beyond its reach, which is the failure being reported, over and over.
- **The gate also split a mirror pair, which is the proof it was wrong.** MC17 and
  MC20 are mirror partners. Under the 1-pitch gate MC17 was captured and MC20 was
  not: a symmetric field answered asymmetrically, the same class of failure Entry
  046 caught when a disc gate split the symmetric maxima MAX_7 / MAX_4.
- **The fix is the user's option 2, applied totally.** Every gradient path that
  ends on trustworthy ground is committed to the NEAREST topologically identified
  maximum, at any distance, with a great-circle leg to its centre. No gate.
  "Nearest" is unambiguous on a converged path -- MC20's terminal point is 7.13 deg
  from MC12 and 118.61 deg from the next maximum, a factor of sixteen.
- **Option 1 provably cannot terminate, and this was measured, not assumed.** The
  path is in a 2-cycle limit orbit (347 samples alternating between two points) and
  the gradient at the terminal sample points 165 deg AWAY from the maximum -- that
  is the overshoot itself. More integration produces more orbit. Option 2 always
  terminates by construction, so option 2 is the mechanism.
- **The cost of totality is published, not hidden.** A threshold was being used to
  avoid fabricating an association. The honest instrument is to make the completion
  total and make its cost visible: `completed_leg_deg` on every record,
  `max_completed_leg_deg` across the table, `n_completed` in the summary. A
  connector that needed a 90 deg carry is a measurement of the field, on the table,
  for the operator to judge -- not a case dropped by a constant.
- **THE INVARIANT AS CODE.** `n_gradient_uncommitted` counts branches whose
  connector is a traced gradient path that does not terminate at a maximum. It must
  be zero. It is in the summary, in the probe's `audit : OK/FAIL` line, and in the
  viewer's red AUDIT FAIL block. This is the user's sentence, executed.
- **Truncation at the closest approach.** The first cut anchored the leg on the
  TERMINAL sample, and the mirror pair came out asymmetric: legs 5.06 and 7.13 deg.
  The two paths orbit at {5.06, 7.25} and {4.96, 7.13}; the budget cutoff lands on a
  different PHASE of the same bounce, so the terminal sample is an artifact.
  `_complete` now truncates the path at its closest approach to the maximum before
  appending the leg. The samples past it are the integrator bouncing between two
  points, carrying no information about the field. Result: legs 5.056 and 4.956 deg,
  agreeing to 0.1 deg, the residual mesh discretization rather than phase. Arc
  lengths drop 896 -> 130 deg: the orbit tail is gone from the drawing.
- **`capture_pitch` is now an opt-in limit, default None (total).** Zero disables
  the completion, which restores the pure Step 4 record -- and that is what the
  record-contract tests need, so `connect_seed_to_maxima` threads the parameter
  through for them. The constant `CAPTURE_PITCH = 1.0` survives as the documented
  value a caller may ask for, with its history in the comment.
- **Result on the user's case: 9/10 -> 10/10 connected, 0 no_target, 2 completed,
  0 gradient paths uncommitted.** MC20 (`SAD_8487`) br0 terminates at MC12
  (`MAX_4864`).
- **The invariant holds on every field measured.** ethene3 curvedness 10/10, shape
  index 4/4, Electron Density 10/10, V 2/2 -- all with 0 gradient paths uncommitted.
  The shape-index branches still TRANSIT (4) and arrive inside the disc, so they
  need no completion: transit runs first, totality is the last resort.
- **No-ops, as required.** Pd_20K: 0 completions on both fields, table
  byte-identical to HEAD (96/96, 52/52) including `completed_leg_deg`.
- **Drift confined to the connector layer.** Full-pipeline reproduce on the
  snapshot: 71/77, and the 6 drifts are exactly
  `connection_stats.{n_connected,n_gradient,n_no_target,n_unresolved,n_completed,mean_separation_deg}`,
  all the expected 9 -> 10. Geometry, morse, reduction and launch all match.
- **Tests** (565, was 561):
  - `test_capture_does_not_reach_across_a_real_gap` DELETED. It asserted the old
    semantics -- a distant maximum is not captured -- which is precisely the
    behaviour the user is rejecting. Replaced by
    `test_capture_completes_a_far_convergence_and_publishes_the_carry`, which
    asserts the far case IS committed and reports a long leg, and
    `test_capture_pitch_is_an_opt_in_limit_not_the_default`, which shows a caller
    can still ask for the gate.
  - `test_a_plateau_stalls_without_a_maximum` keeps its subject (the INTEGRATOR) by
    disabling capture; `test_a_plateau_stall_is_committed_to_the_nearest_maximum`
    adds the same field under the default, asserting the 90 deg carry is published.
  - `test_a_pool_with_no_maximum_is_the_only_way_to_stay_uncommitted` pins the
    single legitimate exception.
  - The transit and nesting tests disable capture so each isolates its own
    mechanism; the capture pair's non-vacuity assertion is updated for truncation
    (the path is SHORTER by design, and the endpoint is the maximum).
- **Validation**: `pytest` 565 passed. ruff / mypy / pre-commit clean. Headless
  viewer smoke on the snapshot.
- **Open next**:
  1. A completion leg is drawn in the same style as the measured path. A distinct
     rendering (dashed, lower lift) would separate carried from measured on screen;
     the data already distinguishes them via `completed_leg_deg`.
  2. The 2-cycle orbit is a symptom that one mesh pitch is too coarse for these
     peaks. An A/B against `TRACE_STEP_PITCH = 0.5` would show how much of the
     completion machinery the step size is responsible for.
  3. The invariant is displayed, not asserted at the pipeline boundary (Entry 055
     open 2). A hard check in `stage_connect`/the viewer would turn a silent
     regression into a loud one.

### [Entry 058] Ridge Refinement: Piecewise Transverse Relaxation with Pinned Knots

- **Objective**: the user's request -- "now we have all ascending saddle launch
  points associated with a gradient path that takes us to a morse detected maximum.
  While this path is close to the actual ridge line connecting maxima to saddles, it
  is on occasion just off. I am wondering if we can improve this path and 1) get it
  closer to the actual ridge, and 2) smooth it out. My idea is to pin the gradient
  path (just those found from gradient following, the cyan ones) at the saddle and
  the maxima. With their ends fixed we nudge the path slightly. My feeling is that we
  should begin the nudge in the middle of the path where the field is less noisy. At
  the saddle side we are moving through very flat regions and at the maxima the
  variation along the contour is slowly varying. In the middle of the path the visual
  location of the ridge is quite dramatic. The other points to look for are locations
  where the ascending path makes sharp curves. An example is in
  ethene3_atom1_params.json where we see a sharp turn as the path ascends from MC16
  to MC12."
- **Scope, as the user restricted it**: cyan connectors only
  (``method == "gradient"``). A geodesic connector is a great circle between two
  centres -- it approximates nothing, so there is no ridge to align it with, and
  nudging it would move it OFF the line it is defined to be.
- **The user's three refinements to the design, each of which changed it:**
  1. *"These sharp changes of direction are real and correspond to sharp corners on
     the contours marking the ridge line. We don't want to make some smooth curve
     that blunts these changes of direction."* -> the piecewise anchor model. A
     global smoothing pass is disqualified by the physics.
  2. *"The sharp turns should have a small amount of ability to shift position. They
     are real but the exact point of bending has been found where the field is nearly
     flat and so may be variable. We should perhaps knot the two regions together."*
     -> the C0 knot: a shared vertex with INDEPENDENT incoming and outgoing
     tangents, free to move within a small disc. Position continuity, no slope
     continuity. That is exactly what keeps an elbow an elbow while letting its
     apex find the crest.
  3. *"When looking at the gradient path we see as it passes through flat regions a
     saw tooth pattern. The teeth may have angles that would qualify as sharp curves.
     However, it is their average behaviour that we should look at, extending over
     several gradient steps."* -> windowed chord deflection. Over W ~ 4 pitches the
     teeth cancel: the incoming and outgoing chords of a sawtooth are parallel.
- **The second trap, found by measurement, not by argument.** The window kills the
  sawtooth, and then a latitude circle defeats it. A small circle at colatitude
  0.20 rad deflects **56.7 deg per window at every interior vertex** -- above the
  35 deg threshold -- because a curving ridge bends everywhere. Sustained curvature
  is not a corner. An absolute threshold calls the entire arc corners.
  The fix is the BASELINE: a corner is a vertex whose deflection exceeds the
  MEDIAN deflection of its own path by ``CORNER_EXCESS_DEG``. Measured excess on
  the latitude circle at theta = 0.20 / 0.30 / 0.45: **0.0 / 0.0 / 0.0**. The test
  is scale-free -- it does not care whether the ridge is gently bowing or violently
  wiggling, only whether this vertex bends more sharply than the path around it.
  This is the same shape of lesson as Entry 057 (a threshold cannot express an
  invariant), reached from the opposite direction: here the threshold was not too
  tight, it was absolute when it needed to be relative.
- **THE FAILURE, AND IT IS THE ENTRY'S FINDING.** The first working version
  estimated the crest at each vertex and applied it. On synthetic fields it was
  excellent: a path 1 pitch off a crest recovers -1.011 pitch (true answer -1.0),
  and on a crest it drifts 0.06 pitch, the IDW noise floor -- an 11x separation.
  On the REAL field it made the paths worse:

  | path | teeth > 35 deg (base -> refined) | total turning (base -> refined) | field gain |
  |---|---|---|---|
  | SAD_1594/1 | 6 -> **15** | 628 -> **1374** | +0.03% span |
  | SAD_1681/1 | 2 -> **16** | 297 -> **1543** | +0.02% span |
  | SAD_8400/0 | 6 -> **19** | 688 -> **1553** | +0.06% span |
  | SAD_8487/0 | 10 -> **19** | 1049 -> **1806** | +0.35% span |

  It bought 0.02-0.35% of field span and paid by DOUBLING the wiggle -- the exact
  opposite of request 2. The diagnosis: the IDW interpolant ripples at the mesh
  scale, with amplitude comparable to the field change over one pitch, so the
  per-vertex crest estimates disagree with their neighbours by 0.18-0.22 pitch
  (median, up to 1.67). Applying them vertex by vertex injects that disagreement
  into the geometry.
- **The fix is the same estimator idea, one dimension over.** The parabola fit
  denoises ACROSS the ridge. Nothing denoised ALONG it. The offset of a ridge from
  a path is a smooth function of position along the path -- a ridge does not wiggle
  at the mesh scale -- so the per-vertex estimates are FITTED as one field over
  each sub-segment (``_smooth_offsets``, degree 2) before any of them is applied.
  A least-squares fit along the path averages the noise exactly as the parabola
  averages it across. Re-measured on the same field:

  | path | teeth > 35 (base -> raw -> fitted) | turning (base -> raw -> fitted) | field gain (raw -> fitted) |
  |---|---|---|---|
  | SAD_1594/1 | 6 / 15 / **6** | 628 / 1374 / **647** | +0.029% -> +0.029% |
  | SAD_1681/1 | 2 / 16 / **2** | 297 / 1543 / **317** | +0.023% -> +0.021% |
  | SAD_8400/0 | 6 / 19 / **6** | 688 / 1553 / **727** | +0.061% -> **+0.438%** |
  | SAD_8487/0 | 10 / 19 / **10** | 1049 / 1806 / **1059** | +0.346% -> **+0.442%** |

  Roughness returns to EXACTLY baseline, and the field gain is equal or better --
  on two paths 7x and 1.3x better. The fit is not a trade of accuracy for
  smoothness; the raw version was throwing away accuracy AS WELL as smoothness.
  On a synthetic rippled field the ratio is starker: raw multiplies total turning
  by x155, the fit by x5, and the fitted path sits CLOSER to the crest (0.9902 vs
  0.9838).
- **The second defect, caught by a test that should have failed earlier.** The
  first implementation read each vertex's frame from the LIVE array, so a vertex's
  cross-track depended on where its neighbour had just moved, and a chain of small
  tilts compounded: a path already sitting on the crest walked 4.3 deg off it.
  Relaxation is now JACOBI -- frames read from a snapshot, displacements written to
  the working array -- which is how every string method is written and how this one
  should have been written first time.
- **The user's MC16 -> MC12 case, found and fixed.** All four gradient connectors on
  ethene3 curvedness run to MAX_4864 (MC12) from MC15, MC16, MC19, MC20 -- a mirror
  quartet -- and the detector promotes ONE vertex on each, at index 9, 10, 9, 9.
  Consistent across the symmetry, which is the test Entry 046 and Entry 057 taught
  us to run. The corner survives the refinement:

  | branch | windowed deflection (base -> refined) | point-to-point turn at the knot |
  |---|---|---|
  | MC15/1 | 71.8 -> 83.4 | 78.7 -> 109.7 |
  | MC16/1 | 60.3 -> 53.7 | 96.0 -> 89.5 |
  | MC19/0 | 61.1 -> 56.3 | **20.2 -> 54.4** |
  | MC20/0 | 55.7 -> 54.7 | 90.4 -> 73.0 |

  All stay far above the 35 deg corner threshold: nothing is blunted. MC19 is the
  case the user's "knot" instinct predicted -- the detected vertex was 20.2 deg from
  the true apex, and seating the knot on the crest SHARPENED it to 54.4 deg. The
  corner was in the right place all along; the ascent had just stopped short of it.
- **Estimator, not threshold, throughout.** The crest fit is a parabola in LOG
  space, which is exact for a Gaussian ridge, and its rejections are structural:
  the fit must be CONCAVE (a crest exists in the window at all) and its vertex must
  lie INSIDE the window (a crest beyond it is not resolved by it, and clamping to
  the edge would move the vertex the full nudge width on no evidence). No gain
  threshold was added -- that would be the Entry-057 mistake for the third time.
- **A signed-step trap worth recording.** ``sphere_frame.rotate_towards`` clamps a
  non-positive angle to a no-op and returns the origin. Correct for forward
  extrapolation, fatal for a symmetric cross-track search: every negative sample
  collapsed onto the vertex, the search saw one side of the ridge, and the whole
  refinement was silently dead (max_shift 0.0). ``_offset_unit`` carries the sign in
  the DIRECTION and passes the angle as a magnitude.
- **The cost is published, as always.** ``refined``, ``refine_max_shift_deg`` and
  ``n_corners`` on every record; ``n_refined``, ``max_refine_shift_deg`` and
  ``n_refined_corners`` in ``connection_stats``; a cyan line in the drawer and a
  ``refined`` chip in the arcs table. The refined line is an INTERPRETATION of the
  ascent, and the overlay must never be mistaken for the measurement.
- **A post-pass, deliberately not part of ``connect_seed_to_maxima``.** The
  connection table is a record of what the gradient did; the refinement is a reading
  of it. Keeping them separate is what lets the viewer toggle without re-integrating
  a single ascent, and it is why the default is OFF.
- **Result**: 4 of 4 cyan connectors refined, max shift 2.62 deg (was 4.51 before
  the fit), 4 corners kept, audit OK. **Verdict-neutral**: 10/10 connected, 0
  no_target, 2 completed, unchanged -- the refinement moves geometry, not topology.
- **No-ops, as required.** Pd_20K curvedness: 928 branches, 0 gradient, every record
  returned BY IDENTITY, stats identical. ethene3 reproduce with the control off:
  **79/79, zero drift**. With it on: also 79/79, because the counters that change
  are inside ``connection_stats`` and the snapshot records the off state.
- **Schema**: ``PARAMS_SCHEMA_VERSION`` 3 -> 5, with 3 admitted as legacy. Version 4
  was the REVERTED Step 5 connector-refinement control (``cd99bbd``, reverted by
  ``b48b6cf``); its snapshots are already legacy, and the new control is named
  ``refine_ridge_paths`` to avoid colliding with the dead ``refine_connectors``.
  Every v3 file on disk loads with the control off, which is the faithful reading:
  it did not exist, so it was not on screen.
- **Tests** (592 pass, was 565): 20 in ``test_ridge_refiner.py`` and 7 in
  ``test_max_connection.py``. The ones that carry the design:
  - ``test_sustained_curvature_is_not_a_corner`` -- the latitude circle, with a
    non-vacuity assertion that the fixture really deflects 56.7 deg per window.
  - ``test_sawtooth_chatter_is_not_a_corner`` -- asserts the teeth give huge
    point-to-point turns (> 60 deg) and zero corners.
  - ``test_the_displacement_is_fitted_not_applied_vertex_by_vertex`` -- the
    regression above, on a rippled field: raw x155 roughness, fitted x5, and the
    fitted path closer to the crest.
  - ``test_an_off_crest_path_settles_onto_the_ridge`` /
    ``test_an_on_crest_path_barely_moves`` -- the signal and the noise floor, with
    the ratio asserted (off-crest mean shift > 8x on-crest).
  - ``test_endpoints_never_move``, ``test_a_transited_excision_centre_is_pinned``,
    ``test_a_corner_is_not_blunted_by_the_neighbouring_relaxation``.
  - ``test_refinement_is_frame_independent`` -- off-origin centre equals centred.
  - ``test_a_path_with_no_interior_is_returned_unchanged`` and
    ``test_a_path_too_short_to_refine_is_returned_unchanged`` -- a caller may hand
    every connector without pre-filtering.
- **Fixture lessons recorded in repo memory** (``ridge_refinement.md``): a 600-point
  sphere has 8.3 deg spacing, so PITCH = 2 deg is a fiction and a 4-pitch ridge is
  unresolvable; do not demand exactness from a least-squares fit on a noisy
  interpolant; a sin-tapered bulge legitimately has more turning than the straight
  line it started from, so roughness must be compared across degrees of freedom, not
  against the baseline.
- **Validation**: ``pytest`` 592 passed. ruff / mypy / pre-commit clean. Headless
  viewer smoke on the v3 snapshot with the new control.
- **Open next**:
  1. The corner threshold is relative to the path's own median, which is right for
     curvature and costs sensitivity for a kink shallower than the bow it sits on:
     a 70 deg kink on a 0.30-rad bow gives 25.3 deg of excess and is NOT promoted
     (90 deg gives 36.5 and is). A curvature-PROFILE test (excess over a LOCAL
     baseline, not the global median) would recover the shallow cases.
  2. The refinement is applied to the whole path between pinned nodes. The user's
     original instinct was to begin in the middle, which the sin taper implements;
     a per-vertex confidence weight from the fit residual would let flat ends relax
     less and steep middles more, adaptively.
  3. Gold geodesic connectors are untouched by design. Where a geodesic ray resolved
     a branch, the connector is a straight line and the ridge may curve; refining
     those would need the ridge, not the ascent, as the initial guess.
  4. ``fit_degree`` is a module constant, not a control. It behaves as a quality
     parameter, not a tuning knob (0 and 2 both beat raw by ~30x on roughness), so
     promoting it to a slider is not recommended.

### [Entry 059] The Ridge Probe, and the Operator's Challenge That Found Two Defects

- **Objective**: Entry 058's refinement failed inspection. The user: "This
  correction produced no notable change to the gradient paths, if anything it made
  them less smooth and did not move them at all toward the true ridge. The sawtooth
  patterns are as prevalent as before." Agreed direction: build the ridge criterion
  as an INSPECTABLE PROBE -- mark the ridge beside the unchanged path, let the eye
  adjudicate, move the path only once the points are trusted. Scope: cyan
  (gradient-traced) connectors only.
- **The probe** (`ridge_refiner.probe_gradient_ridge`): detect corners (reusing
  Entry 058's windowed-chord + median-baseline test UNCHANGED -- it is exactly the
  user's "sharp and continuous change as compared to saw teeth"), pin endpoints and
  corners, resample each sub-segment every 5.5 deg along the path, and at each
  sample step onto the ridge by the field's own curvature: fit log f over a
  2.5-pitch disc, take the eigenvector of the most-negative Hessian eigenvalue (the
  across-crest direction, a property of the FIELD, not of the path -- this breaks
  the circularity of `_transverse_frame`, whose docstring claimed the smoothed
  direction and whose code read the path's own i±1 vertices), Newton-step to the
  zero crossing of the tangential derivative. Viewer: black circles at knots, white
  at ridge points, toggle `probe_ridge_points` (schema 5 -> 6, 5 admitted legacy).
  Headless: `--probe-ridge` prints per-sample offset, gain, and the other side.
- **The user's challenge, and it was the right question.** "How did you decide which
  direction to move the point? In ethene3 I believe there is a more developed ridge
  when the probe points are moved the OTHER direction. To my eye they should have
  moved as much as five degrees the other way. Check MC10 to MC17." Interrogating
  the code: the Newton step lands on the NEAREST zero crossing -- a local rule. The
  operator's original phrasing ("move around its contour to find the ridge") is a
  GLOBAL search along the cross-track line. So: scan both directions, find all
  crests, pick the most developed. Implemented, measured, and the measurement
  refuted the naive version of the rule -- twice, in ways that improved it.
- **Defect 1, found by the two-sided scan on a synthetic flat-topped ridge:
  "highest value" is the wrong rule on a broad crest.** The field varies across a
  pitch by LESS than the IDW ripple, so a ripple peak on the flank is higher in
  VALUE than the true crest. Measured: the highest-value rule stepped a sample from
  latitude +0.718 deg to +1.249 deg -- AWAY from a crest lying at 0. A zero crossing
  of the tangential derivative is localised on a flat top; an argmax is not. The
  final rule: Newton locates the crest, the scan GATES it and REPORTS the other
  side (`RidgePoint.other_side`), per sample, in the shipped tool.
- **Defect 2, found by measuring the gain distribution: a window-relative
  prominence gate waves the ripple through.** In the flat ground near a saddle the
  scan window's entire value range IS the ripple, so "prominent relative to this
  window" passes ripple peaks. The gate must be ABSOLUTE and physical: the crest
  must rise above the sample by a fraction of the FIELD'S SPAN. Measured on ethene3
  (110 samples, gain of the best crest within reach, % of span):
  f < 5% span (n=82): median +0.0008, max +0.156 -- noise.
  f in 5-25%  (n=22): min +0.167, median +10.1 -- a ridge.
  f > 25%     (n=15): min +7.28,  median +9.3 -- a ridge.
  The populations do not overlap; the gate at 0.05% rejects 81/82 noise candidates
  and keeps 36/36 real ones. `_CREST_MIN_GAIN_FRACTION = 0.0005`.
- **The reach, measured then multiplied by 1.5 at the operator's instruction.** The
  bound must be measured on the quantity it gates -- the Newton step, not the scan's
  crest positions. All 110 samples: the 36 with a real gain step at most **1.07 deg
  (0.48 pitch)**; the 74 ripple samples step FARTHER (max 2.39 deg) for zero gain --
  the signature of extrapolation on flat ground. 1.5 x 1.07 = 1.61 deg = **0.72
  pitch** = `RIDGE_REACH_PITCH` (was 2.0, chosen). In pitches, so it travels with
  the mesh. Read as a RADIUS from the path; as a full diameter it would reject the
  very sample that defines it.
- **The answer to "five degrees the other way", per sample, in the tool.** On the
  final probe of ethene3 (k_pitch 3.2): 36 ridge points, 8 knots, median gain
  +9.14% of span, and **zero samples found a crest on the opposite side** -- the
  scan is two-sided and the report says so per branch. On MC17 -> MC10
  (SAD_8400 br0): 30/50 accepted, offsets flipping sign 14 times with gains of
  7-19% of span -- the field itself reports the path CHATTERING across a real crest,
  the sawtooth the operator sees. The refusals are informative, not noise:
  `along_path` clusters near the saddle (the steepest-descent axis points along the
  path there), `no_crest` in the flat region (f < 5% of span: the field genuinely
  does not say where the ridge is), `out_of_reach` where the crest is a different
  ridge. A probe that invented points in the flat region would be drawing
  interpolation noise and calling it a crest.
- **The report is the product.** Two bugs fixed in `format_probe` on the way: the
  arc column restarted at zero after each knot (two rows labelled "5.5" in one
  table -- unreadable), now cumulative; and the summary must print the MEDIAN, not
  the max -- Entry 058 reported a max of 2.62 deg where the median was 0.06-0.40
  pitch, and the max was the knot. The shipped line: `gain of the chosen crest over
  the path: median +9.142% of span` and `no sample found a crest on the opposite
  side of the path`.
- **Tests** (603 pass, was 592): 11 probe tests in `test_ridge_refiner.py`.
  `test_probe_points_ignore_the_sawtooth` is the one the 058 suite lacked -- every
  058 test passed on a refinement that moved nothing. `test_probe_refuses_a_crest_
  beyond_the_measured_reach` keeps the bound a bound (a 1.5-pitch offset must be
  told out_of_reach, not stepped two thirds of the way).
  `test_scan_finds_crests_on_both_sides_of_the_sample` proves the far side is SEEN
  (two crests, correct signs, taller = higher). Fixture lesson recorded: the
  two-crest scan test needs an 8000-point mesh -- at 1200 the spacing is 2.8
  pitches and a 1.2-pitch crest is not representable in the interpolant.
- **No-ops**: the probe moves nothing (tested: input byte-identical; sample FROM
  positions on the path, ridge points off it). Pd_20K: 0 gradient branches, the
  viewer's `probed` list is empty, nothing runs. The v6 snapshot with
  `probe_ridge_points: true` reproduces 82/82 headless -- display-only, inert in
  the pipeline, like `show_max_saddle_arcs`.
- **Open next**:
  1. Rebuild the relaxation on the probe's points: move the path to the accepted
     ridge points (the 36 with median gain 9% of span), with the tangential tension
     term that is the ONLY thing that can remove teeth -- a cross-track correction
     cannot, which is why 058 failed request 2.
  2. The flat region near saddles yields `no_crest` honestly. If the operator wants
     continuity there, the answer is interpolation between accepted samples along the
     segment, not a weaker gate.
  3. The 21 `along_path` refusals on a sawtooth are the sample's local tangent being
     a tooth. A segment-averaged tangent for the axis test would accept more;
     measure before choosing.

### [Entry 060] The Probe Renders at Its Own Scale: Ticks, Refusals, and a Bound Vindicated

- **Objective**: the user inspected the Entry-059 probe on
  `ethene3_atom1_params.json` (MC15 selected, probe on): "I was expecting to see
  white dots all along the cyan legs. But in this case I see a couple near the
  maxima, the black knot pin, but not displaced white dots at about 6 degree
  increments along the segments." Three observations; two defects and one
  vindication.
- **Defect 1, "not displaced" -- a display bug, and a general rule for diagnostic
  overlays.** The measured shift is 0.0063 A (0.39 pitch at R = 0.4127 A). The
  probe glyph DIAMETER was 0.020 A. A dot three times wider than the displacement
  it marks renders on top of the path: the displacement was real and invisible.
  The fix is to draw the displacement itself: a TICK -- a thin tube (radius
  0.0015 A, a quarter of the shift) from each sample to its ridge point -- and a
  glyph shrunk to 0.005 A. RULE: a diagnostic mark must render at the scale of the
  quantity it shows; when the shift is the message, the glyph must be smaller than
  the shift, and the shift must be drawn.
- **Observation 2, "a couple near the maxima" -- the field, not a bug, and the
  bound is vindicated.** 21 samples per branch, 2 accepted, both near the maximum
  (along 114/119 of a 123 deg path). The temptation after "only 2 of 21" is to
  loosen `RIDGE_REACH_PITCH`. The counterfactual was computed first: for every one
  of the 22 out_of_reach refusals on this snapshot, the refused Newton step was
  evaluated -- f(step) equals f(path) to four decimals at every one of them (gains
  +0.001% to +0.012% of span; the largest, 0.009%, is a tenth of the 0.05% gate).
  **0 of 22 refused steps would have gained anything.** The bound is not
  over-tight; the ridge is developed near the maxima (f = 25-35% of span) and flat
  along the legs (f = 0.4-1.3% of span). On this field the operator's expectation
  -- "white dots all along the cyan legs" -- describes a ridge that the curvedness
  does not contain there. This is the same lesson as Entry 057 from the display
  side: a gate loosened to make a picture look busy is a gate that lies.
- **Observation 3, "at about 6 degree increments" -- the sampling was always
  there; only the accepted points were drawn.** 6 deg is `SEGMENT_SAMPLE_DEG =
  5.5`. The bare legs read as unprobed because refusals were invisible, which is
  indistinguishable from a broken probe. Every refused sample now gets a dim grey
  mark ON the path (`RIDGE_REFUSE_RGB`), and the drawer line names the count: "N
  samples refused (grey): the field defines no ridge within reach there". The
  accepted points stand out of a visible sampling, and the geometry of the probe
  -- where it looked, what it found, what it refused -- is all on screen.
- **Why the refusals cluster the way they do** (from the reason counts, per
  branch): `no_crest` dominates the flat legs (the gain gate: the step lands on
  ground no higher than the sample); `out_of_reach` fires just past the reach
  (0.73-1.03 pitch, all zero-gain); `along_path` clusters near the saddle (the
  steepest-descent axis points along the path there). Each is a fact about the
  field, and each is now drawn.
- **Tests** (603 pass, unchanged): the probe math is untouched -- this entry is
  the display of the same measurement. The viewer smoke builds all four probe
  actors (knots, points, ticks, refusals).
- **Open next**:
  1. The tick makes the shift visible at the default camera; if the operator
     zooms out, 0.0015 A tubes vanish with it. A screen-space line width
     (vtkActor property, not tube geometry) would keep the tick visible at any
     zoom; the tube is the right call for the current inspection distance.
  2. The flat legs are the real finding for the refinement step: a relaxation
     built on the probe's points will move the path near the maxima and leave it
     alone along the legs, because that is where the field speaks. If the operator
     wants ridge-following along the legs too, the criterion there is not the
     local Hessian -- it is the separatrix definition (the stable manifold of the
     saddle), which is a global object. That is a design conversation, not a
     constant.

---

### [Entry 061] The Sawtooth Smoother: Chatter Removed as Shape, Not as Cross-Track Shift

- **Objective**: Entry 058's transverse nudge failed inspection -- "no notable
  change... if anything it made them less smooth and did not move them at all
  toward the true ridge. The sawtooth patterns are as prevalent as before." The
  diagnosis, agreed with the operator: **a cross-track correction cannot remove a
  tooth, because the tooth IS the cross-track motion.** The teeth are removed by
  changing the path's shape ALONG its length. Plan agreed in four units: (1)
  `smooth_sawtooth_runs` as pure geometry, unit-tested on synthetic sawtooths; (2)
  wire it as the refinement, retiring the nudge machinery, keeping the probe as the
  diagnostic; (3) viewer A/B -- the existing toggle now means *smooth the teeth*;
  (4) journal + commit.
- **The construction.** Each maximal chatter run is deleted and replaced by ONE
  spherical Catmull-Rom (SQUAD) curve through its bounding anchors. The control
  point at an anchor is built from the anchor's neighbours IN THE ANCHOR POLYGON,
  not in the raw path -- which is the whole trick. A Catmull-Rom through the teeth
  reproduces the teeth; one through the nodes the teeth hang between draws the
  midline they oscillate about, because the midline is exactly what the anchor
  polygon is. Anchors are immovable: endpoints, knots, transited centres. A knot
  keeps its angle because the curve passes through it and is free to kink there --
  no tangent continuity is imposed across a knot.
- **The defect the pure-sawtooth fixture caught, before any test was written.** The
  first version pinned each run to its entry/exit VERTEX. On a 41-vertex sawtooth
  that gave a curve with latitude in **[1.500, 2.998] deg** when the midline is 0:
  the entry/exit vertices ARE teeth, at +amp, so the curve was built from tooth to
  tooth and sat a whole tooth off the crest. Total turning collapsed 3509 -> 0 deg,
  which looked like a triumph and was a lie -- a flat line in the wrong place. The
  fix: consecutive teeth sit at +amp and -amp, so the midline is crossed at the
  **midpoint of the edge between them**; runs are anchored at fractional indices
  `lo - 0.5` / `hi + 0.5`. Interior latitude then spans 0.32 deg from +/-1.5 deg
  teeth, and the reported anchors became `(0.0, 0.5, 39.5, 40.0)`. Lesson: a metric
  that improves by 100% (turning -> 0) is the one to distrust first.
- **Chatter vs corner, the second discriminator.** `detect_corners` (windowed chord
  deflection above the median baseline) is REUSED UNCHANGED -- Entry 059 called it
  "exactly the operator's distinction". The new `detect_chatter_runs` inverts it: a
  vertex is a TOOTH when its point-to-point turn is large (it kinks here) AND its
  windowed deflection is NOT (the average direction does not). A corner is large in
  both. Corners win -- a knot is never a tooth -- and a run shorter than
  `SMOOTH_MIN_TEETH = 3` is left alone, because one zigzag is a vertex's noise, not
  a sawtooth, and replacing it is a coin flip about the field.
- **Geodesic primitives promoted to `sphere_frame.py`** (the Entry 039 precedent):
  `sphere_log`, `sphere_exp`, `slerp`, `squad_control`, `squad_point`. A curve that
  must pass through given points on this surface is built out of exactly these, and
  the construction is geometry, not topology. Verified: log/exp round trip exact,
  SQUAD interpolates its knots to 0.0 deg, and a great circle gets **0.0 deg of
  bulge** -- the property that makes it safe on a path that is already straight.
- **Output density: 0.45 pitch, mesh-relative** (the operator's 1 deg on ethene3's
  2.2278 deg pitch, chosen as a pitch so it travels with the mesh like
  `RIDGE_REACH_PITCH`). Only runs are resampled; every stretch the field was never
  in question keeps its original vertices, so the record's shape survives and the
  smoothed line is visibly finer than the mesh.
- **`max_shift_deg` had to die.** Once the output carries a different vertex count
  there is no vertex-to-vertex shift to report, and a per-vertex metric would also
  hide the fact that a resampled curve passes BETWEEN its samples. Replaced by
  `max_deviation_deg` (each output vertex to the INPUT polyline) plus
  `total_turn_before_deg` / `after_deg`. The deviation is a closed-form
  point-to-arc distance, not a sampled walk: the first version sampled every segment
  at 0.05 deg and turned a unit test into 55 seconds; the closed form agrees with
  brute force to 0.0004 deg and the suite runs in 8.
- **Measured on ethene3 curvedness** (`--refine-ridges`, the headless A/B):
  `3 gradient connectors, 4 sawtooth runs removed (3 corners kept sharp)`,
  `turning 3118 deg -> 1922 deg total; max deviation from traced 0.87 deg`
  (= 0.39 pitch, within a tooth, as it must be). Per branch: SAD_8487 br0
  1623.7 -> 1053.5 (35.1% removed), SAD_1594 br1 1078.4 -> 576.7 (46.5%), SAD_8400
  br0 1195.4 -> 868.5 (27.3%, 2 runs), and **SAD_1681 br1 691.8 -> 691.8, "no
  chatter"** -- returned verbatim, which is the non-vacuity proof that the detector
  has negatives. No branch's turning increased. Audit OK, 0 degenerate arc pairs.
- **The resampling risk the plan flagged, tested not assumed.** `degenerate_arcs`
  compares polylines point-for-point and requires equal vertex counts, and it runs
  on the REFINED table. `test_smoothing_preserves_the_degenerate_arc_audit` proves
  both halves: two identical sawtooth legs resample to the same 68 vertices and are
  still flagged as collapsed; two legs with different tooth depth resample to the
  same count and are NOT flagged -- the fixture is built so the length guard cannot
  be what separates them (max point-for-point deviation 0.800 deg against a 0.001
  deg tolerance).
- **The probe's contract, violated by the resampling and fixed.**
  `format_probe`'s docstring requires the TRACED table ("the probe measures the
  ascent, so it must not be shown a line already moved to its answer"). The viewer
  built `probed` from `runtime_max_connections`, which is the smoothed table --
  harmless under the nudge, which preserved vertex count, and a real violation once
  the smoother resamples. `runtime_traced_connections` is now kept and the probe
  looks the traced record up by `(saddle_id, branch_index)`.
- **Schema: v6, key reused.** `refine_ridge_paths` keeps its name and the version
  stays 6. The semantic change is invisible to the file format: every snapshot on
  disk has the control OFF, so a v6 snapshot replays identically, and one saved with
  it ON asks for "the interpretation of the ascent", which is what this is.
  Verified: the untouched `ethene3_atom1_params.json` reproduces **82/82**; the
  same file with the toggle forced on reproduces 79/82, the three drifts being
  exactly `n_refined`, `n_refined_corners`, `max_refine_shift_deg` -- the stats the
  control is supposed to change.
- **The nudge machinery retired** (7 functions, 4 constants): `refine_gradient_ridge`,
  `_crest_offset`, `_crest_offset_at`, `_smooth_offsets`, `_refine_knot`,
  `_stationary_point`, `_transverse_frame`, and `NUDGE_MAX_PITCH`, `NUDGE_SAMPLES`,
  `DISPLACEMENT_FIT_DEGREE`, `KNOT_MAX_PITCH`. The retirement set was computed as
  the transitive call closure of `refine_gradient_ridge` MINUS the closure of
  `probe_gradient_ridge`, which is what kept the shared helpers (`_offset_unit`,
  `_sample_field`, `_segment_bounds`, `_window_steps`, `_path_units`,
  `chord_deflection_deg`, `detect_corners`, `tangent_project_local`) alive for the
  probe. `refine_connections` no longer takes `mesh_pts`, `f_vals`, or
  `find_k_nearest` -- the smoother samples no field, so the signature now says so.
- **Tests**: 606 pass (was 603). **20 added, 17 deleted with the code they tested.**
  `test_ridge_refiner.py` +16 / -14: 3 geodesic primitives (log-exp round trip,
  degenerate cases, SQUAD interpolation and zero bulge), 2 chatter detection, 9
  smoother (teeth removed, midline, anchors and endpoints, corner kept sharp,
  smooth path untouched, finer than the mesh, deterministic, cost reported, short
  path no-op), frame independence, transited centre. The 14 deleted are the nudge's
  four test sections -- transverse relaxation, knots, frame contract, displacement
  smoothing. `test_max_connection.py` +4 / -3: the post-pass now tests smoothing
  (chatter removed, geodesic untouched, smooth path untouched, short path
  untouched, transit pinned, endpoints preserved, summary, degenerate-arc audit)
  against the nudge's three crest-onto tests.
- **Open next**:
  1. The smoother removes chatter; it does NOT fix drift off the crest (defect 1).
     The probe measures that drift and on ethene3 reports median |offset| 0.39
     pitch. Closing it needs the separatrix definition (the stable manifold of the
     saddle), a global object -- Entry 060's point, unchanged.
  2. `detect_corners` does not promote a corner EMBEDDED in a full sawtooth: the
     teeth raise the median baseline, so the corner's excess falls under the
     threshold. Verified on a synthetic 70 deg elbow between two chattering legs
     (returns `()`). The smoother's contract is that the caller supplies knots, so
     the machinery is correct and the detector is the limit. If the operator wants
     corners found inside chatter, the baseline must be estimated from the
     smoothed (midline) path, not the raw one.
  3. The two 44.7 deg endpoint turns on a pure-sawtooth fixture are the pinned
     endpoint teeth, not residual chatter. Real launch ports and maxima are not
     teeth, so this is a fixture artefact; worth remembering if a synthetic test
     ever reports a suspiciously large `total_turn_after_deg`.

---

### [Entry 062] The Cost Was the Metric, Not the Smoothing -- and the Shape Index Spiral

- **Objective**: the operator ran the smoother on `ethene3_atom1_params1.json` and
  reported it "way too computationally intense", plus two structural observations:
  knots appearing inside the MC2 excision boundary where there should be a single
  pinned centre, and a missing knot between MS3 and MS2. Proposed: pick a few points
  along the boundaries between saddles/knots/excision regions/maxima and put one
  smooth curve through them, at spacing of MULTIPLE grid pitches.
- **The profile, before touching anything.** The connect stage on that snapshot:
  **172 s**, of which `_deviation_to_polyline_deg` is **164.5 s (96%)**. Isolating
  the pieces on one 445-vertex path: `smooth_sawtooth_runs` 16.61 s, the same call
  with the metric stubbed to `0.0`: **0.037 s**. The geometry was 0.2% of the cost;
  the diagnostic I added in Entry 061 was 99.8% of it. The operator's instinct that
  "smoothing is the most intense part" was right about the wall clock and wrong
  about the cause -- and the fix is not a faster smoother.
- **Why the closed form was still slow.** Entry 061 replaced a 0.05-deg sampled
  walk with an exact per-segment formula, which fixed the ALGORITHM (a unit test
  went 55 s -> 8 s) and left the IMPLEMENTATION as a Python loop over points x
  segments calling `np.cross`/`np.dot` on 3-vectors. At 293 output points and 444
  segments that is 130k iterations of numpy call overhead: 2.2M `np.cross` calls
  profiled at 32 s of self time, 11M `normalize_axis_tuple` at 24 s. Vectorised to
  (N, S) blocks: **18.1 s -> 0.0067 s on the same data, 2700x**, agreeing with the
  loop form to 2.8e-14 deg. Pipeline: **172 s -> 4.95 s**. The lesson is the one
  this project keeps re-teaching: measure the cost, then measure the cost of the
  measurement.
- **The two snapshots are different problems, and that is the real finding.**
  `params.json` is **curvedness** (paths 57-129 vertices); `params1.json` is
  **shape index** (paths 440-445 vertices). Shape index: path length 988 deg,
  endpoints 121 deg apart -- **8.1x the straight separation**. The path is not
  chattering across a crest; it is SPIRALLING into the maximum. Inside MC2 (radius
  6.13 deg) it spends 371 of 441 vertices and 825 deg of travel winding 0.5 turns
  around the centre, at a near-constant 4.5 deg from it.
- **Observation 1, "knots inside the excision boundary": confirmed, and it is not
  the smoother's doing.** E-SAD_MS3 br1 carries **31 knots, all 31 inside MC2**,
  spaced 5-33 vertices apart. The vertex nearest the MC2 centre (v402, 2.19 deg
  off) is NOT a knot. The smoother never creates knots -- `detect_corners` does,
  and it is being fed a spiral whose windowed deflection exceeds 150 deg at **162
  of 441 vertices**. A spiral is a sustained, uniform, huge deflection, so the
  median-baseline test that separates a corner from a sawtooth has nothing to
  separate: the whole winding is "above baseline". The pinned-centre mechanism
  exists and works (the transit centre is pinned), but it pins ONE vertex and the
  corners are promoted independently of it. Inside an excision region the field is
  declared untrustworthy by construction, so a corner detected there is measuring
  the interpolant, not the ridge -- the detector has no notion of the regions.
- **Observation 2, "missing knot between MS3 and MS2": confirmed, and the
  mechanism is the same test failing in the other direction.** The bend at the MS3
  exit peaks at 43 deg of windowed deflection (v33-35), against a threshold of
  baseline(3.2) + 35 = 38.2 -- it clears the threshold by 5 deg and is not
  promoted, while the spiral's 150-180 deg deflections are. The median baseline is
  computed over the WHOLE path, so a path that spends 84% of its length inside a
  spiral has its baseline set by the spiral, and the one real bend at the launch
  port is judged against it. Both observations are one defect: **the corner test's
  baseline is a global statistic on a path that is not globally homogeneous.**
- **The operator's proposal, measured.** Coarse anchors spaced k pitches along the
  traced path, one SQUAD through them, on the 441-vertex shape-index path (traced
  total turning 66010 deg):

  | spacing | output pts | total turning | max dev from traced |
  |---|---|---|---|
  | 1.0 pitch | 3513 | 65935 deg | 0.34 deg |
  | 4.0 pitch | 193 | 1806 deg | 1.60 deg |
  | 8.0 pitch | 89 | 651 deg | 2.66 deg |
  | 12.0 pitch | 65 | 532 deg | 2.77 deg |

  At 1 pitch the curve reproduces the spiral almost exactly (65935 of 66010 deg) --
  it is a faithful copy of a path that should not be followed. At 4+ pitches the
  winding is averaged away and the turning drops by 36x. This is the quantitative
  form of the operator's intuition: **the spacing is not a density knob, it is the
  low-pass filter's cutoff.** 0.45 pitch (Entry 061's choice) is below the spiral's
  own scale, which is exactly why "the smoothing helped but was applied over too
  small a region".
- **What this means for the design, and what it does not.** The smoother is correct
  for what it was built for: mesh-scale chatter on a path that is otherwise the
  ridge (curvedness, 4-6 runs, 3118 -> 1922 deg). It is the wrong tool for an
  8x-overshoot spiral, where the thing to remove is not teeth but the winding
  itself, and where the right description is the one the operator gave: a few
  nodes at the structural boundaries (saddle, knot, excision entry/exit, maximum)
  and one curve through them. That is a DIFFERENT object from a smoothed ascent --
  it is a rendering of the connection, and the deviation-from-traced metric stops
  being the right cost function for it, because the whole point is to depart from
  the traced path by many pitches.
- **Open next** (needs the operator's decision, not a constant):
  1. Corner detection must be scoped to the region it is inside: no knots promoted
     inside an excision boundary (the field is declared untrustworthy there), and
     the median baseline computed per-segment, not globally, so a spiral cannot set
     the bar for the launch port.
  2. Choose the target: (a) keep the ascent-fidelity smoother at 0.45 pitch and add
     a separate coarse structural curve for rendering, or (b) move the whole
     refinement to coarse spacing and accept 1.6-2.7 deg deviation from the traced
     path. The answer depends on what the 3D representation is for, which is the
     operator's call.
  3. The shape-index spiral itself is a finding about the field: 8.1x path length
     over endpoint separation is the ascent orbiting, not the ridge winding. Worth
     checking against the separatrix definition before smoothing it away.

---

### [Entry 063] Two Curves, Not One: the Coarse Structural Curve, and a Snapshot That Vetoed the Plan

- **Objective**: the operator's decision from Entry 062 -- replace the refinement
  with a coarse structural curve (one smooth curve through nodes at the structural
  boundaries), fix the corner baseline locally, node spacing 8 mesh pitches.
- **The plan was vetoed by evidence, and the evidence was a file on disk.** The
  Entry-061 key-reuse decision rested on a premise I stated to the operator and
  never verified: "every snapshot on disk has this control OFF." Both ethene3
  snapshots have `refine_ridge_paths = True`, saved at 21:22 and 21:41 UTC -- after
  Entry 061, recording the fidelity smoother. Overloading the key would replay
  those files with geometry that was never on screen, which is the one thing a
  snapshot must not do. So the coarse curve is a NEW control
  (`coarse_ridge_curves`, schema v6 -> v7, v6 admitted as legacy), not a new meaning
  for an old one, and `refine_connections(coarse=...)` defaults to False. The
  operator's "replace it" became "add it alongside", because replacing it would
  have destroyed the record of what was measured.
- **The proof, before and after.** The fidelity smoother reproduces the saved
  curvedness stats EXACTLY: `n_refined 4, n_refined_runs 6, total_turn_after_deg
  5987.902, max_refine_shift_deg 0.898` -- 4/4 to the last digit, and the v6
  snapshot replays **85/85**. Running the Entry-061 code against the shape-index
  file reproduces its saved `n_refined_corners = 93` exactly. The 2 residual drifts
  on that file (93 -> 30 corners, turn_after 169619 -> 170200) are the intended
  baseline fix, not a regression: 93 was the knot storm.
- **`coarse_structural_curve`** (ridge_refiner): nodes spaced `NODE_PITCH = 8.0`
  mesh pitches by ARC LENGTH (not vertex index, so the spacing is a physical
  distance), plus the endpoints, every knot, and every caller-pinned centre, joined
  by one SQUAD tessellated at `CURVE_OUTPUT_PITCH = 1.0`. Pure geometry, no field,
  deterministic. `RidgeRefinement` gains `n_nodes`; `n_runs` stays, and each mode
  fills only its own field, so a snapshot's recorded numbers mean what they say.
- **The spacing is a low-pass filter's cutoff, now asserted in a test.**
  `test_coarse_curve_removes_the_winding` pins the measured behaviour: at 1-pitch
  node spacing the curve reproduces the winding (65935 of 66010 deg on the real
  shape-index connector; >90% of the traced turning on the fixture), at 8 pitches it
  filters it out. A fixture lesson: `_winding_then_kink` is a smooth small circle
  (14 deg/vertex) and does NOT reproduce the property -- the real spiral winds at
  150 deg/vertex. The deep sawtooth (amplitude 2.0 pitch) does, because 1-pitch nodes
  land on the teeth.
- **Measured, shape index (the spiralling field):** 4 connectors, 256 nodes, 30
  corners kept sharp, **263888 -> 17877 deg** total turning (92-94% per branch),
  445 -> 147 vertices, deviation 0.82-1.03 pitch. Audit OK, 0 degenerate arcs.
  **Curvedness (the field that already worked):** 73-95% turning removed, deviation
  0.46-0.52 pitch, 129 -> 75 vertices.
- **`_p` now defaults a missing control, mirroring the loader.** The stage tests
  build a params dict in code carrying only the controls their fixture cares about,
  so a new control read by a stage broke them. The fix is the project's own legacy
  rule, applied in one place: a control absent from the params reads its default,
  because a file predating a control shows it off. Not a `KeyError` at every call
  site, and not a silent `.get(key, False)`, which would hide a typo in a control
  name forever -- `DEFAULT_UI_STATE[key]` raises loudly for a control that does not
  exist.
- **Tests**: 621 pass (was 612). +8 coarse-curve unit tests (few nodes, winding
  filtered, endpoints/knots interpolated exactly, knot kept sharp, geodesic
  reproduced to 0.05 deg, frame independence, determinism, transited centre pinned),
  +1 v6-legacy replay test, and 3 `refine_connections` tests rewritten for the
  two-mode contract.
- **Open next**:
  1. The fidelity smoother is the default and the coarse curve is opt-in. If the
     coarse curve becomes the primary rendering, the default flips and the v6
     snapshots need migrating, not just admitting as legacy.
  2. `NODE_PITCH = 8` is the operator's choice, measured at 4 and 12. The curve's
     deviation from the traced path is informational, so there is no objective
     optimum -- the right value is the one that keeps every bend the operator wants
     to see. It is a slider candidate.
  3. The knot storm inside excision regions is reduced (93 -> 30) not eliminated.
     The remaining 30 are all inside regions. Suppressing promotion inside an
     excision boundary is the clean fix and the operator declined it in favour of
     the local baseline; it stays open.

## 2026-11-10 -- Coarse-curve toggle never fired a recompute; the card's output reaches `diagnose_params.py`

- **Symptom (operator report, `ethene3_atom1_params.json`)**: flipping "Coarse
  Structural Curves (Cyan Ridges)" did nothing. The cyan curves only changed after a
  second, unrelated toggle (e.g. "Smooth Sawtooth Chatter") fired.
- **Root cause**: `update_topology` reads `state.coarse_ridge_curves` correctly, and
  the control is in the card, the schema (v7 default False), and the snapshot writer.
  What was missing is the subscription: `on_param_change` is decorated
  `@state.change("refine_ridge_paths", "probe_ridge_points", ...)` and
  `coarse_ridge_curves` was never in that list, so the toggle changed no watched key
  and the handler never ran. The value stayed set, so the NEXT watched change (any
  other control) rebuilt the scene and picked it up -- which is exactly the reported
  "toggle it, then toggle something else".
- **Fix**: one line, `sphere_viewer.py:3578`. The list is the registry: a control
  that changes the analysis must appear in it.
- **Why it recurred**: a feature spans 7 sites (card, state key, DEFAULT_UI_STATE,
  schema key + version, snapshot, pipeline read, subscription). The first six are all
  needed for the value to save and load, so a snapshot replays correctly and
  `diagnose_params.py` agrees with the viewer -- which is precisely why the parity
  test suite passed while the GUI toggle was dead. The 7th is invisible to every
  replay-based check.
- **Guard** (`tests/test_sphere_viewer_param_triggers.py`, 5 tests, AST-based): every
  `DEFAULT_UI_STATE` key must be in the `@state.change` list; no duplicate keys;
  every watched key must be a known control or context string; the handler must
  actually call `update_field()`; and the three Step-3 ridge switches must be watched
  side by side. Parsing the source is the point -- the decorator lives in the
  `run_sphere_viewer` closure, so importing it to read it would need a render window.
  Verified 2 of 5 fail with the fix reverted.
- **`diagnose_params.py`**: the connect stage already called
  `refine_connections(..., coarse=bool(coarse_ridge_curves))` faithfully, and
  `summarize_connections` already aggregated the post-pass table -- the ethene3
  replay reproduces 86/86. What was missing was the card's OWN output:
  1. `connection_ridge_details` / `summarize_ridge_postpass`: per-connector
     `refined`, `refine_max_shift_deg`, `n_nodes`, `n_runs`, `n_corners`,
     `total_turn_{before,after}_deg`, plus a footer (n_refined / n_coarse /
     n_smoothed / max shift / node count).
  2. `stage_probe` (new stage, after `connect`): `probe_ridge_points` changes NO
     geometry, so no other metric can show it. Runs `probe_gradient_ridge` over every
     traced ascent and publishes the viewer's own `ridge_probe_stats` names. It reads
     `ctx.traced_connections`, mirroring the viewer's `runtime_traced_connections` --
     the probe measures the ASCENT, not the smoothed line drawn beside it.
  3. `--out` now writes `connection_ridge_details`, `ridge_probe_stats`,
     `probe_details`, and the 7 ridge fields on every `max_connections` row.
- **Measured signature** (ethene3 atom 1, saved with coarse ON / smooth OFF / probe
  OFF; both are off by default, so `_p` + `--set` drive them):
  | switches | n_refined | n_coarse | n_curve_nodes | max shift |
  |---|---|---|---|---|
  | both off | 0 | 0 | 0 | 0.0 |
  | coarse | 4 | 4 | 38 | 1.697 |
  | smooth | 3 | 0 | 0 | 0.871 |
  Probe on: 4 gradient connectors, 4 knots, 8 accepted points, 79 refused, median
  offset 0.387 mesh pitches. `refined` alone cannot separate the two switches --
  `n_nodes` vs `n_runs` can, which is why both are recorded.
- **Tests**: 639 pass, 1 skipped (was 621 + 17 new: 5 trigger-registry, 12 card
  output). The reproduce check stays 86/86.
- **Open next**:
  1. No automated check launches the trame app and flips a switch, so the registry
     test is the only guard on the subscription list. A headless `server.state`
     smoke test (set a key, assert the handler ran) would cover the handler-body
     failure modes this test cannot see.
  2. `sphere_viewer.py` carries 57 pre-existing pyright strictness findings (VTK
     `vtk_to_numpy` dtype unions, `get_server()` returning `ServerProxy | None`);
     `mypy` per `pyproject.toml` is clean. They predate this change (verified by
     toggling the fix on and off) and are untouched by it.

## 2026-11-11 — Probe Ridge Points: the measurement now travels with the switch

**Task.** Make the "Probe Ridge Points (Diagnostic)" output appear in the saved
parameter JSON, and make `diagnose_params.py` reproduce it.

**What the file actually held.** `probe_ridge_points` was already saved — the
*request* was there, and the *answer* was not. The drawer computed
`state.ridge_probe_stats` (connections, knots, accepted points, refusals, median
offset) and `PARAM_METRIC_KEYS` did not list it, so every snapshot dropped the
measurement and died with the process. A saved file is the citation for an
observation; the citation had the question and not the finding.

**Design: one aggregator, not two.** The counters moved out of the drawer's loop
into `sphere_viewer.make_ridge_probe_stats(rows, on=, basin_id=, saddle_id=)`,
which the viewer calls and `diagnose_params` *imports*. The rule that decides
"refused" (`n_probe == 0`) and the median live in one place, so a saved number
and a replayed number cannot drift by a sum. `offsets` counts what it averages
(`probe_max_offset_pitch` was summarised as a mean and never averaged).

**Scope, named.** The probe measures the arcs on screen, so the record carries
`scope_basin_id` / `scope_saddle_id` and the replay follows the same rule. Without
the ids, `n_connections: 4` cannot be told apart from 4 of 928. The saved selection
gained `selected_saddle_id` — a stalled ascent has no maximum to select, so a
saddle-scoped measurement was unreproducible from a file. Schema 7 → 8, v7 listed
legacy; v7 files load, and their absent scope degrades to `None` (never invented).

**Replay.** `stage_probe` scopes to the saved selection by default — reading the
display table and looking the ascent up by `(saddle_id, branch_index)`, the viewer's
own `traced_by_key` — with `--probe-all` for every gradient connector in the run.
Scope filtering is skipped only in the mode that ignores scope.

**Measured on `ethene3_atom1_params.json`** (1.03 GB PLT, ~13 s/run). The v7 file
stores `probe_ridge_points: false`, so a faithful replay reports an honest all-zero
probe. With `--set probe_ridge_points=true`: 4 connections, 4 knots, 8 accepted
points, 79 refused, median offset 0.38729 mesh pitches. Scoped to the file's own
`MAX_4864` and whole-run agree exactly, because every gradient connector in this run
belongs to that basin — the two paths cross-check each other on real data.

**Tests.** 651 pass, 1 skipped (was 639 + 12 new: shared-aggregator semantics,
selection scoping, `--probe-all`, `_scope_id` v7 degradation, export round-trips).

**Incidental.** Every pre-existing pyright site in the functions this change touched
is fixed: 57 → 36 (trame `server`/`state` None narrowing with a named failure,
`GetArray` name coercion, `vtk_to_numpy` dtype casts, and a `_state_number` helper
that names the control a bad value came from). The 36 left are `VCardItem`/`VTable`
stub gaps and the `runtime_* = None` init family — `mypy` is clean on all of them.

**Open next.**
1. The probe's counters are compared as a dict; a saved v7 file has none, so the
   first true probe-to-file parity check needs a v8 snapshot from the app.
2. `--probe-all` and the scoped path are equal on this file by construction; a
   multi-basin field (Pd_20K) is the case that separates them.

## 2026-11-11 — Pyright: 66 findings to zero, and what they were really hiding

**Task.** Clear the pre-existing pyright findings recorded above (57 in
`sphere_viewer.py` at the start of this session), with ruff/mypy/tests green.

**Measurement first.** No pyright config existed, so the editor was running default
settings with no venv resolution — and the file is over the pi-lens 5,000-line probe
cap, so the lens counts were the only signal. Added `pyrightconfig.json` (venv
`.venv`, `typeCheckingMode: basic`, `include` set to exactly the three directories
`[tool.mypy] files` covers) and ran pyright 1.1.414 through `uv tool run`. Baseline:
**66 errors in 8 files** — 36 in `sphere_viewer.py`, 30 in the inspect/verify scripts
and one test.

**The widget family was not a stub problem.** 19 errors said `"VCardItem" is not a
known attribute of module "trame.widgets.vuetify"` — note **vuetify**, not vuetify3.
`trame_vuetify` defines every component; the real cause was this file's own
`try: from trame.widgets import vuetify3 as v3 / except ImportError: ... import
vuetify as v3`. pyright types `v3` as the union of the two modules, which exposes
only what BOTH have, so every v3-only component failed at every call site. Fixed once
at the import: the v2 fallback is now `v3 = cast(Any, _vuetify2)`, with a comment
saying why. No stubs, no per-site ignores.

**The `state` family was one type.** 14 sites read `state.<attr>`, which trame types
`Any | None` — the `None` cascading into vtk calls, resolvers and `for` loops. Added
`_state_number`, `_state_list`, `_state_dict`, `_state_str` beside the module's other
helpers: each narrows to the shape the consumer declares, treats an unset control as
empty, and names the control when the value is the wrong type. The previous
`float(state.k_pitch)` raised a bare `TypeError` naming nothing; now it says
`trame control 'k_pitch' is not a number: 'abc'`.

**One behavior change, on purpose.** An unset `selected_field` reached
`GetArray(None)` and raised inside VTK. It now passes `""`, `GetArray` returns `None`,
and the caller's existing `if not arr: return` fires — the missing-array path, which
is what an unset field is.

**The scripts family was one pattern.** 30 sites handed `vtk_to_numpy(...)` — whose
signature admits `uint8` — to parameters typed `NDArray[float64]`/`NDArray[int32]`.
Wrapped in `np.asarray(..., dtype=...)`, matching the `.astype(np.float64)` convention
already used in those files, `int32` for cell arrays. `np.asarray` is a no-op view
when the dtype already matches, so the read-only pipelines pay nothing.

**Result.** pyright 0 errors / 65 files, mypy clean, ruff clean, **651 passed, 1
skipped**, `sphere_viewer` imports, accessors smoke-tested on the shapes trame
delivers. pyright is NOT wired into `.pre-commit-config.yaml` — the project gate
stays mypy + pytest; the config just makes the editor's checker reproducible and
CLI-runnable.
### [Entry 064] The Config That Made "0 Findings" Honest -- and What "Passing" Means Here

**User request (m00691):** "fix the pre-existing pyright findings you mentioned earlier. Also make
sure that ruff/mypy//tests are properly passing."

This entry is about the second sentence more than the first. Clearing 66 findings is mechanical;
establishing what "passing" means in a repo whose root has six top-level packages is not, and the
tooling here disagrees with itself about the answer.

**`[tool.mypy]` has no `files` key.** I asserted in the journal that my `pyrightconfig.json` include
matched "the directories `[tool.mypy] files` covers" -- there is no such setting. The actual
`[tool.mypy]` section is `python_version = 3.12`, `warn_return_any`, `warn_unused_configs`,
`disallow_untyped_defs`, `disallow_incomplete_defs`, `check_untyped_defs`, plus
`ignore_missing_imports`/`follow_imports = "silent"` overrides for vtk, vtkmodules, trame,
trame_vuetify, trame_vtk, pyvista, gudhi, networkx, BondalyzerParaView, bondalyzer_viewer. The
direct evidence I had ignored: `uv run mypy` with no arguments exits with "Missing target module,
package, files, or command" -- which is only possible if no `files` list exists. Claim corrected in
place; the claim was load-bearing (it justified a config value) and the correction is the config value.

**`pyrightconfig.json` v1 listed `scripts`, which does not exist in this repo** (only
`BondalyzerParaView/ analysis_output/ analysis_output_small/ bondalyzer_viewer/ gba_topology/
gba_topology2/ journal/ plots/ tests/` are at the root). Pyright prints
"File or directory .../scripts does not exist" and **still exits 0**, so the config looked accepted
while naming a phantom path. And its `$schema` key is "unrecognized setting" -- pyright validates its
own config and does not use the VS Code JSON-schema keyword. v2: include `["gba_topology2", "tests"]`,
no `$schema`, plus `venvPath`/`venv` so the CLI resolves imports the way the editor does.

**Why the CLI showed me 16 phantom errors at first:** `--pythonpath .venv/bin/python` sets the
*interpreter* but not the import search path, so every `import numpy` became
`reportMissingImports` and the real type errors never surfaced (an unread file yields no findings).
`--project .` with `venvPath`/`venv` is what produces the same picture the editor has.

**"Properly passing" is scope-relative, and the root is not clean.** The declared project (the
pre-commit `files:` regexes, and mypy's real behaviour with no target) is `gba_topology2/` +
`tests/`: the pre-commit config says so three times -- `files: ^(gba_topology2/|tests/)` on both
ruff hooks and `entry: uv run mypy gba_topology2 tests` on the mypy hook. Against that declared
scope: ruff passes, `ruff format --check` 67 files clean, mypy 65 files clean, pyright 0 findings,
651 tests + 1 skip. The root also carries `BondalyzerParaView/` (251 ruff findings), `gba_topology/`
(77), and `analysis_output*/`. `uv run ruff check .` finds 328 of them. All are in directories this
branch has not touched in 12 commits; every "clean tree" baseline in this session was measured on
them. I did not "fix" them: 251 findings in a vendored ParaView tool is a project, not a typo, and a
mass mechanical pass through code with no test coverage is how you turn a visible mess into an
invisible one. `uv run mypy .` fails for a structural reason worth knowing --
`BondalyzerParaView/plt_gba_to_vtm.py: Source file found twice under different module names
("plt_gba_to_vtm" and "BondalyzerParaView.plt_gba_to_vtm")`, no `__init__.py` in that directory,
plus `Cannot find implementation or library stub for module named "paraview"` (ParaView ships its
own interpreter; that file cannot be checked in this venv). That is why mypy needs targets and why
`gba_topology2/` + `tests/` is the honest answer to "the project".

**Gate state at commit:** pyright 0 / 0 (65 files). mypy: "Success: no issues found in 65 source
files". ruff: All checks passed. format: 67 files already formatted. pytest: 651 passed, 1 skipped.

## 2026-11-12 — Basins that render must be basins that can be read off the file (schema v9)

Operator report: with **Show All Minimum Basin Boundaries** on, SAVE PARAMETERS wrote a JSON with
no saddle->max graph and no minimum->boundary association -- the drawn cycles were undiagnosable
from the file. Root cause: `save_params_to_dict` exports the fixed `PARAM_METRIC_KEYS` list
(7 keys); `connection_graph`, `minimum_basins`, `max_connections_list` and `basin_centers_list`
were simply not in it. `diagnose_params.dump_results` DID carry two of them, in two ad-hoc shapes
and only when non-empty, so the two files were not diffable and a missing key was ambiguous
between "stage never ran" and "topology found nothing".

**The fix is a schema bump, not an exporter patch.** v9 (legacy set gains 8) adds the four keys
to `PARAM_METRIC_KEYS` -- they are METRICS, not controls: an absent section in an older file means
the feature did not exist, never a computed-empty. `basin_boundary.serialize_connection_graph` /
`serialize_minimum_basins` are now the ONE shape, shared by the viewer export and the results dump
(`selected_basin_center_id` set the same way for both). The dump emits them unconditionally.
`selection` gains `selected_minimum_id`. The viewer writes the serialized dict the moment
`compute_and_show_basin` computes the cache, and the arc-table rebuild clears cache + both metric
keys, so a saved file can never carry a graph lagging its arcs. `state.connection_graph=None`
initialization distinguishes "never computed" (key absent) from "computed empty" (`{}`).

**Same-day basin fixes** (pre-v9, commits 07a80e2..57b4cc6): `find_minimum_basins` was keying
basins by a truncated CP vertex id (`5011`) so the selection callback's `MIN_5011` always missed
the cache -- the mapping now preserves the full id; the graph includes unverified CPs (verified
ones win per id) and derives saddle positions from `arc_points[0]` because the CP table has no
unified id space; monkey saddles join basin adjacency as all 3 member saddles;
on_selection cp-click restores the saddle end (`selected_saddle_id`/`is_max` in
`_apply_cp_picked`) so LOAD PARAMETERS reproduces selection-scoped probe stats; `apply_probe_display_gates`
now writes back the gated arcs (`max_arcs_list`, `max_arcs_saddle_endpoint_list`).

**Verification:** `diagnose_params.py ethene3_atom1_params.json --out ...` reproduces 94/94 against
the v8 file and the dump carries the graph (2 saddles/4 arcs) and 3 minimum basins (4 boundary
arcs each, all around MAX_4864 -- the three minima of this field share one basin ring).

**Gate state at commit:** mypy clean on the touched modules; ruff format/check clean (scoped to
`gba_topology2/` + `tests/` -- repo-wide `ruff format .` churns the legacy `BondalyzerParaView/`
and `gba_topology/` dirs, which are not formatted to this standard; those edits were reverted via
stash). pytest: 686 passed, 1 skipped.

## 2026-11-13 — The null that ate the basin graph (schema v9 fallout, fix)

Commit range: this entry + fix. Files: `src/basin_boundary.py`
(`connection_from_state_dict`), `src/sphere_viewer.py` (reconstruction block
replaced), `tests/test_basin_boundary.py` (+10 tests, ethene3 replay fixture).

The v9 export did its job: the user's saved params showed
`connection_graph: {}` + `minimum_basins: {}` beside a full four-row
`max_connections_list` — the arcs exist, the graph built from them does not.
Replaying the viewer's dict→`MaxConnection` reconstruction headlessly on the
actual saved rows reproduced it: every row carries `"cross_track_deg": null`
(None is the DOCUMENTED value for "the geodesic ray hit nothing"), the
hand-written `float(conn_dict.get("cross_track_deg", 0.0))` runs `float(None)`
— `.get`'s default fires only on MISSING keys, not on present nulls — and the
per-row `except (KeyError, ValueError, TypeError): pass` swallows all four
arcs. Empty conn_objs → empty graph → cached `{}` → no rings. "Show Arcs From
Selected Maximum" renders the dicts directly, so it kept working; only the
path that needs typed objects died, silently.

Fix: `connection_from_state_dict(d)` — one rehydrator driven by
`dataclasses.fields(MaxConnection)` + `get_type_hints`, so it cannot drift
from the dataclass. Contract: present-null on an Optional field passes
through (null is data); null/missing on a defaulted field gets the default;
null/missing on a required non-Optional raises. The caller logs dropped rows
per-row (`[sphere_viewer] basin topology: dropped arc ...`) — silent drops
were the second bug; a 4-of-4 reconstruction failure must be louder than an
empty field.

Gotcha found by the tests: `get_args(tuple[tuple[float,float,float], ...])`
is `(tuple[float,float,float], Ellipsis)` — taking args[0] as the base type
coerces arc_points as a flat stall_point. Only treat `X | None` unions as
Optional (origin `UnionType`/`Union`); plain annotations keep their outer
type as base.

Replay test is the user's screen as a test: the four saved ethene3 rows
(arc polylines trimmed, endpoints kept) + four CP rows → 4 arcs → 2-saddle
graph → 3 basins × 4 boundary arcs around MAX_4864.

## 2026-11-13 — The descent connector: basins named by where they drain (basin v10, Units A+B)

Commit range: 92bcf9a (`trace_downhill`) + this commit (descent connector).
Files: `src/separatrix_tracer.py` (`trace_downhill`),
`src/max_connection.py` (`_prepare_targets`, `_BranchWalk`, `_walk_branch`,
`_classify_gate` arrival_kind, `MinConnection`, `connect_seed_to_minima`,
`_min_no_target`, `_gradient_descent_connection`),
`tests/test_separatrix_tracer.py` (+12), `tests/test_max_connection.py` (+9).

**Why.** The v9 basin boundary used K-NEAREST PROXIMITY to associate minima
with saddles, and it saturated: on ethene3 every basin claimed the same 3
arcs of 3 (2 minima, 2 saddles: k=2x3=6>3), every ring passed through both
saddles, and a real structure was reported as "K too small; raise to 4-8".
It was not a parameter. A minimum's basin is bounded by the stable manifolds
of saddles whose DESCENT runs into that minimum — connectivity, not
proximity. The association the rule needs is measured off the field itself,
and the instrument already existed: the ascent connector (journal 2026-11-12
entries) walks a saddle's ASCENT branches by great-circle ray with cone
capture on maxima, directional-disc occlusion on everything else, and a
gradient climb as fallback. The descent is its mirror image with the arrival
polarity inverted.

**Unit A — `trace_downhill` (separatrix_tracer).** A THIN wrapper: negate the
field (`f_grad = -g`), relabel the minima in the pool as maxima
(`replace(t, polarity="maximum", value=-t.value)`), and call `trace_uphill`
verbatim. The integrator, the stall rule, the TOTAL completion and the
transit record are inherited, not forked — a bug fixed in the ascent is fixed
in the descent the same second. Parity is pinned by a test: on a pool of
relabelled minima, `trace_downhill(g) == trace_uphill(-g)` field for field
(`stall_point` is the raw-field endpoint, so it negates between the two).
Negated-peak well: descends onto the pole from 36.9 deg out in 64 steps;
flat plateau: no-op stall; negated ridge: descends to the flanks.

**Unit B — the descent connector (max_connection).** The ray stage of
`connect_seed_to_maxima` was extracted, verbatim, into
`_prepare_targets` / `_walk_branch(seed_id, ray, prepared, cone_rad,
max_arc_rad, arrival_kind)`; `_classify_gate` took the same `arrival_kind`
parameter (arrivals are scored directly off the pool; the pre-filtered
candidate list cannot be used because the entries it EXCLUDES are exactly the
ones being diagnosed). `connect_seed_to_minima` is the same loop over
`seed.descent_branches` with `arrival_kind="minimum"`, emitting
`MinConnection` records — the ascent fields minus `mutual` and the smoother
stats (reciprocation is a maximum-side statement: a minimum claims every
inflowing saddle, there is no tightest), plus `min_id`/`min_value`. Unresolved
branches fall via `_gradient_descent_connection`, bridged saddle-centre to
port exactly like the ascent: a gradient path is never left uncommitted.

**Semantics pinned by tests.** Cone capture on minima; the nearest wins and
pool order cannot matter; a maximum ON the meridian occludes like any
non-arrival whose disc the ray enters, while one OFF the cone leaves the gate
`empty` (the pool has no minima — not "cone"), which is the polarity
inversion stated as a gate; a well behind the port gives the geodesic rule
`behind` and the fall arrives (method `gradient`, gate kept, value at the
minimum is the RAW field value); omitting `mesh_pts`/`f_vals` leaves the
pure-geodesic contract untouched (the Pd_20K no-op). 73 tests in the file;
the 65 ascent tests pass unmodified across the extraction — the refactor is
behavior-preserving by construction.

**Next (Unit C).** params_schema/diagnose_params: the basin_bounded_by rule
becomes graph-adjacency ("the boundary of minimum M's basin is the
saddle->max arcs at the saddles whose descent branches reach M", from
`_minimum_basins` + `connect_seed_to_minima`), replacing the proximity tier
and its k advice. The k-nearest list itself stays — it remains the right
answer for the "closest maxima" question; the error was ever feeding it to a
boundary rule.

## 2026-11-13 basin v10: boundaries from descent connectivity (Units C-E)

The consolidation landed. `find_minimum_basins` now takes the descent table
(`connect_seed_to_minima` output, Unit B) and bounds a minimum's basin by the
saddle->max arcs at the saddles whose descent branches DRAIN into it. The v9
rule -- k=3 spatially nearest saddles by Euclidean distance -- is deleted with
`_spatial_distance` and every trace of the k advice; there is no proximity
fallback, because proximity is exactly what the rule replaced.

The viewer computes the descent connector on every arc rebuild, on the RAW
pre-refine table (refine is an ascent-side concept; the drain says where water
goes), from the same pool, regions and field as the ascent. `state.min_connections_list`
(rows from `min_connection_to_state_dict`, arc geometry included -- the v9
lesson: a saved observation must carry everything the render reads) and
`state.descent_connection_stats` join `derived_metrics` as schema **v10**; v9 is
admitted legacy, and a v9 file's replay honestly reports its basins as
un-derivable-by-v10 rather than reconstructing them from the rejected rule.
The basin pipeline rehydrates the rows through the generalized
`connection_from_state_dict(d, MinConnection)` -- dataclass-driven, None-safe,
loud per-row drops (the 4606cac pattern; the descent rows carry the same
Optional-nulls that emptied v9's arc table).

`diagnose_params` gained the adjacency in `stage_connections` and rewrote
`stage_minimum_basins`: per-minimum ring counts, `n_descent_{connected,
occluded,no_target}`, `rings_covering_every_arc` (normal when the adjacency
supports it -- two saddles that drain everywhere SHOULD share every arc), and
`minima_without_basins` naming the failure gates of the unresolved drains.
The saturation signature the v9 panel answered with "raise k to 4-8" is now a
measurement, not an accusation.

The ethene3 replay states the shape of the fix: with a consistent 2+2-edge
descent topology the rings are 4/2/2 arcs -- E-MIN_MC3 bounded by both saddles,
each bare minimum by the one that drains into it. v9 gave every basin all 4
arcs and every ring passed through both saddles, and called it a k problem.

Gate: 729 passed, mypy clean, ruff clean, LSP probe on the three edited sources
shows only the pre-existing baseline advisories.

## 2026-11-13 — A red gate that ran nothing: shipping the source half a committed test file imported

Basin v11's record/replay half is now closed end to end. `99d1255` added the
maximum-side dual in `basin_boundary.py` (`MaximumBasin`, `find_maximum_basins`,
`serialize_maximum_basins`) and `1d33109` made it recordable and replayable:
`PARAMS_SCHEMA_VERSION` 10 -> 11, `maximum_basins` in `PARAM_METRIC_KEYS` and in
`_PARAMS_SCHEMA`, `stage_minimum_basins` computing both sides, and
`check_minimum_basin_topology` returning a `(minimum_side, maximum_side)` tuple so
one rule fires per polarity. The asymmetry worth remembering: a *maximum's* basin
is bounded by its **descent** arcs and its adjacency comes from the **ascent**
table, exactly inverted from the minimum side -- the ring around a hill is made of
the streams that run off it, and the hill's neighbours are the saddles that climb
into it. `connect_seed_to_minima` needs the same two tables, which is why the dual
was ~40 lines and not a port.

This commit is the viewer half, and the reason it exists is the more important
record. `1d33109` committed a 658-line `tests/test_sphere_viewer_basin_groups.py`
that imports nine names -- `MAX_BASIN_GATE_DEG`, `MIN_BASIN_GATE_DEG`,
`MAX_BASIN_RING_COLOR`, `MIN_BASIN_RING_COLOR`, `RAYCAST_OCCLUDERS`,
`basin_draw_order`, `basin_ring_color`, `classify_basin_connection`, `occludes` --
which exist in **no commit**. They live only in the working tree, in the in-progress
v11 viewer refactor. Consequence at `1d33109`: pytest dies during collection
(`763 tests collected, 1 error`), the run aborts, and **zero tests execute**. Five
further failures sat behind it in `test_sphere_viewer_params_load.py`, all fixed by
the source being committed here.

The lesson is about the gate, not the code: a suite that cannot collect does not
fail loudly, it reports *nothing*, and `git commit` succeeded anyway. Two rules
follow. (1) When a commit adds a test file, run that file in the commit's own tree
before calling the gate green -- `git worktree add /tmp/bond-base HEAD` then the
main venv's `python -m pytest` gives a clean read of HEAD without disturbing a
shared working tree. (2) In a tree two agents are editing at once, `-S"def name"`
on the imported symbol is a 5-second check that catches "test shipped, source did
not".

Rule (1) has a trap worth writing down, because it produced four phantom failures
on first use: a fresh worktree has no **git-ignored** data, and this suite reads
`*.plt` field files (`ethene3/4/5.plt`, `Pd5k/Pd20k.plt`,
`ethene4_1d_zones.vtm` -- `*.plt` is `.gitignore:13`). Without them
`test_sphere_viewer_params_load.py` raises `FileNotFoundError: PLT file not found:
/private/tmp/bond-base/ethene4.plt` from `sphere_viewer.py:728` and reports 4
failures / 7 skips on a commit that is actually green. Symlink the ignored data in
first, then trust the number: 829 passed, 1 skipped at `f1d97c7`.

Viewer changes here: the module-level basin logic the tests name (occlusion rule,
draw order computed over the full set before anything draws, the classifier that
feeds both the render gate and the warning text so the message cannot contradict the
screen, `is_basin_only_change` routing all four basin switches to
`_refresh_basin_overlays`), plus one fix that was mine -- the `no_basin` return of
`classify_basin_connection` omitted `min_sep_text` and `gate_text`, which the panel
template binds unconditionally, so the two boxes rendered blank in exactly the state
where a reader wants a number. Pinned with a key-set contract computed as the UNION
over every status, because a single-path assertion cannot see this class of bug.

Known gap, deliberately not touched: `select_basin_center` (sphere_viewer.py:3993,
3995) calls `compute_and_show_basin` / `compute_and_show_max_basin` per side behind
conditionals, which is the one shape `_refresh_basin_overlays`'s own docstring
forbids -- a one-sided refresh half-updates the occlusion pair. The new call-site
guard lists it as a known site with a comment; the fix belongs to whoever owns the
selection handlers.

Gate: 829 passed, 1 skipped (HEAD before this commit: collection error, 0 executed).
