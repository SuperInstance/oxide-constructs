# oxide-constructs

> **Git-native construct loader for the Flux→PTX distributed GPU runtime.**

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Rust Edition](https://img.shields.io/badge/rust-2021-orange.svg)](https://rust-lang.org)

---

## What is a Construct?

In the age of agent-native computing, static binaries are too rigid. A **construct** is a self-contained unit of GPU capability—a kernel, a skill, or a piece of equipment—that lives in an ordinary git repository. Constructs can be discovered, loaded, verified, compiled, and deployed at runtime without restarting your distributed application. They are the atoms of a living, evolving GPU runtime.

This crate, `oxide-constructs`, is the loader and registry that makes that vision concrete. It is one piece of the broader [SuperInstance](https://github.com/SuperInstance/SuperInstance) ecosystem, which reimagines GPU programming as a dynamic, peer-to-peer negotiation of capabilities rather than a static ahead-of-time compilation step.

### The Two Families of Constructs

| Family | Description | Example |
|--------|-------------|---------|
| **Skills** | Software capabilities—kernels, shaders, compute graphs | A fused attention kernel with a `kernel_main` entry point |
| **Equipment** | Hardware requirements and declarations—SM version, VRAM, tensor cores | A declaration that "I need SM 8.0+, 4 GiB of VRAM, and tensor cores" |
| **Hybrid** | Both bundled together—a skill that carries its own equipment contract | A ternary matrix-multiply kernel that also asserts SM 9.0 and unified memory |

Skills are what your code *can do*. Equipment is what your hardware *must provide*. Hybrids let you ship both as a single, self-describing artifact.

---

## Why Git-Native?

Traditional GPU code distribution relies on opaque binary blobs, container images, or centralized package registries. These approaches work, but they create friction:

- **Version skew** across a fleet of GPUs is inevitable. Nodes run different driver versions, different CUDA toolkits, and different kernel builds.
- **Trust is hard.** A binary kernel offers no provenance. You cannot audit it, sign it meaningfully, or trace its lineage.
- **Hotswap is impossible.** Once a kernel is loaded into device memory, replacing it typically requires tearing down the entire process.

By making constructs **git-native**, we solve all three problems at once:

1. **Git is a CRDT.** Every clone is a full replica. Every tag is an immutable reference. A fleet of GPU nodes can gossip construct manifests using the same merge semantics that git uses for branches.
2. **Git is auditable.** A construct's entire history—who wrote it, when it was reviewed, what changed—is visible in `git log`. Identity is bound to the commit via DIDs (Decentralized Identifiers) and cryptographic signatures.
3. **Git is lazy.** Shallow clones, sparse checkouts, and LFS mean you only fetch the kernels you actually need, when you need them.

---

## How It Works

### The Construct Lifecycle

A construct moves through a well-defined state machine from discovery to deployment:

```
Discovered → Validated → Resolved → Compiled → Deployed → Cached
                ↑___________↓
```

| State | Meaning |
|-------|---------|
| `Discovered` | The construct's git repository has been located (e.g., via `flux-index` tag search). |
| `Validated` | The `CONSTRUCT.toml` manifest has been parsed and sanity-checked: non-empty name, declared compute capabilities, valid semver. |
| `Resolved` | All dependencies (other constructs this one imports) have been fetched and version-matched. |
| `Compiled` | The skill has been lowered to PTX via the `flux-importer` → `cuda-oxide` pipeline and cached as a binary blob. |
| `Deployed` | The PTX has been loaded into GPU device memory and is ready for launch. |
| `Cached` | The construct has been unloaded from the GPU but the PTX blob is retained in host memory for fast re-deployment. |
| `Failed` | Something went wrong—recorded with a descriptive message. |

This lifecycle is enforced by the `ConstructLoader`. Invalid state transitions—attempting to deploy before compiling, or unload before deploying—are rejected at compile time via Rust's type system and at runtime via precise error variants.

### The Registry as a Living Database

`ConstructRegistry` is an in-memory, CRDT-ready database of every construct known to the local node. It supports:

- **Registration and deregistration** of individual constructs.
- **Type filtering**—list only `Skill`, `Equipment`, or `Hybrid` constructs.
- **Deployment tracking**—query which constructs are currently resident on the GPU.
- **Tag search**—discovery via `flux-index` compatible tags such as `attention`, `reduce`, or `gpu`.
- **Fleet-wide merge**—combine two registries with last-write-wins semantics based on semantic version. This is the primitive that enables SmartCRDT propagation across a cluster.

When node A discovers a newer version of a kernel that node B already hosts, the merge operation automatically upgrades B's registry. The next deployment cycle will transparently pick up the improved kernel.

### Semantic Versioning with GPU Awareness

Constructs use a GPU-aware semver scheme. Compatibility is determined not just by API surface, but by **compute capability** and **SM version**:

```rust
let v = SemVer::parse("v2.1.3").unwrap();
assert!(v.is_compatible_with(&SemVer::new(2, 2, 0))); // same major, higher minor
assert!(!v.is_compatible_with(&SemVer::new(3, 0, 0))); // major bump = breaking
```

A construct declaring `compute_capabilities = [80, 86, 89, 90]` is explicitly telling the runtime: "I have been tested on Ampere, Ada, and Hopper. Do not attempt to run me on Turing or earlier."

---

## Quick Start

Add `oxide-constructs` to your `Cargo.toml`:

```toml
[dependencies]
oxide-constructs = "0.1"
```

Load a construct from any git repository:

```rust
use oxide_constructs::{ConstructLoader, ConstructType};

let loader = ConstructLoader::new();
let mut construct = loader
    .load_from_repo("SuperInstance/ternary-attention-kernel", "v2.0.0")
    .expect("manifest should be valid");

println!("Loaded: {} ({:?})", construct.manifest.name, construct.manifest.construct_type);

// Compile to PTX
loader.compile(&mut construct).unwrap();

// Deploy to GPU
loader.deploy(&mut construct).unwrap();
```

Query a registry for deployed capabilities:

```rust
use oxide_constructs::ConstructRegistry;

let mut registry = ConstructRegistry::new();
registry.register(construct);

for c in registry.list_deployed() {
    println!("GPU resident: {}", c.manifest.name);
}
```

---

## Applications

### Live Kernel Hotswap

In production inference clusters, rolling out a new fused attention kernel traditionally requires draining the node, rebooting the service, and warming up caches. With `oxide-constructs`, a new kernel version can be pushed to a git tag, pulled by the registry, compiled, and atomically swapped into the GPU without dropping in-flight requests. The old kernel stays `Cached` for instant rollback.

### Dynamic Capability Negotiation

Imagine a heterogeneous fleet: some nodes have A100s, others H100s, a few experimental ones with Blackwell. An agent submits a workload that requires tensor cores and unified memory. The runtime queries the local `ConstructRegistry` for `Equipment` constructs, matches the agent's requirements against available hardware, and selects the optimal `Skill` implementation for that node class. No centralized scheduler required—just local registry search.

### Fleet-Wide Propagation via SmartCRDT

In edge deployments, nodes may be disconnected from the internet for hours or days. Each node maintains its own `ConstructRegistry`. When connectivity is restored, registries are merged. Newer kernels win. Malformed or untrusted constructs (those failing identity verification) are rejected. The fleet converges on a consistent, verified set of capabilities using the same CRDT mathematics that power collaborative document editing.

### Supply-Chain Security

Every construct can carry a `ConstructIdentity` containing a DID, a creator fingerprint, and a manifest signature. The loader can optionally enforce identity verification before compilation or deployment. Because the manifest lives in git, audit trails are permanent and tamper-evident. You know exactly who wrote every kernel running on your GPUs.

---

## Architecture & Design Philosophy

### Separation of Concerns

`oxide-constructs` deliberately does not compile CUDA source or generate PTX itself. It is the **orchestrator**, not the **compiler**. Compilation is delegated to the `flux-importer` → `cuda-oxide` pipeline. This separation means:

- The loader can be used with any backend that speaks PTX (or eventually SPIR-V, ROCm, or oneAPI).
- The compiler can evolve independently—new LLVM passes, new ternary lowering strategies, new target architectures—without touching the loading logic.
- Security boundaries are crisp: the loader handles trust and identity; the compiler handles code generation.

### Zero-Copy Metrics

Every `Construct` tracks its own runtime metrics: invocations, total execution time, peak memory usage, error count, and last invocation timestamp. These metrics are cheap to maintain (a handful of atomic increments) and enable data-driven decisions about which kernels to keep resident, which to evict, and which to flag for performance regression.

### Error Transparency

Every failure mode is an explicit, structured variant:

- `GitCloneFailed` — the repository is unreachable or corrupt.
- `ManifestParseFailed` — the `CONSTRUCT.toml` is malformed.
- `ValidationFailed` — required fields are missing.
- `IdentityVerificationFailed` — the DID signature does not match.
- `DependencyNotFound` — a required sub-construct is missing from the registry.
- `CompilationFailed` — the `cuda-oxide` pipeline rejected the source.
- `DeploymentFailed` — the CUDA driver refused the PTX blob.
- `InvalidState` — the state machine was violated.

No opaque error codes. No stringly-typed diagnostics. Every failure is actionable.

---

## Relationship to the SuperInstance Ecosystem

This crate is a foundational building block of [SuperInstance](https://github.com/SuperInstance/SuperInstance), an agent-native GPU runtime built on three radical premises:

1. **Ternary computation** (`{-1, 0, +1}`) as a first-class numeric type, enabling branchless neural networks and 1.58-bit-per-value memory bandwidth.
2. **Flux**, a surface language that compiles to PTX via a ternary-aware SSA intermediate representation.
3. **`open-parallel`**, an async runtime that treats GPU kernels as awaitable tasks with structured concurrency and cancellation.

`oxide-constructs` sits at the boundary between the distributed world (git, CRDTs, identity) and the GPU world (PTX, CUDA, memory). It is the bridge that lets agents pull capabilities out of the ether and run them on silicon.

Other crates in the ecosystem include:

- `cuda-oxide` — The Rust→CUDA/PTX compiler frontend (forked and extended from NVlabs research).
- `open-parallel` — Async GPU task runtime with coroutine-based kernel dispatch.
- `flux-lexer`, `flux-parser`, `flux-syntax` — The Flux language toolchain.
- `agent-identity` — DID-based identity and signing for autonomous agents.

---

## Contributing

We welcome contributions that improve the loader, expand the registry query surface, or add new identity backends. Please see the main [SuperInstance repository](https://github.com/SuperInstance/SuperInstance) for contributor guidelines, RFC processes, and community discussion.

All code in this repository is licensed under the Apache License 2.0.

---

## Acknowledgments

The construct model was inspired by the intersection of git-native package management (Nix, Guix), capability-based security (Capsicum, Fuchsia), and the ternary computation research pioneered by the SuperInstance team. The CRDT registry merge strategy owes a debt to Martin Kleppmann's *Designing Data-Intensive Applications* and the Riak DT library.
