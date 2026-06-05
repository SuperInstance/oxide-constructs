# oxide-constructs

> **Git-native construct loader for the Flux→PTX distributed GPU runtime.**

```
    Discovered → Validated → Resolved → Compiled → Deployed → Cached
         ↑                                                    │
         └──────────────── unload ────────────────────────────┘
```

In distributed GPU systems, capability is not a static property of the hardware—it is a *negotiable asset* that flows through the fleet. A node may discover that it needs a ternary attention kernel it has never seen before. Another node may advertise spare tensor-core capacity as a fungible resource. The question is not whether these capabilities exist, but how they are **named**, **verified**, **resolved**, **compiled**, and **deployed** at runtime.

**oxide-constructs** answers that question. It treats GPU capability as a git-native artifact—a *construct*—that can be loaded from any repository, validated against a declarative manifest, compiled to PTX, and hot-swapped into a running process without restarting the runtime.

---

## Table of Contents

1. [What Is a Construct?](#what-is-a-construct)
2. [The Construct Lifecycle](#the-construct-lifecycle)
3. [Construct Manifests](#construct-manifests)
4. [The Construct Registry](#the-construct-registry)
5. [Discovery by Tags](#discovery-by-tags)
6. [Identity Verification](#identity-verification)
7. [API Examples](#api-examples)
8. [Relationship to the Agent Stack](#relationship-to-the-agent-stack)
9. [Installation](#installation)
10. [License](#license)

---

## What Is a Construct?

A **construct** is a self-contained unit of GPU capability that lives in a git repository. It is the atomic currency of the Flux→PTX runtime: if you can express a capability as a construct, any node in the fleet can discover it, verify it, and deploy it.

Constructs come in three fundamental kinds:

| Kind | Description | Example |
|------|-------------|---------|
| **Skill** | A software capability—kernels, shaders, compute graphs. | A fused attention-forward kernel with entry point `attention_main`. |
| **Equipment** | A hardware requirement or advertisement—SM version, VRAM, tensor cores. | A declaration that a node offers SM 8.0+, 24 GB VRAM, and tensor cores. |
| **Hybrid** | A skill bundled with its minimum equipment requirements. | A kernel that *includes* its own SM and VRAM floor, so the scheduler knows immediately whether the node can run it. |

The distinction matters because the fleet negotiates capability at two levels simultaneously: *what can this node do* (skills) and *what does this node need* (equipment). Hybrids collapse that negotiation into a single atomic unit, which simplifies scheduling and reduces round-trips.

---

## The Construct Lifecycle

Every construct moves through a strict state machine. State transitions are enforced by the loader; you cannot deploy a construct that has not been compiled, and you cannot compile a construct whose dependencies have not been resolved.

```
┌─────────────┐    load_from_repo    ┌─────────────┐
│  Discovered │ ───────────────────> │  Validated  │
│  (initial)  │                      │ (manifest   │
│             │                      │  parsed)    │
└─────────────┘                      └─────────────┘
                                            │
                                            │ resolve deps
                                            ▼
                                     ┌─────────────┐
                                     │  Resolved   │
                                     │ (deps ok)   │
                                     └─────────────┘
                                            │
                                            │ compile
                                            ▼
                                     ┌─────────────┐     deploy      ┌─────────────┐
                                     │  Compiled   │ ──────────────> │  Deployed   │
                                     │ (PTX ready) │                 │ (on GPU)    │
                                     └─────────────┘                 └─────────────┘
                                                                              │
                                                                              │ unload
                                                                              ▼
                                                                       ┌─────────────┐
                                                                       │   Cached    │
                                                                       │ (PTX kept,  │
                                                                       │  GPU freed) │
                                                                       └─────────────┘
```

### State Definitions

| State | Meaning |
|-------|---------|
| `Discovered` | The construct has been located—perhaps via tag search or fleet gossip—but no manifest has been read yet. |
| `Validated` | The `CONSTRUCT.toml` manifest has been parsed and passed structural validation: name is non-empty, at least one compute capability is declared, and the semantic version is well-formed. |
| `Resolved` | All declared dependencies have been located in the registry or fetched from their own git repositories. Dependency resolution is recursive and version-aware. |
| `Compiled` | The construct has been lowered to PTX (or another target binary) and the result is cached in memory. For skills, this means the kernel is ready to launch. For equipment, compilation is a no-op but the state transition still signifies readiness. |
| `Deployed` | The compiled artifact has been uploaded to the GPU and is actively resident in device memory. The construct may now be invoked. |
| `Cached` | The construct has been unloaded from the GPU, but its compiled binary is retained in host memory so that redeployment is cheap. |
| `Failed` | A terminal error state. The string payload records the reason so that operators and retry policies can inspect it. |

---

## Construct Manifests

Every construct repository contains a `CONSTRUCT.toml` at its root. This file is the *single source of truth* for everything the loader needs to know about the construct: what it provides, what it needs, who wrote it, and how to verify it.

```toml
# CONSTRUCT.toml — example hybrid construct
name = "ternary-attention-kernel"
version = "2.0.0"
description = "Fused ternary attention with KV-cache compression"
tags = ["attention", "transformer", "gpu", "kv-cache"]

[construct]
type = "hybrid"          # "skill" | "equipment" | "hybrid"
provides = "attention-forward"
entry_point = "ternary_attention_main"

[equipment]
min_sm_version = 80
min_vram_mb = 4096
requires_tensor_cores = true
requires_unified_memory = false

[dependencies]
"SuperInstance/reduce-ops" = { version = "1.3.0", symbol = "warp_reduce" }
"SuperInstance/flash-attn-base" = { version = "0.9.0", symbol = "softmax_scale" }

[identity]
did = "did:flux:abc123..."
creator_fingerprint = "SHA256:deadbeef..."
signature = "base64(sig(manifest))"

[compute]
capabilities = [80, 86, 89, 90]
```

### Key Sections

- **`name`** and **`version`** form the unique identity of the construct within the fleet. Versions follow strict semantic versioning (see `SemVer`).
- **`construct`** declares the type. A `skill` provides software; `equipment` advertises hardware; `hybrid` does both.
- **`dependencies`** lists other constructs by repository path. The loader resolves them recursively, using semantic-version compatibility rules (same major, equal or greater minor).
- **`identity`** binds the construct to a Decentralized Identifier (DID). See [Identity Verification](#identity-verification).
- **`tags`** enable discovery via the flux-index. See [Discovery by Tags](#discovery-by-tags).
- **`compute.capabilities`** lists the CUDA compute capabilities the construct supports. The loader rejects manifests that declare none, because a construct with no declared target is a construct that cannot be scheduled.

---

## The Construct Registry

The `ConstructRegistry` is the in-memory catalog of every construct known to the local node. It is not merely a hash map; it is a **CRDT-style replicated store** designed for fleet-wide synchronization.

### Version-Based Merge

When a node receives a registry delta from a peer, it merges the two registries with a *last-write-wins* policy keyed by semantic version:

```rust
// On node A
let mut local = ConstructRegistry::new();
local.register(my_construct);

// Receive registry from node B over the fleet mesh
local.merge(&remote_registry);

// If both registries contain "kernel-a", the higher SemVer wins.
```

The merge rule is simple but sufficient for distributed convergence:

1. If the local registry has no entry for a construct, add it.
2. If both registries have the same construct, keep the one with the **greater** semantic version.
3. If the versions are equal, keep the local copy (ties are deterministic).

This is a state-based CRDT: merge is associative, commutative, and idempotent. Any two nodes that exchange registries will converge to the same set of constructs without coordination.

### Registry Operations

```rust
use oxide_constructs::{ConstructRegistry, ConstructState};

let mut registry = ConstructRegistry::new();

// Register a construct
registry.register(construct);

// Query by name
if let Some(c) = registry.get("ternary-attention-kernel") {
    println!("Found: {}", c.manifest.name);
}

// List everything currently on the GPU
let active: Vec<_> = registry.list_deployed();

// Remove a construct
registry.unregister("old-kernel");
```

---

## Discovery by Tags

Constructs advertise themselves through tags. A tag is an arbitrary string—conventionally lowercase, hyphenated—used by the flux-index to route discovery queries.

```rust
let attention_kernels = registry.search_by_tags(&[
    "attention".to_string(),
    "gpu".to_string(),
]);
```

A tag search returns every construct whose manifest contains **at least one** of the requested tags. This is intentionally permissive: a scheduler looking for `"attention"` should not miss a hybrid construct that is also tagged `"kv-cache"`.

Tag-based discovery enables *capability negotiation* without prior knowledge. A node that needs an attention kernel does not need to know the exact repository name; it asks the registry for `"attention"` and receives every candidate that matches.

---

## Identity Verification

Constructs are executable code that will run on your GPU. Before compilation or deployment, the loader can verify that a construct was published by a trusted party.

Identity is anchored by **DIDs** (Decentralized Identifiers) from the `agent-identity` system:

```rust
pub struct ConstructIdentity {
    /// DID from agent-identity (e.g., did:flux:abc123...)
    pub did: String,
    /// Public-key fingerprint of the creator.
    pub creator_fingerprint: String,
    /// Signature of the manifest content.
    pub signature: Option<String>,
}
```

When `verify_identity` is enabled (the default), the loader:

1. Retrieves the DID document from the agent-identity resolver.
2. Checks that the manifest's `creator_fingerprint` matches a key listed in the DID document.
3. Verifies the signature over the canonicalized manifest bytes.

If any step fails, the construct transitions to `Failed(IdentityVerificationFailed)` and will not be compiled or deployed. This prevents supply-chain attacks: a malicious construct injected into a public repository cannot pass identity checks unless the attacker also controls the associated DID.

---

## API Examples

### Loading a Construct from a Git Repository

```rust
use oxide_constructs::ConstructLoader;

let loader = ConstructLoader::new();
let mut construct = loader
    .load_from_repo("SuperInstance/ternary-attention-kernel", "v2.0.0")?;

assert_eq!(construct.state, ConstructState::Validated);
println!("Loaded: {} @ {}", construct.manifest.name, construct.manifest.version);
```

The loader clones the repository (or uses a local cache), parses `CONSTRUCT.toml`, and validates the manifest. The construct is now `Validated` but not yet ready for the GPU.

### Resolving Dependencies

```rust
// In practice, resolution happens automatically during load or compile.
// If you need manual control:
loader.resolve_dependencies(&mut construct)?;
assert_eq!(construct.state, ConstructState::Resolved);
```

### Compiling to PTX

```rust
loader.compile(&mut construct)?;
assert_eq!(construct.state, ConstructState::Compiled);

// The compiled binary is now cached in construct.ptx_cache
if let Some(ref ptx) = construct.ptx_cache {
    println!("PTX size: {} bytes", ptx.len());
}
```

In production, `compile` invokes the `flux-importer` → `cuda-oxide` pipeline. In test environments, the binary may be a placeholder; the state machine is still exercised.

### Deploying to the GPU

```rust
loader.deploy(&mut construct)?;
assert_eq!(construct.state, ConstructState::Deployed);

// The kernel is now resident in device memory and may be launched.
```

### Unloading and Caching

```rust
loader.unload(&mut construct)?;
assert_eq!(construct.state, ConstructState::Cached);

// GPU memory is freed, but the PTX cache is retained.
// Redeployment is a single loader.deploy() call away.
```

### Fleet-Wide Registry Merge

```rust
use oxide_constructs::ConstructRegistry;

// Node A's registry
let mut local = ConstructRegistry::new();
local.register(node_a_construct);

// Merge in Node B's registry (received over the mesh)
local.merge(&node_b_registry);

// Fleet now converges on the latest version of every construct.
```

### Metrics Inspection

Every construct carries runtime metrics:

```rust
println!("Invocations: {}", construct.metrics.invocations);
println!("Avg time: {:.2} µs", construct.metrics.avg_time_us());
println!("Peak memory: {} bytes", construct.metrics.peak_memory_bytes);
println!("Errors: {}", construct.metrics.errors);
```

These metrics are updated by the runtime after every kernel launch and can be gossiped across the fleet for load-balancing decisions.

---

## Relationship to the Agent Stack

`oxide-constructs` is one crate in a larger agent-native stack. It does not stand alone; it collaborates with three sibling systems:

| System | Role | How Constructs Interact |
|--------|------|------------------------|
| **git-agent** | Git-native agent runtime that executes tasks inside repositories. | Constructs *are* git repositories. The git-agent checks out the repository, reads `CONSTRUCT.toml`, and hands the manifest to the loader. |
| **agent-manifest** | Declarative manifest system for agent capabilities and requirements. | `CONSTRUCT.toml` is a specialization of the agent-manifest schema. The manifest crate provides the TOML parser and validation rules; oxide-constructs adds GPU-specific semantics (compute capabilities, PTX caching, deployment states). |
| **agent-identity** | DID-based identity and signature verification for agents and artifacts. | Every construct can carry an `identity` block. The loader delegates signature verification to the agent-identity resolver, ensuring that constructs are cryptographically attributable to their publishers. |

In the full stack, the flow looks like this:

1. A **git-agent** discovers a new repository tagged as a construct.
2. The **agent-manifest** parser validates the `CONSTRUCT.toml` syntax.
3. **oxide-constructs** loads the construct, resolves dependencies, compiles it, and manages its lifecycle.
4. Before deployment, **agent-identity** verifies the publisher's DID and signature.
5. The construct is now a first-class GPU capability, discoverable by any node via tag search and synchronizable via CRDT merge.

---

## Installation

Add `oxide-constructs` to your `Cargo.toml`:

```toml
[dependencies]
oxide-constructs = { git = "https://github.com/SuperInstance/oxide-constructs" }
```

The crate is pure Rust (standard library only) and builds on stable Rust 1.70+.

---

## License

Apache-2.0

---

*Constructs are not binaries you install. They are capabilities you negotiate. The GPU fleet is not a data center; it is a market of skills and equipment, resolved at runtime, verified by identity, and synchronized without a central ledger. oxide-constructs is the protocol that makes that market possible.*
