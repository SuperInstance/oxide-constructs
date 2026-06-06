# oxide-constructs

> Git-native construct loader for dynamic GPU capability deployment across the SuperInstance fleet.

## Background Theory

In traditional GPU computing, a kernel is compiled, linked, and deployed as a static artifact. The host program knows the kernel's name, signature, and file path at build time. This model breaks down when the authors of the kernel are not the authors of the host program — for example, when an agent writes a kernel in response to a user request, or when a fleet of heterogeneous GPUs must dynamically discover what each node can execute.

`oxide-constructs` introduces the **construct**: a self-contained unit of GPU capability that lives in a git repository, carries a semantic version, declares its own dependencies, and can be loaded, compiled, deployed, and unloaded at runtime. Constructs blur the line between software package and hardware requirement. A construct can be:

- A **Skill**: A kernel, shader, or compute graph that provides a named capability.
- An **Equipment**: A hardware requirement such as SM version, VRAM, or tensor core availability.
- A **Hybrid**: Both skill and equipment bundled together, enabling self-describing deployments.

The theoretical move here is from **static linkage** to **git-native capability negotiation**. The fleet does not need to know every kernel at compile time; it only needs to know how to discover, validate, and merge constructs from decentralized repositories.

## How It Works

### Construct Identity

Every construct is identified by:

- A `name` and `SemVer` version.
- A `ConstructType` (Skill, Equipment, Hybrid).
- A list of `ConstructDependency` entries for transitive resolution.
- Optional `ConstructIdentity` (DID, creator fingerprint, manifest signature).
- Tags for discovery.
- Supported CUDA compute capabilities.

Semantic versioning follows a pragmatic compatibility rule: versions with the same major and a less-or-equal minor are considered compatible. This favors conservative fleet upgrades while allowing patch drift.

### Lifecycle

Constructs move through a strict state machine:

```
Discovered → Validated → Resolved → Compiled → Deployed → Cached
     ↑______________________________________________|
     └──────────────── Failed
```

- **Discovered**: Known to exist but not yet inspected.
- **Validated**: Manifest parsed and sanity-checked.
- **Resolved**: Dependencies satisfied.
- **Compiled**: PTX generated and cached.
- **Deployed**: Loaded onto GPU and ready for invocation.
- **Cached**: Unloaded from GPU but kept locally for fast redeployment.
- **Failed**: Terminal error state with reason.

The `ConstructLoader` orchestrates transitions. It can load from a git repo, compile a validated construct to PTX, deploy it to GPU, and unload it safely.

### Registry and CRDT Merge

The `ConstructRegistry` tracks all known constructs. Registries on different fleet nodes can be merged using a last-write-wins CRDT rule based on semantic version. This lets capabilities propagate organically: when a new construct is published to a git repo, it eventually reaches every node that merges registries with its neighbors.

## Experiments

The test suite encodes the following scientific claims:

```rust
#[test]
fn test_construct_lifecycle() {
    // Discovered → Validated → Compiled → Deployed → Cached transitions succeed.
}

#[test]
fn test_registry_merge() {
    // Merging registries preserves the higher semantic version.
}

#[test]
fn test_invalid_state_transitions() {
    // Illegal transitions (e.g., Discovered → Deployed) are rejected.
}
```

A larger experiment: publish 100 construct repositories, each depending on 0-5 others, and measure:

- Mean time to resolve all dependencies.
- CRDT convergence time across a 16-node ring topology.
- Version conflict rate when 10% of repos publish concurrent major versions.
- Load latency reduction from Cached vs. Compiled state.

## Applications

- **Live kernel hotswap**: Deploy a new attention kernel to a running inference fleet without restarting the host process.
- **Agentic skill markets**: Agents publish constructs to git; other agents discover and load them via tag search.
- **Hardware-aware scheduling**: `oxide-fleet` reads Equipment constructs to match work to nodes with the right GPUs.
- **Verified kernel supply chain**: Signature and DID fields enable audit trails for who compiled what kernel.
- **A/B kernel testing**: Load two versions of the same skill and let `oxide-canary` choose between them.

## Open Questions

1. **Transitive dependency closure**: How deep can dependency graphs grow before resolution becomes a fleet-wide bottleneck?
2. **Signature model**: Should constructs be signed by authors, by build infrastructure, or by both? What revocation mechanism is appropriate?
3. **Git vs. content-addressed storage**: Git is familiar but mutable. Should production constructs be pinned by content hash (IPFS, OCI) instead of git ref?
4. **Permissioned constructs**: How does a fleet node decide it is authorized to load a construct with a given DID?

## Cross-Links

- [SuperInstance agent-knowledge / AGENT-TO-AGENT-PROTOCOL.md](https://github.com/SuperInstance/agent-knowledge/blob/main/AGENT-TO-AGENT-PROTOCOL.md) — How constructs are advertised between agents.
- [SuperInstance agent-knowledge / DEPLOYMENT-AND-OPERATIONS.md](https://github.com/SuperInstance/agent-knowledge/blob/main/DEPLOYMENT-AND-OPERATIONS.md) — Fleet patterns for construct rollout.
- [SuperInstance agent-knowledge / TESTING-AS-PROOF.md](https://github.com/SuperInstance/agent-knowledge/blob/main/TESTING-AS-PROOF.md) — Why construct tests must be verifiable.
- `oxide-fleet` — Coordinates where constructs are loaded and executed.
- `oxide-sandbox` — Validates Flux constructs before they enter the live pipeline.
- `oxide-canary` — Progressively rolls out new construct versions.

## Quick Start

```rust
use oxide_constructs::{ConstructLoader, ConstructRegistry, ConstructState};

let loader = ConstructLoader::new();
let mut construct = loader.load_from_repo("SuperInstance/ternary-attention-kernel", "v2.0.0").unwrap();
assert_eq!(construct.state, ConstructState::Validated);

loader.compile(&mut construct).unwrap();
loader.deploy(&mut construct).unwrap();

let mut registry = ConstructRegistry::new();
registry.register(construct);
println!("Deployed {} construct(s)", registry.list_deployed().len());
```
