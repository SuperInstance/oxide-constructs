//! # oxide-constructs
//!
//! Git-native construct loader for the Flux→PTX distributed GPU runtime.
//!
//! A **construct** is a self-contained unit of GPU capability — a kernel, a skill,
//! a piece of equipment — that lives in a git repository. Constructs can be loaded
//! and unloaded at runtime, enabling live kernel hotswap, dynamic capability
//! negotiation, and fleet-wide capability propagation via SmartCRDT.
//!
//! ## Two Types of Constructs
//!
//! - **Skills**: Software capabilities (kernels, shaders, compute graphs)
//! - **Equipment**: Hardware requirements (SM version, VRAM, tensor cores)
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use oxide_constructs::{ConstructLoader, ConstructType};
//!
//! let loader = ConstructLoader::new();
//! let construct = loader.load_from_repo("SuperInstance/ternary-attention-kernel", "v2.0.0")?;
//! println!("Loaded: {} ({:?})", construct.manifest.name, construct.manifest.construct_type);
//! ```

use std::collections::HashMap;

/// Unique identifier for a construct.
pub type ConstructId = String;

/// Semantic version for a construct.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct SemVer {
    pub major: u32,
    pub minor: u32,
    pub patch: u32,
}

impl SemVer {
    pub fn new(major: u32, minor: u32, patch: u32) -> Self {
        Self { major, minor, patch }
    }

    pub fn parse(s: &str) -> Option<Self> {
        let s = s.trim_start_matches('v');
        let parts: Vec<&str> = s.split('.').collect();
        if parts.len() != 3 { return None; }
        Some(Self {
            major: parts[0].parse().ok()?,
            minor: parts[1].parse().ok()?,
            patch: parts[2].parse().ok()?,
        })
    }

    pub fn is_compatible_with(&self, required: &SemVer) -> bool {
        self.major == required.major && self.minor <= required.minor
    }
}

impl std::fmt::Display for SemVer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

/// The type of construct.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConstructType {
    /// A software skill: kernel, shader, compute graph.
    Skill {
        /// What this skill provides (e.g., "attention-forward", "reduce-sum").
        provides: String,
        /// The GPU kernel entry point.
        entry_point: String,
        /// Supported compute capabilities.
        min_compute_capability: u32,
    },
    /// A piece of equipment: hardware requirements.
    Equipment {
        /// Minimum SM version (e.g., 70, 80, 90).
        min_sm_version: u32,
        /// Minimum VRAM in MB.
        min_vram_mb: u32,
        /// Whether tensor cores are required.
        requires_tensor_cores: bool,
        /// Whether unified memory is required.
        requires_unified_memory: bool,
    },
    /// A hybrid: skill + equipment requirements bundled together.
    Hybrid {
        skill: String,
        entry_point: String,
        min_sm_version: u32,
        min_vram_mb: u32,
    },
}

/// Identity information for a construct.
#[derive(Debug, Clone)]
pub struct ConstructIdentity {
    /// Decentralized Identifier (DID) from agent-identity.
    pub did: String,
    /// The public key fingerprint of the creator.
    pub creator_fingerprint: String,
    /// Signature of the manifest content.
    pub signature: Option<String>,
}

/// A construct manifest — the declaration file in a construct repo.
#[derive(Debug, Clone)]
pub struct ConstructManifest {
    /// Unique name of this construct.
    pub name: String,
    /// Semantic version.
    pub version: SemVer,
    /// Type of construct (skill, equipment, hybrid).
    pub construct_type: ConstructType,
    /// Dependencies on other constructs.
    pub dependencies: Vec<ConstructDependency>,
    /// Identity information.
    pub identity: Option<ConstructIdentity>,
    /// Description for humans and agents.
    pub description: String,
    /// Tags for discovery via flux-index.
    pub tags: Vec<String>,
    /// Supported CUDA compute capabilities.
    pub compute_capabilities: Vec<u32>,
}

/// A dependency on another construct.
#[derive(Debug, Clone)]
pub struct ConstructDependency {
    pub repo: String,
    pub version: SemVer,
    pub symbol: String,
}

/// A loaded construct with resolved state.
#[derive(Debug, Clone)]
pub struct Construct {
    /// The manifest.
    pub manifest: ConstructManifest,
    /// Git repository URL.
    pub repo_url: String,
    /// Git reference (tag, branch, or commit hash).
    pub git_ref: String,
    /// Load state.
    pub state: ConstructState,
    /// Cached kernel binary (PTX), if compiled.
    pub ptx_cache: Option<Vec<u8>>,
    /// Metrics from this construct's execution.
    pub metrics: ConstructMetrics,
}

/// The load state of a construct.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConstructState {
    /// Discovered but not yet loaded.
    Discovered,
    /// Manifest parsed and validated.
    Validated,
    /// Dependencies resolved.
    Resolved,
    /// Compiled to PTX and ready for deployment.
    Compiled,
    /// Deployed to GPU and running.
    Deployed,
    /// Unloaded from GPU but still cached.
    Cached,
    /// Failed to load.
    Failed(String),
}

/// Runtime metrics for a construct.
#[derive(Debug, Clone, Default)]
pub struct ConstructMetrics {
    /// Number of times this construct has been invoked.
    pub invocations: u64,
    /// Total execution time in microseconds.
    pub total_time_us: u64,
    /// Peak GPU memory usage in bytes.
    pub peak_memory_bytes: u64,
    /// Number of errors.
    pub errors: u64,
    /// Last invocation timestamp (epoch millis).
    pub last_invocation_ms: u64,
}

impl ConstructMetrics {
    pub fn avg_time_us(&self) -> f64 {
        if self.invocations == 0 { 0.0 } else { self.total_time_us as f64 / self.invocations as f64 }
    }
}

/// The construct registry — tracks all known constructs across the fleet.
#[derive(Debug, Clone, Default)]
pub struct ConstructRegistry {
    constructs: HashMap<ConstructId, Construct>,
}

impl ConstructRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a new construct.
    pub fn register(&mut self, construct: Construct) {
        let id = construct.manifest.name.clone();
        self.constructs.insert(id, construct);
    }

    /// Unregister a construct by name.
    pub fn unregister(&mut self, name: &str) -> Option<Construct> {
        self.constructs.remove(name)
    }

    /// Get a construct by name.
    pub fn get(&self, name: &str) -> Option<&Construct> {
        self.constructs.get(name)
    }

    /// Get a mutable reference to a construct.
    pub fn get_mut(&mut self, name: &str) -> Option<&mut Construct> {
        self.constructs.get_mut(name)
    }

    /// List all constructs of a given type.
    pub fn list_by_type(&self, filter: &ConstructType) -> Vec<&Construct> {
        self.constructs.values().filter(|c| {
            match (&c.manifest.construct_type, filter) {
                (ConstructType::Skill { .. }, ConstructType::Skill { .. }) => true,
                (ConstructType::Equipment { .. }, ConstructType::Equipment { .. }) => true,
                (ConstructType::Hybrid { .. }, ConstructType::Hybrid { .. }) => true,
                _ => false,
            }
        }).collect()
    }

    /// List all deployed constructs.
    pub fn list_deployed(&self) -> Vec<&Construct> {
        self.constructs.values()
            .filter(|c| c.state == ConstructState::Deployed)
            .collect()
    }

    /// List constructs matching tags (for flux-index discovery).
    pub fn search_by_tags(&self, tags: &[String]) -> Vec<&Construct> {
        self.constructs.values()
            .filter(|c| c.manifest.tags.iter().any(|t| tags.contains(t)))
            .collect()
    }

    /// Total number of constructs.
    pub fn len(&self) -> usize {
        self.constructs.len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.constructs.is_empty()
    }

    /// Merge another registry into this one (CRDT-style, last-write-wins by version).
    pub fn merge(&mut self, other: &ConstructRegistry) {
        for (id, construct) in &other.constructs {
            match self.constructs.get(id) {
                Some(existing) => {
                    if construct.manifest.version > existing.manifest.version {
                        self.constructs.insert(id.clone(), construct.clone());
                    }
                }
                None => {
                    self.constructs.insert(id.clone(), construct.clone());
                }
            }
        }
    }
}

/// The construct loader — loads constructs from git repositories.
pub struct ConstructLoader {
    /// Cache directory for cloned repos.
    cache_dir: String,
    /// Whether to verify identities.
    verify_identity: bool,
}

impl ConstructLoader {
    pub fn new() -> Self {
        Self {
            cache_dir: "/tmp/oxide-constructs".to_string(),
            verify_identity: true,
        }
    }

    /// Load a construct from a git repository.
    pub fn load_from_repo(&self, repo: &str, git_ref: &str) -> Result<Construct, ConstructError> {
        let manifest = self.parse_manifest(repo, git_ref)?;
        self.validate_manifest(&manifest)?;

        Ok(Construct {
            manifest,
            repo_url: format!("https://github.com/{}.git", repo),
            git_ref: git_ref.to_string(),
            state: ConstructState::Validated,
            ptx_cache: None,
            metrics: ConstructMetrics::default(),
        })
    }

    fn parse_manifest(&self, repo: &str, git_ref: &str) -> Result<ConstructManifest, ConstructError> {
        // In production: git clone + parse CONSTRUCT.toml
        // For now: construct a manifest from the repo name
        let name = repo.split('/').last().unwrap_or("unknown").to_string();
        let version = SemVer::parse(git_ref).unwrap_or(SemVer::new(0, 1, 0));

        Ok(ConstructManifest {
            name,
            version,
            construct_type: ConstructType::Hybrid {
                skill: repo.to_string(),
                entry_point: "kernel_main".to_string(),
                min_sm_version: 80,
                min_vram_mb: 1024,
            },
            dependencies: Vec::new(),
            identity: None,
            description: format!("Construct from {}", repo),
            tags: vec!["gpu".to_string(), "kernel".to_string()],
            compute_capabilities: vec![80, 86, 89, 90],
        })
    }

    fn validate_manifest(&self, manifest: &ConstructManifest) -> Result<(), ConstructError> {
        if manifest.name.is_empty() {
            return Err(ConstructError::ValidationFailed("name is empty".to_string()));
        }
        if manifest.compute_capabilities.is_empty() {
            return Err(ConstructError::ValidationFailed("no compute capabilities declared".to_string()));
        }
        Ok(())
    }

    /// Compile a validated construct to PTX.
    pub fn compile(&self, construct: &mut Construct) -> Result<(), ConstructError> {
        if construct.state != ConstructState::Validated && construct.state != ConstructState::Resolved {
            return Err(ConstructError::InvalidState {
                expected: "Validated or Resolved".to_string(),
                actual: format!("{:?}", construct.state),
            });
        }
        // In production: invoke flux-importer → cuda-oxide pipeline
        construct.ptx_cache = Some(vec![0x00; 64]); // Placeholder
        construct.state = ConstructState::Compiled;
        Ok(())
    }

    /// Deploy a compiled construct to the GPU.
    pub fn deploy(&self, construct: &mut Construct) -> Result<(), ConstructError> {
        if construct.state != ConstructState::Compiled {
            return Err(ConstructError::InvalidState {
                expected: "Compiled".to_string(),
                actual: format!("{:?}", construct.state),
            });
        }
        if construct.ptx_cache.is_none() {
            return Err(ConstructError::ValidationFailed("no PTX cache".to_string()));
        }
        construct.state = ConstructState::Deployed;
        Ok(())
    }

    /// Unload a deployed construct from the GPU.
    pub fn unload(&self, construct: &mut Construct) -> Result<(), ConstructError> {
        if construct.state != ConstructState::Deployed {
            return Err(ConstructError::InvalidState {
                expected: "Deployed".to_string(),
                actual: format!("{:?}", construct.state),
            });
        }
        construct.state = ConstructState::Cached;
        Ok(())
    }
}

impl Default for ConstructLoader {
    fn default() -> Self {
        Self::new()
    }
}

/// Errors during construct operations.
#[derive(Debug, Clone)]
pub enum ConstructError {
    GitCloneFailed { repo: String, reason: String },
    ManifestParseFailed { repo: String, reason: String },
    ValidationFailed(String),
    IdentityVerificationFailed { did: String, reason: String },
    DependencyNotFound { repo: String, version: String },
    CompilationFailed { reason: String },
    DeploymentFailed { reason: String },
    InvalidState { expected: String, actual: String },
}

impl std::fmt::Display for ConstructError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::GitCloneFailed { repo, reason } => write!(f, "git clone failed for {}: {}", repo, reason),
            Self::ManifestParseFailed { repo, reason } => write!(f, "manifest parse failed for {}: {}", repo, reason),
            Self::ValidationFailed(msg) => write!(f, "validation failed: {}", msg),
            Self::IdentityVerificationFailed { did, reason } => write!(f, "identity verification failed for {}: {}", did, reason),
            Self::DependencyNotFound { repo, version } => write!(f, "dependency not found: {} @ {}", repo, version),
            Self::CompilationFailed { reason } => write!(f, "compilation failed: {}", reason),
            Self::DeploymentFailed { reason } => write!(f, "deployment failed: {}", reason),
            Self::InvalidState { expected, actual } => write!(f, "invalid state: expected {}, got {}", expected, actual),
        }
    }
}

impl std::error::Error for ConstructError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_semver_parsing() {
        let v = SemVer::parse("v2.1.3").unwrap();
        assert_eq!(v, SemVer::new(2, 1, 3));
        let v2 = SemVer::parse("1.0.0").unwrap();
        assert_eq!(v2, SemVer::new(1, 0, 0));
        assert!(SemVer::parse("invalid").is_none());
    }

    #[test]
    fn test_semver_compatibility() {
        let v1 = SemVer::new(2, 1, 0);
        let v2 = SemVer::new(2, 2, 0);
        let v3 = SemVer::new(3, 0, 0);
        assert!(v1.is_compatible_with(&v2));
        assert!(!v1.is_compatible_with(&v3));
    }

    #[test]
    fn test_construct_lifecycle() {
        let loader = ConstructLoader::new();
        let mut construct = loader.load_from_repo("SuperInstance/test-kernel", "v1.0.0").unwrap();
        assert_eq!(construct.state, ConstructState::Validated);

        loader.compile(&mut construct).unwrap();
        assert_eq!(construct.state, ConstructState::Compiled);

        loader.deploy(&mut construct).unwrap();
        assert_eq!(construct.state, ConstructState::Deployed);

        loader.unload(&mut construct).unwrap();
        assert_eq!(construct.state, ConstructState::Cached);
    }

    #[test]
    fn test_registry() {
        let mut registry = ConstructRegistry::new();
        assert!(registry.is_empty());

        let construct = Construct {
            manifest: ConstructManifest {
                name: "test-construct".to_string(),
                version: SemVer::new(1, 0, 0),
                construct_type: ConstructType::Skill {
                    provides: "attention".to_string(),
                    entry_point: "kernel_main".to_string(),
                    min_compute_capability: 80,
                },
                dependencies: vec![],
                identity: None,
                description: "test".to_string(),
                tags: vec!["gpu".to_string()],
                compute_capabilities: vec![80],
            },
            repo_url: "https://github.com/test.git".to_string(),
            git_ref: "v1.0.0".to_string(),
            state: ConstructState::Deployed,
            ptx_cache: None,
            metrics: ConstructMetrics::default(),
        };

        registry.register(construct);
        assert_eq!(registry.len(), 1);
        assert!(registry.get("test-construct").is_some());
        assert_eq!(registry.list_deployed().len(), 1);
    }

    #[test]
    fn test_registry_merge() {
        let mut r1 = ConstructRegistry::new();
        let mut r2 = ConstructRegistry::new();

        let c1 = Construct {
            manifest: ConstructManifest {
                name: "kernel-a".to_string(),
                version: SemVer::new(1, 0, 0),
                construct_type: ConstructType::Skill {
                    provides: "reduce".to_string(),
                    entry_point: "reduce_main".to_string(),
                    min_compute_capability: 70,
                },
                dependencies: vec![],
                identity: None,
                description: "v1".to_string(),
                tags: vec![],
                compute_capabilities: vec![70],
            },
            repo_url: String::new(),
            git_ref: "v1.0.0".to_string(),
            state: ConstructState::Deployed,
            ptx_cache: None,
            metrics: ConstructMetrics::default(),
        };

        let mut c2 = c1.clone();
        c2.manifest.version = SemVer::new(2, 0, 0);
        c2.manifest.description = "v2".to_string();

        r1.register(c1);
        r2.register(c2);

        r1.merge(&r2);
        assert_eq!(r1.get("kernel-a").unwrap().manifest.version, SemVer::new(2, 0, 0));
    }

    #[test]
    fn test_tag_search() {
        let mut registry = ConstructRegistry::new();
        let mut construct = Construct {
            manifest: ConstructManifest {
                name: "searchable".to_string(),
                version: SemVer::new(1, 0, 0),
                construct_type: ConstructType::Equipment {
                    min_sm_version: 80,
                    min_vram_mb: 4096,
                    requires_tensor_cores: true,
                    requires_unified_memory: false,
                },
                dependencies: vec![],
                identity: None,
                description: "test".to_string(),
                tags: vec!["attention".to_string(), "gpu".to_string()],
                compute_capabilities: vec![80],
            },
            repo_url: String::new(),
            git_ref: "v1.0.0".to_string(),
            state: ConstructState::Discovered,
            ptx_cache: None,
            metrics: ConstructMetrics::default(),
        };
        registry.register(construct);

        let results = registry.search_by_tags(&["attention".to_string()]);
        assert_eq!(results.len(), 1);

        let no_results = registry.search_by_tags(&["nonexistent".to_string()]);
        assert!(no_results.is_empty());
    }

    #[test]
    fn test_metrics() {
        let metrics = ConstructMetrics {
            invocations: 100,
            total_time_us: 5000,
            peak_memory_bytes: 1024,
            errors: 2,
            last_invocation_ms: 1000,
        };
        assert!((metrics.avg_time_us() - 50.0).abs() < 0.001);
    }

    #[test]
    fn test_invalid_state_transitions() {
        let loader = ConstructLoader::new();
        let mut construct = Construct {
            manifest: ConstructManifest {
                name: "test".to_string(),
                version: SemVer::new(1, 0, 0),
                construct_type: ConstructType::Skill {
                    provides: "test".to_string(),
                    entry_point: "main".to_string(),
                    min_compute_capability: 80,
                },
                dependencies: vec![],
                identity: None,
                description: String::new(),
                tags: vec![],
                compute_capabilities: vec![80],
            },
            repo_url: String::new(),
            git_ref: "v1.0.0".to_string(),
            state: ConstructState::Discovered,
            ptx_cache: None,
            metrics: ConstructMetrics::default(),
        };

        // Can't compile from Discovered
        assert!(loader.compile(&mut construct).is_err());
        // Can't deploy from Discovered
        assert!(loader.deploy(&mut construct).is_err());
        // Can't unload from Discovered
        assert!(loader.unload(&mut construct).is_err());
    }
}
