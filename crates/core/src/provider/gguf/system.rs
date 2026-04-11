//! System detection and parameter computation for llama.cpp.
//! Uses GGML backend APIs to detect CPU/GPU hardware and compute optimal params.

use llama_cpp_2::{LlamaBackendDeviceType, context::LlamaContext, list_llama_ggml_backend_devices};

/// Bytes per gigabyte constant.
const GB: f64 = 1024.0 * 1024.0 * 1024.0;

/// Information about detected hardware devices.
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    /// Device name (e.g. "Vulkan0", "CUDA0")
    pub name: String,
    /// Backend name (e.g. "Vulkan", "CUDA", "CPU")
    pub backend: String,
    /// Device type
    pub device_type: LlamaBackendDeviceType,
    /// Total memory in bytes
    pub memory_total_bytes: usize,
    /// Free/available memory in bytes
    pub memory_free_bytes: usize,
}

/// System hardware information.
#[derive(Debug, Clone)]
pub struct SystemInfo {
    /// Number of available CPU threads
    pub cpu_threads: usize,
    /// Detected devices (CPU, GPU, etc.)
    pub devices: Vec<DeviceInfo>,
}

/// Computed optimal parameters for llama.cpp.
#[derive(Debug, Clone)]
pub struct ComputeParams {
    /// Context window size
    pub n_ctx: u32,
    /// Logical batch size for prompt processing
    pub n_batch: u32,
    /// Physical batch size for prompt processing
    pub n_ubatch: u32,
    /// Number of GPU layers to offload (0 = CPU only, >0 = number of layers)
    pub n_gpu_layers: i32,
    /// Number of threads for inference
    pub n_threads: i32,
    /// Enable flash attention
    pub flash_attention: bool,
}

impl Default for ComputeParams {
    fn default() -> Self {
        Self {
            n_ctx: 4096,
            n_batch: 512,
            n_ubatch: 512,
            n_gpu_layers: 0,
            n_threads: 4,
            flash_attention: true,
        }
    }
}

/// Config overrides from user configuration.
#[derive(Debug, Default, Clone)]
pub struct ConfigOverrides {
    pub n_ctx: Option<u32>,
    pub n_batch: Option<u32>,
    pub n_gpu_layers: Option<i32>,
    pub n_threads: Option<i32>,
    pub flash_attention: Option<bool>,
    pub n_ubatch: Option<u32>,
}

/// Quantization type detected from model file type.
#[derive(Debug, Clone, Copy, PartialEq)]
#[allow(non_camel_case_types)]
pub enum Quantization {
    /// Full precision (16-bit float)
    F16,
    /// 8-bit quantization
    Q8_0,
    /// 6-bit quantization with block structure
    Q6_K,
    /// 5-bit quantization with block structure
    Q5_K,
    /// 4-bit quantization with block structure (medium)
    Q4_K,
    /// 4-bit quantization simple
    Q4_0,
    /// Unknown quantization type
    Unknown,
}

impl Quantization {
    /// Parse quantization type from file type string.
    /// File type examples: "Q4_K_M", "Q8_0", "F16", "Q6_K", "Q5_K_S"
    pub fn from_file_type(file_type: &str) -> Self {
        let file_type_upper = file_type.to_uppercase();
        if file_type_upper.contains("F16") {
            Quantization::F16
        } else if file_type_upper.contains("Q8_0") {
            Quantization::Q8_0
        } else if file_type_upper.contains("Q6_K") {
            Quantization::Q6_K
        } else if file_type_upper.contains("Q5_K") {
            Quantization::Q5_K
        } else if file_type_upper.contains("Q4_K") {
            Quantization::Q4_K
        } else if file_type_upper.contains("Q4_0") {
            Quantization::Q4_0
        } else {
            Quantization::Unknown
        }
    }

    /// Returns estimated memory per 1 billion parameters per layer in GB.
    /// These values are approximate and vary based on model architecture.
    pub fn memory_per_layer_base_gb(&self) -> f64 {
        match self {
            Quantization::F16 => 2.0,
            Quantization::Q8_0 => 1.0,
            Quantization::Q6_K => 0.7,
            Quantization::Q5_K => 0.5,
            Quantization::Q4_K => 0.35,
            Quantization::Q4_0 => 0.5,
            Quantization::Unknown => 0.5, // Conservative estimate
        }
    }
}

/// Error returned when memory was trimmed during decode (for session recovery).
/// This signals the caller to also trim message history.
#[derive(Debug)]
pub struct MemoryTrimmedError;

impl std::fmt::Display for MemoryTrimmedError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "memory was trimmed, session history needs trimming")
    }
}

impl std::error::Error for MemoryTrimmedError {}

/// Detect system hardware using GGML backend APIs.
pub fn detect_system_info() -> SystemInfo {
    let devices: Vec<DeviceInfo> = list_llama_ggml_backend_devices()
        .into_iter()
        .map(|d| DeviceInfo {
            name: d.name,
            backend: d.backend,
            device_type: d.device_type,
            memory_total_bytes: d.memory_total,
            memory_free_bytes: d.memory_free,
        })
        .collect();

    let cpu_threads = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(4);

    tracing::debug!(
        cpu_threads = cpu_threads,
        devices = devices.len(),
        "detected system info"
    );

    for device in &devices {
        tracing::debug!(
            name = %device.name,
            backend = %device.backend,
            device_type = ?device.device_type,
            memory_total_mb = device.memory_total_bytes / (1024 * 1024),
            memory_free_mb = device.memory_free_bytes / (1024 * 1024),
            "detected device"
        );
    }

    SystemInfo {
        cpu_threads,
        devices,
    }
}

/// Find the first GPU device (not CPU).
fn find_primary_gpu(devices: &[DeviceInfo]) -> Option<&DeviceInfo> {
    devices.iter().find(|d| {
        matches!(
            d.device_type,
            LlamaBackendDeviceType::Gpu | LlamaBackendDeviceType::IntegratedGpu
        )
    })
}

/// Check if GPU offload is recommended based on available VRAM.
fn should_enable_gpu(system: &SystemInfo) -> bool {
    if let Some(gpu) = find_primary_gpu(&system.devices) {
        // Enable GPU if we have at least 2GB free VRAM
        let free_gb = gpu.memory_free_bytes as f64 / GB;
        tracing::debug!(free_vram_gb = free_gb, "GPU VRAM check");
        return free_gb >= 2.0;
    }
    false
}

/// Compute the number of GPU layers to offload based on available VRAM and model size.
/// Uses quantization-aware memory estimation.
///
/// # Arguments
/// * `system` - System info with GPU device details
/// * `model_layers` - Number of layers in the model
/// * `params_billions` - Model size in billions of parameters (e.g., 4.21 for 4.21B)
/// * `quantization` - Quantization type of the model
///
/// # Returns
/// Number of layers to offload to GPU (0 for CPU-only)
pub fn compute_n_gpu_layers(
    system: &SystemInfo,
    model_layers: u32,
    params_billions: f64,
    quantization: Quantization,
) -> i32 {
    let Some(gpu) = find_primary_gpu(&system.devices) else {
        tracing::debug!("no GPU found, using CPU only");
        return 0;
    };

    let free_vram_gb = gpu.memory_free_bytes as f64 / GB;

    // Reserve 15% of VRAM for KV cache and scratch space
    let usable_vram_gb = free_vram_gb * 0.85;

    // Calculate memory per layer based on model size and quantization
    let mem_per_layer_gb = quantization.memory_per_layer_base_gb() * params_billions;

    // Calculate how many layers can fit in available VRAM
    let layers_fit = (usable_vram_gb / mem_per_layer_gb).floor() as u32;

    // Cap at model layer count
    let n_gpu_layers = layers_fit.min(model_layers) as i32;

    tracing::info!(
        free_vram_gb = free_vram_gb,
        usable_vram_gb = usable_vram_gb,
        mem_per_layer_gb = mem_per_layer_gb,
        model_layers = model_layers,
        layers_fit = layers_fit,
        n_gpu_layers = n_gpu_layers,
        quantization = ?quantization,
        "computed GPU layers"
    );

    n_gpu_layers
}

/// Compute optimal parameters based on system and model info.
pub fn compute_params(
    system: &SystemInfo,
    model_n_ctx_train: u32,
    model_layers: u32,
    params_billions: f64,
    quantization: Quantization,
    overrides: &ConfigOverrides,
) -> ComputeParams {
    // Determine if GPU should be used
    let use_gpu = should_enable_gpu(system);

    // n_gpu_layers: compute based on VRAM and model size
    let computed_n_gpu_layers = if use_gpu {
        compute_n_gpu_layers(system, model_layers, params_billions, quantization)
    } else {
        0
    };

    // n_threads: reduce if using GPU to leave headroom
    let base_threads = system.cpu_threads as i32;
    let n_threads = if use_gpu {
        // Use half threads when GPU is active for better balance
        (base_threads / 2).max(1)
    } else {
        base_threads
    };

    // n_ctx: use model's trained context, capped at reasonable max
    let n_ctx = model_n_ctx_train.max(512);

    // n_batch: scale with context size, cap at 2048
    let n_batch = (n_ctx / 2).clamp(32, 2048);

    // n_ubatch: should be divisor of n_batch, cap at 512
    let n_ubatch = (n_batch / 4).clamp(32, 512);

    // flash_attention: enabled by default (auto-detected by llama.cpp)
    let flash_attention = true;

    // Apply overrides from config
    let computed = ComputeParams {
        n_ctx: overrides.n_ctx.unwrap_or(n_ctx),
        n_batch: overrides.n_batch.unwrap_or(n_batch),
        n_ubatch: overrides.n_ubatch.unwrap_or(n_ubatch),
        n_gpu_layers: overrides.n_gpu_layers.unwrap_or(computed_n_gpu_layers),
        n_threads: overrides.n_threads.unwrap_or(n_threads),
        flash_attention: overrides.flash_attention.unwrap_or(flash_attention),
    };

    tracing::debug!(
        n_ctx = computed.n_ctx,
        n_batch = computed.n_batch,
        n_ubatch = computed.n_ubatch,
        n_gpu_layers = computed.n_gpu_layers,
        n_threads = computed.n_threads,
        flash_attention = computed.flash_attention,
        use_gpu = use_gpu,
        "computed params"
    );

    computed
}

/// Get the llama flash attention policy value.
/// -1 = AUTO, 0 = DISABLED, 1 = ENABLED
pub fn flash_attention_policy(enabled: bool) -> i32 {
    if enabled {
        -1 // LLAMA_FLASH_ATTN_TYPE_AUTO
    } else {
        0 // LLAMA_FLASH_ATTN_TYPE_DISABLED
    }
}

/// Handle memory error by trimming KV cache.
/// Returns true if trim was successful, false otherwise.
///
/// # Arguments
/// * `context` - Llama context with KV cache
/// * `is_hybrid` - Whether the model is hybrid/recurrent
/// * `n_ctx` - Context window size
/// * `checkpoint_manager` - Checkpoint manager for hybrid models
/// * `position` - Current position in context
/// * `previous_tokens` - Previous tokens vector to clear
pub fn handle_memory_error(
    context: &mut LlamaContext,
    is_hybrid: bool,
    n_ctx: i32,
    checkpoint_manager: Option<&mut crate::provider::gguf::checkpoint::CheckpointManager>,
    position: &mut i32,
    previous_tokens: &mut Vec<llama_cpp_2::token::LlamaToken>,
) -> bool {
    if is_hybrid {
        // Hybrid models: Full clear required as kv_cache_seq_keep doesn't work
        tracing::warn!("hybrid model OOM, clearing all state");

        // Clear all checkpoints
        if let Some(cm) = checkpoint_manager {
            cm.clear();
        }

        // Reset position and clear tokens
        *position = 0;
        previous_tokens.clear();

        // Clear KV cache
        context.clear_kv_cache();

        tracing::info!("hybrid state cleared for OOM recovery");
    } else {
        // Non-hybrid: Trim to 50% of context
        let n_keep = ((n_ctx as usize) / 2).max(512);

        // Trim KV cache - keep only the most recent tokens
        // Note: llama_kv_cache_seq_keep keeps only up to seq_id, we use 0 for all sequences
        context.llama_kv_cache_seq_keep(n_keep as i32);

        tracing::info!("trimmed KV cache to {} tokens for OOM recovery", n_keep);
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantization_from_file_type_q4_k() {
        assert_eq!(Quantization::from_file_type("Q4_K_M"), Quantization::Q4_K);
        assert_eq!(Quantization::from_file_type("Q4_K_S"), Quantization::Q4_K);
        assert_eq!(Quantization::from_file_type("q4_k_m"), Quantization::Q4_K);
    }

    #[test]
    fn test_quantization_from_file_type_q8() {
        assert_eq!(Quantization::from_file_type("Q8_0"), Quantization::Q8_0);
        assert_eq!(Quantization::from_file_type("q8_0"), Quantization::Q8_0);
    }

    #[test]
    fn test_quantization_from_file_type_q6() {
        assert_eq!(Quantization::from_file_type("Q6_K"), Quantization::Q6_K);
        assert_eq!(Quantization::from_file_type("q6_k"), Quantization::Q6_K);
    }

    #[test]
    fn test_quantization_from_file_type_q5() {
        assert_eq!(Quantization::from_file_type("Q5_K_S"), Quantization::Q5_K);
        assert_eq!(Quantization::from_file_type("Q5_K_M"), Quantization::Q5_K);
    }

    #[test]
    fn test_quantization_from_file_type_q4_0() {
        assert_eq!(Quantization::from_file_type("Q4_0"), Quantization::Q4_0);
    }

    #[test]
    fn test_quantization_from_file_type_f16() {
        assert_eq!(Quantization::from_file_type("F16"), Quantization::F16);
        assert_eq!(Quantization::from_file_type("f16"), Quantization::F16);
    }

    #[test]
    fn test_quantization_from_file_type_unknown() {
        assert_eq!(
            Quantization::from_file_type("unknown"),
            Quantization::Unknown
        );
        assert_eq!(Quantization::from_file_type(""), Quantization::Unknown);
        assert_eq!(Quantization::from_file_type("Q3"), Quantization::Unknown);
    }

    #[test]
    fn test_memory_per_layer_f16() {
        let mem = Quantization::F16.memory_per_layer_base_gb();
        assert!((mem - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_memory_per_layer_q8_0() {
        let mem = Quantization::Q8_0.memory_per_layer_base_gb();
        assert!((mem - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_memory_per_layer_q6_k() {
        let mem = Quantization::Q6_K.memory_per_layer_base_gb();
        assert!((mem - 0.7).abs() < 0.001);
    }

    #[test]
    fn test_memory_per_layer_q5_k() {
        let mem = Quantization::Q5_K.memory_per_layer_base_gb();
        assert!((mem - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_memory_per_layer_q4_k() {
        let mem = Quantization::Q4_K.memory_per_layer_base_gb();
        assert!((mem - 0.35).abs() < 0.001);
    }

    #[test]
    fn test_memory_per_layer_q4_0() {
        let mem = Quantization::Q4_0.memory_per_layer_base_gb();
        assert!((mem - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_memory_per_layer_unknown() {
        let mem = Quantization::Unknown.memory_per_layer_base_gb();
        assert!((mem - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_memory_trimmed_error_display() {
        let err = MemoryTrimmedError;
        let msg = err.to_string();
        assert!(msg.contains("memory"));
        assert!(msg.contains("trimmed"));
    }

    #[test]
    fn test_config_overrides_partial() {
        let overrides = ConfigOverrides {
            n_ctx: Some(8192),
            n_batch: None,
            n_gpu_layers: Some(16),
            n_threads: None,
            flash_attention: None,
            n_ubatch: None,
        };

        assert_eq!(overrides.n_ctx, Some(8192));
        assert_eq!(overrides.n_batch, None);
        assert_eq!(overrides.n_gpu_layers, Some(16));
    }

    #[test]
    fn test_compute_params_with_overrides() {
        let system = SystemInfo {
            cpu_threads: 8,
            devices: vec![],
        };
        let overrides = ConfigOverrides {
            n_ctx: Some(8192),
            n_batch: Some(1024),
            n_gpu_layers: Some(16),
            n_threads: Some(4),
            flash_attention: Some(false),
            n_ubatch: Some(256),
        };

        let params = compute_params(
            &system,
            4096, // model_n_ctx_train
            32,   // model_layers
            7.0,  // params_billions
            Quantization::Q4_K,
            &overrides,
        );

        assert_eq!(params.n_ctx, 8192);
        assert_eq!(params.n_batch, 1024);
        assert_eq!(params.n_gpu_layers, 16);
        assert_eq!(params.n_threads, 4);
        assert!(!params.flash_attention);
        assert_eq!(params.n_ubatch, 256);
    }

    #[test]
    fn test_compute_params_defaults() {
        let system = SystemInfo {
            cpu_threads: 8,
            devices: vec![],
        };
        let overrides = ConfigOverrides::default();

        let params = compute_params(
            &system,
            4096, // model_n_ctx_train
            32,   // model_layers
            7.0,  // params_billions
            Quantization::Q4_K,
            &overrides,
        );

        // Should use computed defaults (CPU only since no GPU)
        assert_eq!(params.n_ctx, 4096);
        assert_eq!(params.n_batch, 2048);
        assert_eq!(params.n_gpu_layers, 0); // No GPU in system
        assert_eq!(params.n_threads, 8);
        assert!(params.flash_attention);
    }

    #[test]
    fn test_detect_system_info() {
        let info = detect_system_info();

        // Should have at least CPU
        assert!(info.cpu_threads > 0);
        assert!(!info.devices.is_empty());

        // CPU should be present
        let cpu_device = info
            .devices
            .iter()
            .find(|d| d.device_type == LlamaBackendDeviceType::Cpu);
        assert!(cpu_device.is_some());
    }
}
