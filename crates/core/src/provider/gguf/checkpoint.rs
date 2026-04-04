//! Checkpoint manager for hybrid/recurrent/kv-cache model state.
//!
//! Provides state snapshot functionality using llama.cpp's state_seq_*_ext APIs
//! for models that cannot have KV cache selectively cleared.
//!
//! # For Hybrid/Recurrent Models
//!
//! Models like Mamba, RWKV, Jamba, Qwen3xx have recurrent/SSM components
//! where the state cannot be selectively manipulated. This manager uses
//! `PARTIAL_ONLY` flag to save/restore only the recurrent state.

use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::context::session::LlamaStateSeqFlags;
use llama_cpp_2::token::LlamaToken;
use tracing::debug;

/// Result of cache computation.
pub struct CacheStatus {
    pub tokens_to_skip: usize,
    pub checkpoint_restored: bool,
    pub restored_position: Option<i32>,
    pub cache_hit: bool,
    pub restored_transition: Option<TransitionType>,
}

/// Represents a transition point in the conversation where checkpoint is saved.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub enum TransitionType {
    /// Start of a new request (before processing any new tokens) - highest priority
    RequestStart,
    /// Start of a new turn (before user message)
    TurnStart,
    /// After tool call message (before tool execution)
    ToolCall,
    /// After tool response message (after tool completes)
    ToolResponse,
    /// End of turn (after assistant response)
    #[default]
    TurnEnd,
    /// Periodic checkpoint (every N tokens) - lowest priority
    Periodic,
}

/// Priority order for checkpoint matching (highest first)
pub fn checkpoint_priority(transition: TransitionType) -> u8 {
    match transition {
        TransitionType::RequestStart => 60,
        TransitionType::TurnStart => 50,
        TransitionType::ToolCall => 40,
        TransitionType::ToolResponse => 30,
        TransitionType::TurnEnd => 20,
        TransitionType::Periodic => 10,
    }
}

/// A checkpoint of model state at a specific position.
#[derive(Debug, Clone)]
pub struct Checkpoint {
    /// Token sequence this checkpoint represents.
    pub tokens: Vec<LlamaToken>,
    /// Raw state data (recurrent state only if using PARTIAL_ONLY).
    pub state: Vec<u8>,
    /// Position in the token sequence.
    pub position: usize,
    /// Size of the state data in bytes.
    pub state_size: usize,
    /// Transition type when checkpoint was created.
    pub transition: TransitionType,
}

impl Checkpoint {
    /// Creates a new checkpoint.
    pub fn new(
        tokens: Vec<LlamaToken>,
        state: Vec<u8>,
        position: usize,
        transition: TransitionType,
    ) -> Self {
        let state_size = state.len();
        Self {
            tokens,
            state,
            position,
            state_size,
            transition,
        }
    }
}

/// Calculate max checkpoints based on available CPU RAM.
/// Uses up to 2GB of CPU RAM for checkpoints.
///
/// # Arguments
/// * `state_size_bytes` - Size of a single checkpoint in bytes (measured after first checkpoint)
///
/// # Returns
/// Number of checkpoints to maintain (1-16)
pub fn calculate_max_checkpoints(state_size_bytes: usize) -> usize {
    const MAX_RAM_BYTES: usize = 2 * 1024 * 1024 * 1024; // 2GB
    const MIN_CHECKPOINTS: usize = 1;
    const MAX_CHECKPOINTS: usize = 16;
    const BUFFER_BYTES: usize = 512 * 1024 * 1024; // 512MB buffer for system

    if state_size_bytes == 0 {
        return MIN_CHECKPOINTS;
    }

    let usable = MAX_RAM_BYTES.saturating_sub(BUFFER_BYTES);
    let checkpoints_from_memory = usable / state_size_bytes;

    checkpoints_from_memory.clamp(MIN_CHECKPOINTS, MAX_CHECKPOINTS)
}

/// Manages state checkpoints for hybrid/recurrent models.
#[derive(Debug)]
pub struct CheckpointManager {
    checkpoints: Vec<Checkpoint>,
    max_checkpoints: usize,
    is_hybrid: bool,
    flags: LlamaStateSeqFlags,
    first_checkpoint_size: Option<usize>,
    /// Dynamic periodic checkpoint interval (calculated from n_ctx and max_checkpoints)
    periodic_interval: usize,
    /// Last position where a periodic checkpoint was saved
    last_periodic_position: usize,
}

impl CheckpointManager {
    /// Creates a new CheckpointManager.
    ///
    /// # Arguments
    /// * `max_checkpoints` - Maximum number of checkpoints to keep (FIFO eviction)
    /// * `is_hybrid` - Whether the model is hybrid/recurrent (uses PARTIAL_ONLY flag)
    /// * `n_ctx` - Context window size (used to calculate periodic interval)
    pub fn new(max_checkpoints: usize, is_hybrid: bool, n_ctx: usize) -> Self {
        let flags = if is_hybrid {
            LlamaStateSeqFlags::PARTIAL_ONLY
        } else {
            LlamaStateSeqFlags::empty()
        };

        // Calculate periodic checkpoint interval
        // Reserve 4 slots for transitions + 4 for request boundaries = 8
        // Remaining for periodic
        let periodic_slots = max_checkpoints.saturating_sub(8).max(1);
        let periodic_interval = n_ctx / periodic_slots;

        Self {
            checkpoints: Vec::new(),
            max_checkpoints,
            is_hybrid,
            flags,
            first_checkpoint_size: None,
            periodic_interval,
            last_periodic_position: 0,
        }
    }

    /// Checks if context can overflow with current prompt.
    pub fn is_context_overflow(
        &self,
        n_ctx: i32,
        effective_position: i32,
        total_tokens: usize,
        tokens_to_skip: usize,
    ) -> bool {
        if self.is_hybrid {
            // Hybrid: overflow if prompt is longer than available context space
            let new_tokens = (total_tokens as i32 - tokens_to_skip as i32).max(0);
            new_tokens > (n_ctx - effective_position)
        } else {
            // Non-hybrid: only overflow if no checkpoint and prompt longer than cached
            tokens_to_skip == 0
                && effective_position > 0
                && total_tokens as i32 > effective_position
        }
    }

    /// Computes cache status for given tokens.
    pub fn cache_status(
        &self,
        is_hybrid: bool,
        current_position: i32,
        tokens: &[LlamaToken],
        previous_tokens: &[LlamaToken],
    ) -> CacheStatus {
        let mut checkpoint_restored = false;
        let mut checkpoint_tokens_skipped = 0;
        let mut restored_position: Option<i32> = None;

        // KV cache partial prefix matching for non-hybrid models.
        if !is_hybrid {
            let common_prefix_len = tokens
                .iter()
                .zip(previous_tokens.iter())
                .take_while(|(a, b)| a == b)
                .count();

            // Non-hybrid: use KV cache prefix matching.
            // Only count as cache hit if prefix is substantial (>= 8 tokens) to avoid false positives.
            let tokens_to_skip = if common_prefix_len >= 8 {
                common_prefix_len
            } else {
                0
            }
            .min(tokens.len());

            debug!(
                "Checkpoint check: is_hybrid={}, current_position={}, tokens_len={}, previous_tokens_len={}, common_prefix_len={}, tokens_to_skip={}",
                is_hybrid,
                current_position,
                tokens.len(),
                previous_tokens.len(),
                common_prefix_len,
                tokens_to_skip
            );
            return CacheStatus {
                tokens_to_skip,
                checkpoint_restored: false,
                restored_position: None,
                cache_hit: tokens_to_skip > 0,
                restored_transition: None,
            };
        }

        // Hybrid: use checkpoint exact prefix matching with priority.
        let mut best_match_priority: u8 = 0;
        let mut best_match_transition: Option<TransitionType> = None;
        if !self.checkpoints.is_empty() && !tokens.is_empty() {
            for cp in &self.checkpoints {
                // Check if checkpoint matches as prefix
                let is_prefix = if cp.tokens.is_empty() {
                    false
                } else if cp.tokens.len() <= tokens.len() {
                    cp.tokens.iter().zip(tokens.iter()).all(|(a, b)| a == b)
                } else {
                    false
                };

                if is_prefix {
                    let priority = checkpoint_priority(cp.transition);
                    debug!(
                        "Checkpoint check: cp_position={}, cp_tokens_len={}, transition={:?}, priority={}, current_position={}",
                        cp.position,
                        cp.tokens.len(),
                        cp.transition,
                        priority,
                        current_position
                    );

                    // Pick highest priority match
                    if priority > best_match_priority {
                        best_match_priority = priority;
                        checkpoint_restored = true;
                        checkpoint_tokens_skipped = cp.position;
                        restored_position = Some(cp.position as i32);
                        best_match_transition = Some(cp.transition);
                    }
                }
            }

            if !checkpoint_restored {
                debug!(
                    "No checkpoint match: is_hybrid={}, {} checkpoints available, current request has {} tokens",
                    is_hybrid,
                    self.checkpoints.len(),
                    tokens.len()
                );
            }
        }

        let tokens_to_skip = if checkpoint_restored {
            checkpoint_tokens_skipped
        } else {
            0
        };

        CacheStatus {
            tokens_to_skip,
            checkpoint_restored,
            restored_position,
            cache_hit: tokens_to_skip > 0,
            restored_transition: best_match_transition,
        }
    }

    /// Adjusts max_checkpoints based on actual checkpoint size.
    /// Should be called after first checkpoint is saved.
    pub fn adjust_max_checkpoints(&mut self) {
        if let Some(first) = self.checkpoints.first()
            && self.first_checkpoint_size.is_none()
        {
            self.first_checkpoint_size = Some(first.state_size);
            let calculated = calculate_max_checkpoints(first.state_size);
            tracing::debug!(
                "First checkpoint size: {} bytes, adjusting max_checkpoints from {} to {}",
                first.state_size,
                self.max_checkpoints,
                calculated
            );
            self.max_checkpoints = calculated;

            // Evict excess checkpoints if needed
            while self.checkpoints.len() > self.max_checkpoints {
                self.checkpoints.remove(0);
            }
        }
    }

    /// Checks if we should save a periodic checkpoint based on tokens processed.
    ///
    /// # Arguments
    /// * `current_position` - Current position in the sequence
    ///
    /// # Returns
    /// `true` if a periodic checkpoint should be saved
    pub fn should_save_periodic(&self, current_position: usize) -> bool {
        if self.periodic_interval == 0 {
            return false;
        }
        let tokens_since_last = current_position.saturating_sub(self.last_periodic_position);
        tokens_since_last >= self.periodic_interval
    }

    /// Saves a checkpoint at a specific transition point.
    ///
    /// # Arguments
    /// * `ctx` - The llama context to checkpoint
    /// * `tokens` - The tokens that produced this state
    /// * `position` - Current position in the sequence
    /// * `transition` - The type of transition
    ///
    /// # Returns
    /// `true` if checkpoint was saved, `false` if failed or disabled.
    pub fn save_at_transition(
        &mut self,
        ctx: &LlamaContext,
        tokens: Vec<LlamaToken>,
        position: usize,
        transition: TransitionType,
    ) -> bool {
        if self.max_checkpoints == 0 {
            return false;
        }

        // Get state size
        let state_size = ctx.state_seq_get_size_ext(0, self.flags);
        if state_size == 0 {
            tracing::debug!("Checkpoint at {:?}: state size is 0, skipping", transition);
            return false;
        }

        // Allocate and copy state
        let mut state = vec![0u8; state_size];
        let n_written = unsafe { ctx.state_seq_get_data_ext(state.as_mut_ptr(), 0, self.flags) };

        if n_written != state_size {
            tracing::warn!(
                "Checkpoint at {:?}: wrote {} bytes, expected {}",
                transition,
                n_written,
                state_size
            );
            return false;
        }

        // Create checkpoint with transition type
        let checkpoint = Checkpoint::new(tokens, state, position, transition);

        // Check if we already have a checkpoint at this transition type
        // Remove existing checkpoint of same type to avoid duplicates
        // self.checkpoints.retain(|cp| cp.transition != transition);

        self.checkpoints.push(checkpoint);

        // Adjust max checkpoints after first save
        self.adjust_max_checkpoints();

        // FIFO eviction (by total count, keeping latest, remove oldest of same type)
        while self.checkpoints.len() > self.max_checkpoints {
            let oldest_index = self
                .checkpoints
                .iter()
                .position(|cp| cp.transition == transition)
                .unwrap_or(0);
            self.checkpoints.remove(oldest_index);
        }

        tracing::debug!(
            "Checkpoint saved at {:?}: {} bytes, {} checkpoints total, position={}",
            transition,
            state_size,
            self.checkpoints.len(),
            position
        );

        true
    }

    /// Restores a checkpoint to the context.
    ///
    /// # Arguments
    /// * `ctx` - The llama context to restore to
    /// * `checkpoint` - The checkpoint to restore
    ///
    /// # Returns
    /// `true` if restoration succeeded, `false` otherwise.
    pub fn restore(&self, ctx: &mut LlamaContext, checkpoint: &Checkpoint) -> bool {
        // Verify state size hasn't changed
        let current_size = ctx.state_seq_get_size_ext(0, self.flags);
        if current_size != checkpoint.state_size {
            tracing::warn!(
                "Checkpoint restore: size mismatch (current={}, expected={})",
                current_size,
                checkpoint.state_size
            );
            return false;
        }

        // Restore state
        let success = unsafe { ctx.state_seq_set_data_ext(&checkpoint.state, 0, self.flags) };

        if success {
            tracing::debug!("Checkpoint restored at position {}", checkpoint.position);
        } else {
            tracing::warn!("Checkpoint restore failed");
        }

        success
    }

    /// Returns the number of checkpoints.
    pub fn len(&self) -> usize {
        self.checkpoints.len()
    }

    /// Returns a reference to the checkpoints.
    pub fn checkpoints(&self) -> &Vec<Checkpoint> {
        &self.checkpoints
    }

    /// Returns true if there are no checkpoints.
    pub fn is_empty(&self) -> bool {
        self.checkpoints.is_empty()
    }

    /// Clears all checkpoints.
    pub fn clear(&mut self) {
        self.checkpoints.clear();
    }
}

impl Default for CheckpointManager {
    fn default() -> Self {
        Self::new(16, false, 4096)
    }
}

#[cfg(test)]
mod checkpoint_tests {
    use super::*;

    fn token(id: i32) -> LlamaToken {
        LlamaToken::new(id)
    }

    #[test]
    fn test_is_context_overflow_hybrid_true() {
        let mgr = CheckpointManager::new(10, true, 2048);
        assert!(mgr.is_context_overflow(2048, 100, 2000, 0));
    }

    #[test]
    fn test_is_context_overflow_hybrid_false() {
        let mgr = CheckpointManager::new(10, true, 2048);
        assert!(!mgr.is_context_overflow(2048, 100, 500, 0));
    }

    #[test]
    fn test_is_context_overflow_hybrid_with_checkpoint() {
        let mgr = CheckpointManager::new(10, true, 2048);
        assert!(!mgr.is_context_overflow(2048, 50, 2000, 50));
    }

    #[test]
    fn test_is_context_overflow_non_hybrid_no_checkpoint() {
        let mgr = CheckpointManager::new(10, false, 2048);
        assert!(mgr.is_context_overflow(2048, 100, 500, 0));
    }

    #[test]
    fn test_is_context_overflow_non_hybrid_with_prefix() {
        let mgr = CheckpointManager::new(10, false, 2048);
        assert!(!mgr.is_context_overflow(2048, 100, 500, 100));
    }

    #[test]
    fn test_is_context_overflow_non_hybrid_at_start() {
        let mgr = CheckpointManager::new(10, false, 2048);
        assert!(!mgr.is_context_overflow(2048, 0, 500, 0));
    }

    #[test]
    fn test_is_context_overflow_with_checkpoint_at_high_position() {
        let mgr = CheckpointManager::new(10, true, 8192);
        assert!(!mgr.is_context_overflow(8192, 5048, 5028, 5048));
    }

    #[test]
    fn test_is_context_overflow_with_checkpoint_zero_tokens() {
        let mgr = CheckpointManager::new(10, true, 8192);
        assert!(!mgr.is_context_overflow(8192, 100, 50, 100));
    }

    #[test]
    fn test_is_context_overflow_without_checkpoint_at_high_position() {
        let mgr = CheckpointManager::new(10, true, 8192);
        assert!(mgr.is_context_overflow(8192, 5048, 5028, 0));
    }

    // cache_status tests
    #[test]
    fn test_cache_status_hybrid_checkpoint_matches_prefix() {
        let checkpoints = vec![Checkpoint::new(
            vec![token(1), token(2), token(3)],
            vec![1, 2, 3],
            3,
            TransitionType::TurnEnd,
        )];
        let tokens = vec![token(1), token(2), token(3), token(4), token(5)];
        let previous_tokens: Vec<LlamaToken> = vec![];

        let mgr = CheckpointManager::new(10, true, 4096);
        // Manually add checkpoints for testing
        let mgr = add_checkpoints(mgr, checkpoints);

        let result = mgr.cache_status(true, 0, &tokens, &previous_tokens);

        assert!(result.checkpoint_restored);
        assert_eq!(result.tokens_to_skip, 3);
        assert_eq!(result.restored_position, Some(3));
        assert!(result.cache_hit);
    }

    #[test]
    fn test_cache_status_hybrid_no_checkpoint_match() {
        let checkpoints = vec![Checkpoint::new(
            vec![token(1), token(2), token(3)],
            vec![1, 2, 3],
            3,
            TransitionType::TurnEnd,
        )];
        let tokens = vec![token(4), token(5), token(6)];
        let previous_tokens: Vec<LlamaToken> = vec![];

        let mgr = add_checkpoints(CheckpointManager::new(10, true, 4096), checkpoints);
        let result = mgr.cache_status(true, 0, &tokens, &previous_tokens);

        assert!(!result.checkpoint_restored);
        assert_eq!(result.tokens_to_skip, 0);
        assert!(!result.cache_hit);
    }

    #[test]
    fn test_cache_status_hybrid_empty_checkpoints() {
        let tokens = vec![token(1), token(2), token(3)];
        let previous_tokens: Vec<LlamaToken> = vec![];

        let mgr = CheckpointManager::new(10, true, 4096);
        let result = mgr.cache_status(true, 0, &tokens, &previous_tokens);

        assert!(!result.checkpoint_restored);
        assert_eq!(result.tokens_to_skip, 0);
        assert!(!result.cache_hit);
    }

    #[test]
    fn test_cache_status_non_hybrid_prefix_match() {
        let tokens = vec![
            token(1),
            token(2),
            token(3),
            token(4),
            token(5),
            token(6),
            token(7),
            token(8),
            token(9),
            token(10),
        ];
        let previous_tokens = vec![
            token(1),
            token(2),
            token(3),
            token(4),
            token(5),
            token(6),
            token(7),
            token(8),
        ];

        let mgr = CheckpointManager::new(10, false, 4096);
        let result = mgr.cache_status(false, 8, &tokens, &previous_tokens);

        assert!(!result.checkpoint_restored);
        assert_eq!(result.tokens_to_skip, 8);
        assert!(result.cache_hit);
    }

    #[test]
    fn test_cache_status_non_hybrid_short_prefix() {
        let tokens = vec![token(1), token(2), token(3)];
        let previous_tokens = vec![token(1), token(2)];

        let mgr = CheckpointManager::new(10, false, 4096);
        let result = mgr.cache_status(false, 2, &tokens, &previous_tokens);

        assert!(!result.checkpoint_restored);
        assert_eq!(result.tokens_to_skip, 0);
        assert!(!result.cache_hit);
    }

    #[test]
    fn test_cache_status_checkpoint_longer_than_tokens() {
        let checkpoints = vec![Checkpoint::new(
            vec![token(1), token(2), token(3), token(4), token(5)],
            vec![1, 2, 3, 4, 5],
            5,
            TransitionType::TurnEnd,
        )];
        let tokens = vec![token(1), token(2)];

        let mgr = add_checkpoints(CheckpointManager::new(10, true, 4096), checkpoints);
        let result = mgr.cache_status(true, 0, &tokens, &[]);

        assert!(!result.checkpoint_restored);
    }

    #[test]
    fn test_cache_status_checkpoint_empty_skipped() {
        let checkpoints = vec![Checkpoint::new(
            vec![],
            vec![],
            5048,
            TransitionType::RequestStart,
        )];
        let tokens = vec![token(1), token(2), token(3), token(4), token(5)];
        let previous_tokens: Vec<LlamaToken> = vec![];

        let mgr = add_checkpoints(CheckpointManager::new(10, true, 4096), checkpoints);
        let result = mgr.cache_status(true, 0, &tokens, &previous_tokens);

        assert!(!result.checkpoint_restored);
    }

    #[test]
    fn test_cache_status_checkpoint_position_greater_than_tokens() {
        let checkpoints = vec![Checkpoint::new(
            vec![token(1), token(2), token(3)],
            vec![],
            100,
            TransitionType::TurnEnd,
        )];
        let tokens = vec![token(1), token(2)];

        let mgr = add_checkpoints(CheckpointManager::new(10, true, 4096), checkpoints);
        let result = mgr.cache_status(true, 0, &tokens, &[]);

        assert!(!result.checkpoint_restored);
    }

    #[test]
    fn test_cache_status_prioritizes_turn_end_over_periodic() {
        let checkpoints = vec![
            Checkpoint::new(
                vec![token(1), token(2), token(3)],
                vec![1, 2, 3],
                3,
                TransitionType::Periodic,
            ),
            Checkpoint::new(
                vec![token(1), token(2), token(3)],
                vec![1, 2, 3],
                3,
                TransitionType::TurnEnd,
            ),
        ];
        let tokens = vec![token(1), token(2), token(3), token(4), token(5)];
        let previous_tokens: Vec<LlamaToken> = vec![];

        let mgr = add_checkpoints(CheckpointManager::new(10, true, 4096), checkpoints);
        let result = mgr.cache_status(true, 0, &tokens, &previous_tokens);

        assert!(result.checkpoint_restored);
        assert_eq!(result.restored_position, Some(3));
    }

    #[test]
    fn test_cache_status_prioritizes_request_start() {
        let checkpoints = vec![
            Checkpoint::new(
                vec![token(1), token(2), token(3)],
                vec![1, 2, 3],
                3,
                TransitionType::TurnEnd,
            ),
            Checkpoint::new(
                vec![token(1), token(2), token(3)],
                vec![1, 2, 3],
                3,
                TransitionType::Periodic,
            ),
        ];
        let tokens = vec![token(1), token(2), token(3), token(4), token(5)];
        let previous_tokens: Vec<LlamaToken> = vec![];

        let mgr = add_checkpoints(CheckpointManager::new(10, true, 4096), checkpoints);
        let result = mgr.cache_status(true, 0, &tokens, &previous_tokens);

        assert!(result.checkpoint_restored);
        assert_eq!(result.restored_position, Some(3));
    }

    #[test]
    fn test_cache_status_empty_token_checkpoints_skipped() {
        let checkpoints = vec![
            Checkpoint::new(
                vec![token(1), token(2), token(3)],
                vec![1, 2, 3],
                3,
                TransitionType::TurnEnd,
            ),
            Checkpoint::new(vec![], vec![], 100, TransitionType::RequestStart),
        ];
        let tokens = vec![token(1), token(2), token(3), token(4), token(5)];
        let previous_tokens: Vec<LlamaToken> = vec![];

        let mgr = add_checkpoints(CheckpointManager::new(10, true, 4096), checkpoints);
        let result = mgr.cache_status(true, 0, &tokens, &previous_tokens);

        assert!(result.checkpoint_restored);
        assert_eq!(result.restored_position, Some(3));
    }

    // Helper function to add checkpoints to manager for testing
    fn add_checkpoints(
        mut mgr: CheckpointManager,
        checkpoints: Vec<Checkpoint>,
    ) -> CheckpointManager {
        mgr.checkpoints = checkpoints;
        mgr
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn token(id: i32) -> LlamaToken {
        LlamaToken::new(id)
    }

    #[test]
    fn test_checkpoint_creation() {
        let cp = Checkpoint::new(
            vec![token(1), token(2), token(3)],
            vec![1, 2, 3],
            3,
            TransitionType::TurnEnd,
        );
        assert_eq!(cp.tokens.len(), 3);
        assert_eq!(cp.position, 3);
        assert_eq!(cp.state_size, 3);
        assert_eq!(cp.transition, TransitionType::TurnEnd);
    }

    #[test]
    fn test_manager_creation() {
        let mgr = CheckpointManager::new(10, true, 4096);
        assert!(mgr.is_empty());
        assert!(mgr.is_hybrid);
        assert_eq!(mgr.max_checkpoints, 10);
    }
}
