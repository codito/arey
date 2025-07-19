//! Services layer for the app. Uses arey_core domain business logic to provide services for the
//! cli.

/// Manages interactive chat sessions.
pub mod chat;
/// Handles running prompts from play files.
pub mod play;
/// Handles executing single, non-interactive instructions.
pub mod run;
