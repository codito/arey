use crate::cli::chat::commands::{CliCommand, Command, parse_command_line};
use crate::cli::ux::style_chat_text;
use clap::Parser;
use rustyline::completion::{Candidate, Completer};
use rustyline::error::ReadlineError;
use rustyline::hint::Hinter;
use rustyline::{Helper, Highlighter, Validator};

/// Completion candidate for the REPL.
#[derive(Debug)]
pub struct CompletionCandidate {
    text: String,
    display_string: String,
}

impl CompletionCandidate {
    pub fn new(text: &str) -> Self {
        let display_string =
            style_chat_text(text, crate::cli::ux::ChatMessageType::Footer).to_string();
        Self {
            text: text.to_owned(),
            display_string,
        }
    }
}

impl Candidate for CompletionCandidate {
    fn display(&self) -> &str {
        &self.display_string
    }

    fn replacement(&self) -> &str {
        &self.text
    }
}

/// REPL runtime state for command line editing.
#[derive(Helper, Validator, Highlighter)]
pub struct Repl {
    pub command_names: Vec<String>,
    pub tool_names: Vec<String>,
    pub model_names: Vec<String>,
    pub profile_names: Vec<String>,
}

impl Completer for Repl {
    type Candidate = CompletionCandidate;

    fn complete(
        &self,
        line: &str,
        pos: usize,
        _ctx: &rustyline::Context<'_>,
    ) -> Result<(usize, Vec<Self::Candidate>), ReadlineError> {
        let is_command = line.starts_with('/') || line.starts_with('!') || line.starts_with('@');
        if !is_command {
            return Ok((0, Vec::new()));
        }

        let args = parse_command_line(line);
        if let Ok(cli_command) = CliCommand::try_parse_from(&args) {
            return match cli_command.command {
                Command::Tool { .. } => tool_compl(line, pos, &self.tool_names),
                Command::Model { .. } => model_compl(line, pos, &self.model_names),
                Command::Profile { .. } => profile_compl(line, pos, &self.profile_names),
                _ => Ok((0, Vec::new())),
            };
        }

        let candidates = self
            .command_names
            .iter()
            .filter(|name| name.starts_with(line))
            .map(|name| CompletionCandidate::new(name))
            .collect();

        Ok((0, candidates))
    }
}

impl Hinter for Repl {
    type Hint = String;

    fn hint(&self, line: &str, pos: usize, _ctx: &rustyline::Context<'_>) -> Option<Self::Hint> {
        if line.is_empty() || pos < line.len() {
            return None;
        }
        if line.starts_with('/') {
            // Suggest command completions
            self.command_names
                .iter()
                .find(|&cmd_name| cmd_name.starts_with(line))
                .map(|cmd_name| cmd_name[line.len()..].into())
        } else {
            None
        }
    }
}

// Model command completion
fn model_compl(
    line: &str,
    pos: usize,
    model_names: &[String],
) -> Result<(usize, Vec<CompletionCandidate>), ReadlineError> {
    let line_to_pos = &line[..pos];
    if let Some(space_pos) = line_to_pos.rfind(' ') {
        let model_prefix_start = space_pos + 1;
        if model_prefix_start <= line_to_pos.len() {
            let model_prefix = &line_to_pos[model_prefix_start..];
            let mut candidates = model_names
                .iter()
                .filter(|name| name.starts_with(model_prefix))
                .map(|name| CompletionCandidate::new(name))
                .collect::<Vec<_>>();

            if "list".starts_with(model_prefix) && !model_names.contains(&"list".to_string()) {
                candidates.push(CompletionCandidate::new("list"));
            }
            return Ok((model_prefix_start, candidates));
        }
    }
    Ok((0, Vec::new()))
}

// Profile command completion
fn profile_compl(
    line: &str,
    pos: usize,
    profile_names: &[String],
) -> Result<(usize, Vec<CompletionCandidate>), ReadlineError> {
    let line_to_pos = &line[..pos];
    if let Some(space_pos) = line_to_pos.rfind(' ') {
        let profile_prefix_start = space_pos + 1;
        if profile_prefix_start <= line_to_pos.len() {
            let profile_prefix = &line_to_pos[profile_prefix_start..];
            let mut candidates = profile_names
                .iter()
                .filter(|name| name.starts_with(profile_prefix))
                .map(|name| CompletionCandidate::new(name))
                .collect::<Vec<_>>();

            if "list".starts_with(profile_prefix) && !profile_names.contains(&"list".to_string()) {
                candidates.push(CompletionCandidate::new("list"));
            }
            return Ok((profile_prefix_start, candidates));
        }
    }
    Ok((0, Vec::new()))
}

/// Tool command completion
fn tool_compl(
    line: &str,
    pos: usize,
    tool_names: &[String],
) -> Result<(usize, Vec<CompletionCandidate>), ReadlineError> {
    let line_to_pos = &line[..pos];
    if let Some(space_pos) = line_to_pos.rfind(' ') {
        let tool_prefix_start = space_pos + 1;
        if tool_prefix_start <= line_to_pos.len() {
            let tool_prefix = &line_to_pos[tool_prefix_start..];
            let candidates = tool_names
                .iter()
                .filter(|name| name.starts_with(tool_prefix))
                .map(|name| CompletionCandidate::new(name))
                .collect();
            return Ok((tool_prefix_start, candidates));
        }
    }

    Ok((0, Vec::new()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustyline::history::DefaultHistory;

    #[test]
    fn test_repl_completer_for_commands() {
        let repl = Repl {
            command_names: vec!["/help".to_string(), "/clear".to_string()],
            tool_names: vec![],
            model_names: vec![],
            profile_names: vec![],
        };
        let line = "/c";
        let history = DefaultHistory::new();
        let (start, candidates) = repl
            .complete(line, line.len(), &rustyline::Context::new(&history))
            .unwrap();
        assert_eq!(start, 0);
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].replacement(), "/clear");
    }

    #[test]
    fn test_profile_command_completion() {
        let history = DefaultHistory::new();
        let repl = Repl {
            command_names: vec![],
            tool_names: vec![],
            model_names: vec![],
            profile_names: vec!["profile1".to_string(), "profile2".to_string()],
        };
        let line = "/profile pro";
        let (start, candidates) = repl
            .complete(line, line.len(), &rustyline::Context::new(&history))
            .unwrap();
        assert_eq!(start, 9); // "/profile ".len()
        assert_eq!(candidates.len(), 2);
    }

    #[test]
    fn test_tool_command_completion() {
        let history = DefaultHistory::new();
        let repl = Repl {
            command_names: vec![],
            tool_names: vec!["search".to_string(), "browse".to_string()],
            model_names: vec![],
            profile_names: vec![],
        };
        let line = "/tool se";
        let (start, candidates) = repl
            .complete(line, line.len(), &rustyline::Context::new(&history))
            .unwrap();
        assert_eq!(start, 6); // "/tool ".len()
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].replacement(), "search");
    }

    #[test]
    fn test_model_command_completion() {
        let history = DefaultHistory::new();

        // Test command-line completion for the model command
        let repl = Repl {
            command_names: vec![],
            tool_names: vec![],
            model_names: vec!["model1".to_string(), "model2".to_string()],
            profile_names: vec![],
        };

        // Simulate user typing "/model mod"
        let line = "/model mod";
        let (start, candidates) = repl
            .complete(line, line.len(), &rustyline::Context::new(&history))
            .unwrap();

        // Expecting completion to start at the model prefix (after the space)
        assert_eq!(start, 7); // "/model ".len() is 7
        assert_eq!(candidates.len(), 2);
        assert_eq!(candidates[0].replacement(), "model1");
        assert_eq!(candidates[1].replacement(), "model2");

        // Simulate user typing "/model l"
        let line = "/model l";
        let (start, candidates) = repl
            .complete(line, line.len(), &rustyline::Context::new(&history))
            .unwrap();
        assert_eq!(start, 7);
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].replacement(), "list");
    }

    #[test]
    fn test_repl_completer_for_subcommands() {
        let repl = Repl {
            command_names: vec![],
            tool_names: vec!["search".to_string()],
            model_names: vec!["model-1".to_string()],
            profile_names: vec!["prof-1".to_string()],
        };
        let history = DefaultHistory::new();
        let ctx = rustyline::Context::new(&history);

        // Test tool completion delegation
        let line = "/tool s";
        let (start, candidates) = repl.complete(line, line.len(), &ctx).unwrap();
        assert_eq!(start, 6);
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].replacement(), "search");

        // Test profile completion delegation
        let line = "/profile p";
        let (start, candidates) = repl.complete(line, line.len(), &ctx).unwrap();
        assert_eq!(start, 9);
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].replacement(), "prof-1");
    }

    #[test]
    fn test_repl_hinter() {
        let repl = Repl {
            command_names: vec!["/help".to_string(), "/clear".to_string()],
            tool_names: vec![],
            model_names: vec![],
            profile_names: vec![],
        };
        let history = DefaultHistory::new();
        let ctx = rustyline::Context::new(&history);

        // Test successful hint
        let line = "/h";
        let hint = repl.hint(line, line.len(), &ctx).unwrap();
        assert_eq!(hint, "elp");

        // Test no hint for non-command
        assert!(repl.hint("abc", 3, &ctx).is_none());
        // Test no hint when cursor is not at the end
        assert!(repl.hint("/help", 3, &ctx).is_none());
        // Test no hint for empty line
        assert!(repl.hint("", 0, &ctx).is_none());
    }
}
