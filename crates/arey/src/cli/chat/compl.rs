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
    pub agent_names: Vec<String>,
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
                Command::Tool { .. } => {
                    let mut candidates = self.tool_names.clone();
                    candidates.push("clear".to_string());
                    compl(line, pos, &candidates)
                }
                Command::Model { .. } => {
                    let candidates = self.model_names.clone();
                    compl(line, pos, &candidates)
                }
                Command::Profile { .. } => {
                    let candidates = self.profile_names.clone();
                    compl(line, pos, &candidates)
                }
                Command::Agent { .. } => {
                    let candidates = self.agent_names.clone();
                    compl(line, pos, &candidates)
                }
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

/// Command completion function
fn compl(
    line: &str,
    pos: usize,
    names: &[String],
) -> Result<(usize, Vec<CompletionCandidate>), ReadlineError> {
    let line_to_pos = &line[..pos];
    if let Some(space_pos) = line_to_pos.rfind(' ') {
        let prefix_start = space_pos + 1;
        if prefix_start <= line_to_pos.len() {
            let prefix = &line_to_pos[prefix_start..];
            let candidates = names
                .iter()
                .filter(|name| name.starts_with(prefix))
                .map(|name| CompletionCandidate::new(name))
                .collect::<Vec<_>>();
            return Ok((prefix_start, candidates));
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
            agent_names: vec![],
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
    fn test_repl_hinter() {
        let repl = Repl {
            command_names: vec!["/help".to_string(), "/clear".to_string()],
            tool_names: vec![],
            model_names: vec![],
            profile_names: vec![],
            agent_names: vec![],
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

    #[test]
    fn test_completion_function_basic() {
        let names = vec!["model1".to_string(), "model2".to_string()];

        // Test basic name completion
        let result = compl("/model m", 8, &names).unwrap();
        assert_eq!(result.0, 7); // "/model ".len()
        assert_eq!(result.1.len(), 2); // model1, model2
        assert_eq!(result.1[0].replacement(), "model1");
        assert_eq!(result.1[1].replacement(), "model2");
    }

    #[test]
    fn test_completion_function_all_commands() {
        let tool_names = vec![
            "search".to_string(),
            "browse".to_string(),
            "clear".to_string(),
        ];
        let model_names = vec!["gpt-4".to_string(), "claude".to_string()];

        // Test tool completion with search
        let result = compl("/tool s", 7, &tool_names).unwrap();
        assert_eq!(result.0, 6); // "/tool ".len()
        assert_eq!(result.1.len(), 1);
        assert_eq!(result.1[0].replacement(), "search");

        // Test tool completion with "clear"
        let result = compl("/tool c", 7, &tool_names).unwrap();
        assert_eq!(result.0, 6);
        assert_eq!(result.1.len(), 1);
        assert_eq!(result.1[0].replacement(), "clear");

        // Test model completion
        let result = compl("/model g", 8, &model_names).unwrap();
        assert_eq!(result.0, 7);
        assert_eq!(result.1.len(), 1);
        assert_eq!(result.1[0].replacement(), "gpt-4");

        // Test no matches
        let result = compl("/tool x", 7, &tool_names).unwrap();
        assert_eq!(result.0, 6);
        assert_eq!(result.1.len(), 0);
    }

    #[test]
    fn test_completion_function_edge_cases() {
        // Test with empty names
        let result = compl("/agent l", 8, &[]).unwrap();
        assert_eq!(result.0, 7);
        assert_eq!(result.1.len(), 0);

        // Test with no space in command (edge case)
        let result = compl("/agent", 6, &["test".to_string()]).unwrap();
        assert_eq!(result.0, 0);
        assert_eq!(result.1.len(), 0);

        // Test with multiple matches
        let names = vec!["clear".to_string(), "load".to_string()];
        let result = compl("/tool l", 7, &names).unwrap();
        assert_eq!(result.0, 6);
        assert_eq!(result.1.len(), 1); // "load" only
    }

    #[test]
    fn test_repl_integration_with_all_commands() {
        let repl = Repl {
            command_names: vec![],
            tool_names: vec!["search".to_string()],
            model_names: vec!["gpt-4".to_string(), "claude".to_string()],
            profile_names: vec!["creative".to_string(), "precise".to_string()],
            agent_names: vec!["coder".to_string(), "researcher".to_string()],
        };
        let history = DefaultHistory::new();
        let ctx = rustyline::Context::new(&history);

        // Test tool completion (now supports list)
        let (start, candidates) = repl.complete("/tool s", 7, &ctx).unwrap();
        assert_eq!(start, 6);
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].replacement(), "search");

        // Test tool completion with "clear"
        let (start, candidates) = repl.complete("/tool c", 7, &ctx).unwrap();
        assert_eq!(start, 6);
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].replacement(), "clear");

        // Test model completion
        let (start, candidates) = repl.complete("/model gp", 8, &ctx).unwrap();
        assert_eq!(start, 7);
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].replacement(), "gpt-4");

        // Test profile completion
        let (start, candidates) = repl.complete("/profile cr", 10, &ctx).unwrap();
        assert_eq!(start, 9);
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].replacement(), "creative");

        // Test agent completion (with list)
        let (start, candidates) = repl.complete("/agent r", 8, &ctx).unwrap();
        assert_eq!(start, 7);
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].replacement(), "researcher");
    }
}
