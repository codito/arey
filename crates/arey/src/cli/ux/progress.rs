use indicatif::{ProgressBar, ProgressStyle};

/// A spinner to indicate that a response is being generated.
#[derive(Debug)]
pub struct GenerationSpinner {
    spinner: ProgressBar,
}

impl GenerationSpinner {
    /// Creates a new `GenerationSpinner` with a message.
    pub fn new(msg: String) -> Self {
        let spinner = ProgressBar::new_spinner();
        spinner.set_style(
            ProgressStyle::with_template("{spinner:.blue} {msg}")
                .unwrap()
                .tick_strings(&["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]),
        );
        spinner.set_message(msg);
        spinner.enable_steady_tick(std::time::Duration::from_millis(100));

        Self { spinner }
    }

    /// Stops the spinner and clears it from the terminal.
    pub fn clear(&self) {
        self.spinner.finish_and_clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generation_spinner_new() {
        // This is a smoke test to ensure the spinner can be created.
        // Testing terminal output is complex and out of scope for unit tests.
        let spinner = GenerationSpinner::new("Testing...".to_string());
        assert_eq!(spinner.spinner.message(), "Testing...");
        spinner.clear();
    }
}
