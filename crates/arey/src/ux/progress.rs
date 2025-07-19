use indicatif::{ProgressBar, ProgressStyle};

#[derive(Debug)]
pub struct GenerationSpinner {
    spinner: ProgressBar,
}

impl GenerationSpinner {
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

    pub fn clear(&self) {
        self.spinner.finish_and_clear();
    }
}
