use arey::ux::{TerminalRenderer, get_theme};
use console::Term;
use std::thread;
use std::time::Duration;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Demonstrating streaming markdown highlighting...");
    println!("(Press Ctrl+C to exit)\n");
    thread::sleep(Duration::from_secs(1));

    let theme = get_theme("dark");
    let mut term = Term::stdout();
    let mut renderer = TerminalRenderer::new(&mut term, &theme);

    let markdown_content = r#"# Hello, `arey`!

This is a demonstration of a *stateful* markdown highlighter
for streaming terminal output.

It supports:
- **Bold text**
- *Italic text*
- `inline code`

```rust
fn main() {
    // This is a comment inside a code block.
    println!("Hello, World!");
}
```

And some final text to finish the stream."#;

    // Simulate a stream by processing the content character by character.
    for char in markdown_content.chars() {
        renderer.render_markdown(&char.to_string())?;
        thread::sleep(Duration::from_millis(10));
    }

    println!(); // Ensure the prompt starts on a newline.
    println!("\nStreaming finished.");
    Ok(())
}
