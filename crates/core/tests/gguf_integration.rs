use arey_core::{
    completion::{CancellationToken, ChatMessage, Completion, CompletionModel, SenderType},
    model::{ModelConfig, ModelProvider},
    provider::gguf::GgufBaseModel,
};
use futures::stream::StreamExt;
use once_cell::sync::Lazy;
use std::{
    collections::HashMap,
    env, fs,
    io::copy,
    path::{Path, PathBuf},
};
use tokio::sync::{Mutex, OnceCell};

const MODEL_URL: &str =
    "https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-UD-Q4_K_XL.gguf";
const MODEL_FILENAME: &str = "Qwen3-0.6B-UD-Q4_K_XL.gguf";

static DOWNLOAD_ONCE: OnceCell<()> = OnceCell::const_new();

fn get_test_data_dir() -> PathBuf {
    // If environment variable is set, use that
    if let Ok(dir) = env::var("AREY_TEST_DATA_DIR") {
        return PathBuf::from(dir);
    }

    // Otherwise, use target/arey-test-data in the project root (workspace)
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let mut path = PathBuf::from(manifest_dir);
    // Go up two levels to get workspace root (since we're in crates/core/tests)
    path.pop();
    path.pop();
    path.push("target");
    path.push("arey-test-data");
    path
}

async fn get_model_path() -> Result<PathBuf, Box<dyn std::error::Error>> {
    let data_dir = get_test_data_dir();
    fs::create_dir_all(&data_dir)?;
    let model_path = data_dir.join(MODEL_FILENAME);

    DOWNLOAD_ONCE
        .get_or_init(|| async {
            if !model_path.exists() {
                println!("Downloading test model from {}...", MODEL_URL);
                let response = reqwest::get(MODEL_URL)
                    .await
                    .expect("Failed to download test model");
                let mut dest = fs::File::create(&model_path)
                    .expect("Failed to create model file for download");
                let content = response
                    .bytes()
                    .await
                    .expect("Failed to read model bytes from response");
                copy(&mut content.as_ref(), &mut dest).expect("Failed to write model to file");
                println!("Model downloaded to {}", model_path.display());
            }
        })
        .await;

    Ok(model_path)
}

fn get_model_config(model_path: &Path, name: &str) -> ModelConfig {
    let mut settings = HashMap::new();
    settings.insert("path".to_string(), model_path.to_str().unwrap().into());
    settings.insert("n_gpu_layers".to_string(), 0.into());
    settings.insert("n_ctx".to_string(), 1024.into());

    ModelConfig {
        name: name.to_string(),
        provider: ModelProvider::Gguf,
        settings,
    }
}

#[tokio::test]
async fn test_gguf_model_complete() {
    let model_path = match get_model_path().await {
        Ok(path) => path,
        Err(e) => {
            panic!("Failed to download test model: {}", e);
        }
    };

    let model_config = get_model_config(&model_path, "test-gguf-complete");
    let mut model = GgufBaseModel::new(model_config).unwrap();

    let messages = vec![ChatMessage {
        sender: SenderType::User,
        text: "Once upon a time,".to_string(),
        tools: Vec::new(),
    }];

    let mut settings = HashMap::new();
    settings.insert("max_tokens".to_string(), "5".to_string());

    let mut stream = model
        .complete(&messages, None, &settings, CancellationToken::new())
        .await;

    let mut response_text = String::new();
    let mut finished = false;
    while let Some(result) = stream.next().await {
        match result.unwrap() {
            Completion::Response(resp) => {
                response_text.push_str(&resp.text);
                if resp.finish_reason.is_some() {
                    finished = true;
                }
            }
            Completion::Metrics(_) => {}
        }
    }

    assert!(finished, "Completion should have finished");
    assert!(
        !response_text.is_empty(),
        "Response text should not be empty"
    );
}

#[tokio::test]
async fn test_gguf_model_kv_cache() {
    let model_path = match get_model_path().await {
        Ok(path) => path,
        Err(e) => {
            panic!("Failed to download test model: {}", e);
        }
    };
    let model_config = get_model_config(&model_path, "test-gguf-cache");
    let mut model = GgufBaseModel::new(model_config).unwrap();

    let messages1 = vec![ChatMessage {
        sender: SenderType::User,
        text: "The first three letters of the alphabet are:".to_string(),
        tools: Vec::new(),
    }];

    let mut settings = HashMap::new();
    settings.insert("max_tokens".to_string(), "10".to_string());

    async fn get_prompt_latency(
        model: &mut GgufBaseModel,
        messages: &[ChatMessage],
        settings: &HashMap<String, String>,
    ) -> f32 {
        let mut stream = model
            .complete(messages, None, settings, CancellationToken::new())
            .await;
        while let Some(result) = stream.next().await {
            match result {
                Ok(Completion::Metrics(metrics)) => {
                    return metrics.prompt_eval_latency_ms;
                }
                Ok(_) => {}
                Err(e) => panic!("Stream returned an error: {}", e),
            }
        }
        0.0
    }

    let first_latency = get_prompt_latency(&mut model, &messages1, &settings).await;
    assert!(first_latency > 0.0, "First latency should be positive");

    // Second call with same prompt should be faster due to cache
    let second_latency = get_prompt_latency(&mut model, &messages1, &settings).await;
    assert!(second_latency > 0.0, "Second latency should be positive");
    assert!(
        second_latency < first_latency,
        "Second call with cache should be faster. first: {}, second: {}",
        first_latency,
        second_latency
    );

    // Third call with a longer prompt (append to previous)
    let messages2 = vec![ChatMessage {
        sender: SenderType::User,
        text: "The first three letters of the alphabet are: A, B, C. What comes next?".to_string(),
        tools: Vec::new(),
    }];
    let third_latency = get_prompt_latency(&mut model, &messages2, &settings).await;
    assert!(third_latency > 0.0, "Third latency should be positive");

    // Fourth call with a prompt that is a prefix of the previous one.
    // This tests that the cache is correctly handled when tokens are removed.
    let fourth_latency = get_prompt_latency(&mut model, &messages1, &settings).await;
    assert!(fourth_latency > 0.0, "Fourth latency should be positive");
    assert!(
        fourth_latency < third_latency,
        "Fourth call with prefix prompt should be faster than third. third: {}, fourth: {}",
        third_latency,
        fourth_latency
    );
}
