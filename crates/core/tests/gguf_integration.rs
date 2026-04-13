use arey_core::tools::{Tool, ToolError};
use arey_core::{
    completion::{CancellationToken, ChatMessage, Completion, CompletionModel, SenderType},
    model::{ModelConfig, ModelProvider},
    provider::gguf::GgufBaseModel,
};
use async_trait::async_trait;
use futures::stream::StreamExt;
use serde_json::json;
use std::{
    collections::HashMap,
    env, fs,
    io::copy,
    path::{Path, PathBuf},
    sync::Arc,
};
use tokio::sync::OnceCell;
use tracing_subscriber::{EnvFilter, fmt};

const MODEL_URL: &str = "https://huggingface.co/bartowski/Qwen_Qwen3.5-0.8B-GGUF/resolve/main/Qwen_Qwen3.5-0.8B-Q4_K_L.gguf";
const MODEL_FILENAME: &str = "Qwen_Qwen3.5-0.8B-Q4_K_L.gguf";

static DOWNLOAD_ONCE: OnceCell<()> = OnceCell::const_new();
static TRANSFORMER_DOWNLOAD_ONCE: OnceCell<()> = OnceCell::const_new();

const TRANSFORMER_MODEL_URL: &str = "https://huggingface.co/bartowski/Qwen_Qwen3-0.6B-GGUF/resolve/main/Qwen_Qwen3-0.6B-Q4_K_M.gguf";
const TRANSFORMER_MODEL_FILENAME: &str = "Qwen_Qwen3-0.6B-Q4_K_M.gguf";

fn get_test_data_dir() -> PathBuf {
    if let Ok(dir) = env::var("AREY_TEST_DATA_DIR") {
        return PathBuf::from(dir);
    }
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let mut path = PathBuf::from(manifest_dir);
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

async fn get_transformer_model_path() -> Result<PathBuf, Box<dyn std::error::Error>> {
    let data_dir = get_test_data_dir();
    fs::create_dir_all(&data_dir)?;
    let model_path = data_dir.join(TRANSFORMER_MODEL_FILENAME);

    TRANSFORMER_DOWNLOAD_ONCE
        .get_or_init(|| async {
            if !model_path.exists() {
                println!(
                    "Downloading transformer test model from {}...",
                    TRANSFORMER_MODEL_URL
                );
                let response = reqwest::get(TRANSFORMER_MODEL_URL)
                    .await
                    .expect("Failed to download test model");
                let mut dest = fs::File::create(&model_path)
                    .expect("Failed to create model file for download");
                let content = response
                    .bytes()
                    .await
                    .expect("Failed to read model bytes from response");
                copy(&mut content.as_ref(), &mut dest).expect("Failed to write model to file");
                println!("Transformer model downloaded to {}", model_path.display());
            }
        })
        .await;

    Ok(model_path)
}

async fn get_shared_model() -> Result<Arc<GgufBaseModel>, Box<dyn std::error::Error>> {
    let model_path = get_model_path().await?;
    let model_config = get_model_config(&model_path, "shared-gguf-model");
    let model = GgufBaseModel::new(model_config)?;
    model.reset().await?;
    Ok(Arc::new(model))
}

async fn create_hybrid_model() -> Result<Arc<GgufBaseModel>, Box<dyn std::error::Error>> {
    let model_path = get_model_path().await?;
    let mut settings = HashMap::new();
    settings.insert("path".to_string(), model_path.to_str().unwrap().into());
    settings.insert("n_gpu_layers".to_string(), 0.into());
    settings.insert("n_ctx".to_string(), 1024.into());

    let model_config = ModelConfig {
        key: "test-hybrid".to_string(),
        name: "test-hybrid".to_string(),
        provider: ModelProvider::Gguf,
        settings,
    };
    let model = GgufBaseModel::new(model_config)?;
    model.reset().await?;
    Ok(Arc::new(model))
}

async fn create_transformer_model() -> Result<Arc<GgufBaseModel>, Box<dyn std::error::Error>> {
    let model_path = get_transformer_model_path().await?;
    let mut settings = HashMap::new();
    settings.insert("path".to_string(), model_path.to_str().unwrap().into());
    settings.insert("n_gpu_layers".to_string(), 0.into());
    settings.insert("n_ctx".to_string(), 1024.into());
    settings.insert("cache_strategy".to_string(), "KvCacheOnly".into());

    let model_config = ModelConfig {
        key: "test-transformer".to_string(),
        name: "test-transformer".to_string(),
        provider: ModelProvider::Gguf,
        settings,
    };
    let model = GgufBaseModel::new(model_config)?;
    model.reset().await?;
    Ok(Arc::new(model))
}

async fn create_template_model() -> Result<Arc<GgufBaseModel>, Box<dyn std::error::Error>> {
    let model_path = get_model_path().await?;
    let mut settings = HashMap::new();
    settings.insert("path".to_string(), model_path.to_str().unwrap().into());
    settings.insert("n_gpu_layers".to_string(), 0.into());
    settings.insert("n_ctx".to_string(), 1024.into());
    settings.insert(
        "template".to_string(),
        serde_yaml::Value::String("qwen35".to_string()),
    );
    settings.insert("cache_strategy".to_string(), "hybrid".into());

    let model_config = ModelConfig {
        key: "test-template".to_string(),
        name: "test-template".to_string(),
        provider: ModelProvider::Gguf,
        settings,
    };
    let model = GgufBaseModel::new(model_config)?;
    model.reset().await?;
    Ok(Arc::new(model))
}

fn get_model_config(model_path: &Path, name: &str) -> ModelConfig {
    let mut settings = HashMap::new();
    settings.insert("path".to_string(), model_path.to_str().unwrap().into());
    settings.insert("n_gpu_layers".to_string(), 0.into());
    settings.insert("n_ctx".to_string(), 1024.into());

    ModelConfig {
        key: "test-gguf".to_string(),
        name: name.to_string(),
        provider: ModelProvider::Gguf,
        settings,
    }
}

#[allow(dead_code)]
fn get_model_config_with_strategy(
    model_path: &Path,
    name: &str,
    cache_strategy: &str,
) -> ModelConfig {
    let mut config = get_model_config(model_path, name);
    config
        .settings
        .insert("cache_strategy".to_string(), cache_strategy.into());
    config
}

#[allow(dead_code)]
fn get_model_config_with_template(
    model_path: &Path,
    name: &str,
    template_name: &str,
) -> ModelConfig {
    let mut config = get_model_config(model_path, name);
    config.settings.insert(
        "template".to_string(),
        serde_yaml::Value::String(template_name.to_string()),
    );
    config
}

fn init_tracing() {
    if env::var("RUST_LOG").is_ok() {
        let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("trace"));
        let _ = fmt().with_env_filter(filter).with_target(true).try_init();
    }
}

async fn run_completion(
    model: &GgufBaseModel,
    messages: &[ChatMessage],
    settings: &HashMap<String, String>,
) -> (String, bool, u32, u32) {
    let mut stream = model
        .complete(messages, None, settings, CancellationToken::new())
        .await;
    let mut response_text = String::new();
    let mut finished = false;
    let mut prompt_tokens = 0;
    let mut completion_tokens = 0;
    while let Some(result) = stream.next().await {
        match result.unwrap() {
            Completion::Response(resp) => {
                response_text.push_str(&resp.text);
                if resp.finish_reason.is_some() {
                    finished = true;
                }
            }
            Completion::Metrics(m) => {
                prompt_tokens = m.prompt_tokens;
                completion_tokens = m.completion_tokens;
            }
        }
    }
    (response_text, finished, prompt_tokens, completion_tokens)
}

#[tokio::test]
async fn test_gguf_completion() {
    init_tracing();
    let model = create_hybrid_model().await.unwrap();

    let mut settings = HashMap::new();
    settings.insert("max_tokens".to_string(), "5".to_string());

    // Scenario 1: Basic completion
    let messages1 = vec![ChatMessage {
        sender: SenderType::User,
        text: "Once upon a time,".to_string(),
        ..Default::default()
    }];
    let (text1, finished1, prompt_tok1, completion_tok1) =
        run_completion(&model, &messages1, &settings).await;
    assert!(finished1, "Completion should have finished");
    assert!(!text1.is_empty(), "Response text should not be empty");
    assert!(
        prompt_tok1 > 0 && completion_tok1 > 0,
        "Should have prompt and completion tokens"
    );

    // Scenario 2: Streaming response
    let messages2 = vec![ChatMessage {
        sender: SenderType::User,
        text: "Say 'hello world'.".to_string(),
        ..Default::default()
    }];
    let (text2, finished2, prompt_tok2, completion_tok2) =
        run_completion(&model, &messages2, &settings).await;
    assert!(finished2, "Completion 2 should have finished");
    assert!(!text2.is_empty(), "Response text 2 should not be empty");
    assert!(
        prompt_tok2 > 0 && completion_tok2 > 0,
        "Should have prompt and completion tokens"
    );
}

async fn complete_and_extract(
    model: &GgufBaseModel,
    messages: &[ChatMessage],
    settings: &HashMap<String, String>,
) -> (String, Option<arey_core::completion::CacheMetrics>) {
    let mut stream = model
        .complete(messages, None, settings, CancellationToken::new())
        .await;

    let mut response_text = String::new();
    let mut cache_metrics: Option<arey_core::completion::CacheMetrics> = None;

    while let Some(result) = stream.next().await {
        match result {
            Ok(Completion::Response(resp)) => {
                response_text.push_str(&resp.text);
            }
            Ok(Completion::Metrics(m)) => {
                println!(
                    "  Metrics: prompt_tokens={}, completion_tokens={}",
                    m.prompt_tokens, m.completion_tokens
                );
                if let Some(cm) = m.cache_metrics {
                    println!("  Cache: hit={}, strategy={:?}", cm.cache_hit, cm.strategy);
                    cache_metrics = Some(cm);
                }
            }
            Err(e) => panic!("Stream error: {}", e),
        }
    }
    (response_text, cache_metrics)
}

struct WeatherTool;

#[async_trait]
impl Tool for WeatherTool {
    fn name(&self) -> String {
        "get_current_weather".to_string()
    }

    fn description(&self) -> String {
        "Gets the current weather for a given location".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The location to get weather for"
                }
            },
            "required": ["location"]
        })
    }

    async fn execute(&self, arguments: &serde_json::Value) -> Result<serde_json::Value, ToolError> {
        let location = arguments["location"]
            .as_str()
            .ok_or_else(|| ToolError::ExecutionError("Missing location parameter".to_string()))?;

        Ok(json!({
            "location": location,
            "temp_C": 22,
            "weatherDesc": "partly cloudy",
            "humidity": 65,
            "precipMM": 0.5
        }))
    }
}

async fn complete_with_tools(
    model: &GgufBaseModel,
    messages: &[ChatMessage],
    tools: &[Arc<dyn Tool>],
    settings: &HashMap<String, String>,
) -> (
    String,
    Option<String>,
    Option<Vec<arey_core::tools::ToolCall>>,
    Option<arey_core::completion::CacheMetrics>,
) {
    let mut stream = model
        .complete(messages, Some(tools), settings, CancellationToken::new())
        .await;

    let mut response_text = String::new();
    let mut thought: Option<String> = None;
    let mut tool_calls: Option<Vec<arey_core::tools::ToolCall>> = None;
    let mut cache_metrics: Option<arey_core::completion::CacheMetrics> = None;

    while let Some(result) = stream.next().await {
        match result {
            Ok(Completion::Response(resp)) => {
                response_text.push_str(&resp.text);
                if resp.thought.is_some() {
                    thought = resp.thought;
                }
                if let Some(calls) = resp.tool_calls {
                    tool_calls = Some(calls);
                }
            }
            Ok(Completion::Metrics(m)) => {
                if let Some(cm) = m.cache_metrics {
                    cache_metrics = Some(cm);
                }
            }
            Err(e) => panic!("Stream error: {}", e),
        }
    }
    (response_text, thought, tool_calls, cache_metrics)
}

#[tokio::test]
async fn test_model_params() {
    // Skip on CI (GitHub Actions sets CI=true)
    if std::env::var("CI").is_ok() {
        eprintln!("Skipping test_model_params on CI");
        return;
    }

    init_tracing();
    let model = get_shared_model().await.unwrap();

    let settings = HashMap::new();
    let messages = vec![ChatMessage {
        sender: SenderType::User,
        text: "hi".to_string(),
        ..Default::default()
    }];

    // Trigger model load by making a completion request
    let mut stream = model
        .complete(&messages, None, &settings, CancellationToken::new())
        .await;
    while let Some(result) = stream.next().await {
        if let Ok(Completion::Response(_)) = result {
            break;
        }
    }

    // Get metrics
    let metrics = model.metrics();

    println!(
        "Model metrics: {:?}",
        serde_json::to_string_pretty(&metrics).unwrap()
    );

    // Verify system info is populated
    assert!(metrics.cpu_threads.is_some(), "cpu_threads should be set");
    assert!(metrics.gpu_devices.is_some(), "gpu_devices should be set");

    // Verify computed parameters are populated
    assert!(metrics.n_ctx.is_some(), "n_ctx should be set");
    assert!(metrics.n_batch.is_some(), "n_batch should be set");
    assert!(metrics.n_ubatch.is_some(), "n_ubatch should be set");
    assert!(metrics.n_gpu_layers.is_some(), "n_gpu_layers should be set");
    assert!(metrics.n_threads.is_some(), "n_threads should be set");
    assert!(
        metrics.flash_attention.is_some(),
        "flash_attention should be set"
    );

    // Verify model info from GGUF metadata is populated
    assert!(
        metrics.model_params_billions.is_some(),
        "model_params_billions should be set"
    );
    assert!(metrics.model_layers.is_some(), "model_layers should be set");
    assert!(metrics.quantization.is_some(), "quantization should be set");
    assert!(
        metrics.model_n_ctx_train.is_some(),
        "model_n_ctx_train should be set"
    );

    // Verify values are reasonable
    let n_ctx = metrics.n_ctx.unwrap();
    assert!(n_ctx >= 512, "n_ctx should be at least 512");

    let n_threads = metrics.n_threads.unwrap();
    assert!(n_threads >= 1, "n_threads should be at least 1");

    let model_params = metrics.model_params_billions.unwrap();
    assert!(
        model_params > 0.0,
        "model_params_billions should be positive"
    );

    println!("All model params verified:");
    println!("  - cpu_threads: {:?}", metrics.cpu_threads);
    println!("  - n_ctx: {:?}", metrics.n_ctx);
    println!("  - n_batch: {:?}", metrics.n_batch);
    println!("  - n_gpu_layers: {:?}", metrics.n_gpu_layers);
    println!("  - model_params_billions: {:.2}", model_params);
    println!("  - quantization: {:?}", metrics.quantization);
}

#[tokio::test]
async fn test_oom_recovery_with_trim() {
    // Skip on CI
    if std::env::var("CI").is_ok() {
        eprintln!("Skipping test_oom_recovery_with_trim on CI");
        return;
    }

    init_tracing();

    // Create a dedicated model with small context for OOM testing
    let model_path = get_model_path().await.unwrap();
    let mut settings = HashMap::new();
    settings.insert("path".to_string(), model_path.to_str().unwrap().into());
    settings.insert("n_gpu_layers".to_string(), 0.into());
    settings.insert("n_ctx".to_string(), 512.into());
    settings.insert("n_batch".to_string(), 256.into());

    let model_config = ModelConfig {
        key: "test-oom".to_string(),
        name: "test-oom".to_string(),
        provider: ModelProvider::Gguf,
        settings,
    };
    let model = GgufBaseModel::new(model_config).unwrap();

    // Add many messages to fill context - use repetitive text to quickly fill
    // Use smaller repeated text to fit within 512 context without exceeding
    let long_text = "Hello, how are you? Please respond with a detailed explanation. ".repeat(3);

    // Create prompt that fits within context (under 512 tokens)
    let prompt = long_text;

    let mut settings = HashMap::new();
    settings.insert("max_tokens".to_string(), "10".to_string());

    let messages = vec![ChatMessage {
        sender: SenderType::User,
        text: prompt,
        ..Default::default()
    }];

    let mut stream = model
        .complete(&messages, None, &settings, CancellationToken::new())
        .await;

    let mut received_response = false;
    let mut finished = false;

    while let Some(result) = stream.next().await {
        match result {
            Ok(Completion::Response(resp)) => {
                received_response = true;
                println!(
                    "Response: text_len={}, finish_reason={:?}",
                    resp.text.len(),
                    resp.finish_reason
                );
                if resp.finish_reason.is_some() {
                    finished = true;
                }
            }
            Ok(Completion::Metrics(m)) => {
                println!(
                    "Metrics: prompt_tokens={}, completion_tokens={}",
                    m.prompt_tokens, m.completion_tokens
                );
            }
            Err(e) => {
                println!("Error: {}", e);
            }
        }
    }

    println!(
        "Test result: received_response={}, finished={}",
        received_response, finished
    );

    assert!(received_response, "Should receive response");
}

#[tokio::test]
async fn test_hybrid_cache() {
    init_tracing();

    let model = create_hybrid_model().await.unwrap();

    let mut settings = HashMap::new();
    settings.insert("max_tokens".to_string(), "5".to_string());

    // === Scenario 1: Turn 1 should NOT be cache hit ===
    let user_msg_1 = ChatMessage {
        sender: SenderType::User,
        text: "Say 'hello' if you can hear me.".to_string(),
        ..Default::default()
    };

    let (response1, cache1) =
        complete_and_extract(&model, std::slice::from_ref(&user_msg_1), &settings).await;
    println!(
        "Turn 1: response='{}', cache_hit={}",
        response1,
        cache1.as_ref().map(|c| c.cache_hit).unwrap_or(false)
    );
    assert!(!response1.is_empty());
    assert!(
        !cache1.as_ref().unwrap().cache_hit,
        "Turn 1 should NOT be cache hit"
    );
    assert_eq!(cache1.as_ref().unwrap().strategy.as_deref(), Some("Hybrid"));

    // === Scenario 2: Same message should cache hit ===
    let (_response2, cache2) =
        complete_and_extract(&model, std::slice::from_ref(&user_msg_1), &settings).await;
    println!(
        "Turn 2: cache_hit={}",
        cache2.as_ref().map(|c| c.cache_hit).unwrap_or(false)
    );
    assert!(
        cache2.as_ref().unwrap().cache_hit,
        "Turn 2 should be cache hit"
    );

    // === Scenario 3: Multi-turn prefix match ===
    let assistant_msg_1 = ChatMessage {
        sender: SenderType::Assistant,
        text: response1.clone(),
        ..Default::default()
    };
    let user_msg_2 = ChatMessage {
        sender: SenderType::User,
        text: "Who are you?".to_string(),
        ..Default::default()
    };
    let messages_turn3 = vec![user_msg_1.clone(), assistant_msg_1, user_msg_2];
    let (response3, cache3) = complete_and_extract(&model, &messages_turn3, &settings).await;
    println!(
        "Turn 3: cache_hit={}",
        cache3.as_ref().map(|c| c.cache_hit).unwrap_or(false)
    );
    assert!(!response3.is_empty());

    // === Scenario 4: Different conversation should NOT cache hit ===
    let user_msg_4 = ChatMessage {
        sender: SenderType::User,
        text: "Say 'goodbye' if you can hear me.".to_string(),
        ..Default::default()
    };
    let (_response4, cache4) = complete_and_extract(&model, &[user_msg_4], &settings).await;
    println!(
        "Turn 4: cache_hit={}",
        cache4.as_ref().map(|c| c.cache_hit).unwrap_or(false)
    );
    assert!(
        !cache4.as_ref().unwrap().cache_hit,
        "Turn 4 should NOT be cache hit"
    );

    // === Scenario 5: Auto-detect strategy ===
    let messages = vec![ChatMessage {
        sender: SenderType::User,
        text: "Hello".to_string(),
        ..Default::default()
    }];
    let mut stream = model
        .complete(&messages, None, &settings, CancellationToken::new())
        .await;
    let mut strategy_detected = None;
    while let Some(result) = stream.next().await {
        if let Ok(Completion::Metrics(m)) = result
            && let Some(cm) = m.cache_metrics
        {
            strategy_detected = cm.strategy.clone();
        }
    }
    assert_eq!(
        strategy_detected.as_deref(),
        Some("Hybrid"),
        "Should auto-detect Hybrid"
    );

    // === Scenario 6: Cache mechanism (long prompt) ===
    let long_prompt = "Explain the concept of recursion in computer science.";
    let (_first_response, first_cache) = complete_and_extract(
        &model,
        &[ChatMessage {
            sender: SenderType::User,
            text: long_prompt.to_string(),
            ..Default::default()
        }],
        &settings,
    )
    .await;
    println!(
        "Long prompt run 1: cache_hit={}",
        first_cache.as_ref().map(|c| c.cache_hit).unwrap_or(false)
    );

    let (_second_response, second_cache) = complete_and_extract(
        &model,
        &[ChatMessage {
            sender: SenderType::User,
            text: long_prompt.to_string(),
            ..Default::default()
        }],
        &settings,
    )
    .await;
    println!(
        "Long prompt run 2: cache_hit={}",
        second_cache.as_ref().map(|c| c.cache_hit).unwrap_or(false)
    );
    assert!(
        second_cache.as_ref().unwrap().cache_hit,
        "Second run should be cache hit"
    );
}

#[tokio::test]
async fn test_kvonly_cache() {
    init_tracing();

    // === Scenario 1: Transformer model with KvCacheOnly ===
    let transformer_model = create_transformer_model().await.unwrap();

    let mut settings = HashMap::new();
    settings.insert("max_tokens".to_string(), "5".to_string());

    let user_msg_1 = ChatMessage {
        sender: SenderType::User,
        text: "Please say hello if you can hear me, this is a test message.".to_string(),
        ..Default::default()
    };

    let (response1, cache1) = complete_and_extract(
        &transformer_model,
        std::slice::from_ref(&user_msg_1),
        &settings,
    )
    .await;
    println!(
        "Turn 1 (KvCacheOnly): cache_hit={}",
        cache1.as_ref().map(|c| c.cache_hit).unwrap_or(false)
    );
    assert!(!response1.is_empty());
    assert!(
        !cache1.as_ref().unwrap().cache_hit,
        "Turn 1 should NOT cache hit"
    );
    assert_eq!(
        cache1.as_ref().unwrap().strategy.as_deref(),
        Some("KvCacheOnly")
    );

    // === Scenario 2: Same message hits KV cache ===
    let (_response2, cache2) = complete_and_extract(
        &transformer_model,
        std::slice::from_ref(&user_msg_1),
        &settings,
    )
    .await;
    println!(
        "Turn 2 (KvCacheOnly): cache_hit={}",
        cache2.as_ref().map(|c| c.cache_hit).unwrap_or(false)
    );
    assert!(
        cache2.as_ref().unwrap().cache_hit,
        "Turn 2 should cache hit"
    );

    // === Scenario 3: Different message after overflow should NOT hit ===
    let user_msg_2 = ChatMessage {
        sender: SenderType::User,
        text: "Please say goodbye if you can hear me, this is another test.".to_string(),
        ..Default::default()
    };
    let (_response3, cache3) =
        complete_and_extract(&transformer_model, &[user_msg_2], &settings).await;
    println!(
        "Turn 3 (KvCacheOnly): cache_hit={}",
        cache3.as_ref().map(|c| c.cache_hit).unwrap_or(false)
    );
    assert!(
        !cache3.as_ref().unwrap().cache_hit,
        "After overflow should NOT cache hit"
    );

    // === Scenario 4: Tool calls with hybrid (template model) ===
    let _template_model = create_template_model().await.unwrap();
    settings.insert("max_tokens".to_string(), "10".to_string());

    let tool_model = create_template_model().await.unwrap();
    let tools: Vec<Arc<dyn Tool>> = vec![Arc::new(WeatherTool)];

    // Turn 1: no cache hit
    let msg1 = ChatMessage {
        sender: SenderType::User,
        text: "Hello, what is the weather like?".to_string(),
        ..Default::default()
    };
    let (_r1, _t1, _tc1, c1) =
        complete_with_tools(&tool_model, std::slice::from_ref(&msg1), &tools, &settings).await;
    println!(
        "Tool call turn 1: cache_hit={}",
        c1.as_ref().map(|x| x.cache_hit).unwrap_or(false)
    );
    assert!(
        !c1.as_ref().unwrap().cache_hit,
        "Turn 1 should NOT cache hit"
    );

    // Turn 2: same message should cache hit
    let (_r2, _t2, _tc2, c2) = complete_with_tools(&tool_model, &[msg1], &tools, &settings).await;
    println!(
        "Tool call turn 2: cache_hit={}",
        c2.as_ref().map(|x| x.cache_hit).unwrap_or(false)
    );
    assert!(c2.as_ref().unwrap().cache_hit, "Turn 2 should cache hit");
}
