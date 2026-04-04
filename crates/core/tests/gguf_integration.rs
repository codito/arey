use arey_core::tools::{Tool, ToolError, ToolResult};
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

#[tokio::test]
async fn test_gguf_model_complete() {
    init_tracing();

    let model_path = match get_model_path().await {
        Ok(path) => path,
        Err(e) => {
            panic!("Failed to download test model: {}", e);
        }
    };

    let model_config = get_model_config(&model_path, "test-gguf-complete");
    let model = GgufBaseModel::new(model_config).unwrap();

    let messages = vec![ChatMessage {
        sender: SenderType::User,
        text: "Once upon a time,".to_string(),
        ..Default::default()
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
async fn test_gguf_streaming_response() {
    init_tracing();

    let model_path = match get_model_path().await {
        Ok(path) => path,
        Err(e) => {
            panic!("Failed to download test model: {}", e);
        }
    };

    let model_config = get_model_config(&model_path, "test-gguf-streaming");
    let model = GgufBaseModel::new(model_config).unwrap();

    let messages = vec![ChatMessage {
        sender: SenderType::User,
        text: "Say 'hello world'.".to_string(),
        ..Default::default()
    }];

    let mut settings = HashMap::new();
    settings.insert("max_tokens".to_string(), "10".to_string());

    let mut stream = model
        .complete(&messages, None, &settings, CancellationToken::new())
        .await;

    let mut chunks_received = 0;
    let mut final_chunk_received = false;

    while let Some(result) = stream.next().await {
        match result.unwrap() {
            Completion::Response(resp) => {
                chunks_received += 1;
                if resp.finish_reason.is_some() {
                    final_chunk_received = true;
                }
            }
            Completion::Metrics(m) => {
                println!(
                    "Metrics: prompt_tokens={}, completion_tokens={}",
                    m.prompt_tokens, m.completion_tokens
                );
                assert!(m.prompt_tokens > 0, "Should have prompt tokens");
                assert!(m.completion_tokens > 0, "Should have completion tokens");
            }
        }
    }

    assert!(chunks_received > 0, "Should receive response chunks");
    assert!(
        final_chunk_received,
        "Should receive final chunk with finish_reason"
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

#[tokio::test]
async fn test_gguf_hybrid_cache_prefix_match() {
    init_tracing();

    let model_path = match get_model_path().await {
        Ok(path) => path,
        Err(e) => {
            panic!("Failed to download test model: {}", e);
        }
    };

    let model_config =
        get_model_config_with_strategy(&model_path, "test-gguf-hybrid-cache", "hybrid");
    let model = GgufBaseModel::new(model_config).unwrap();

    let mut settings = HashMap::new();
    settings.insert("max_tokens".to_string(), "10".to_string());

    // Turn 1: First user message (no cache hit expected)
    let user_msg_1 = ChatMessage {
        sender: SenderType::User,
        text: "Say 'hello' if you can hear me.".to_string(),
        ..Default::default()
    };
    let messages_turn1 = vec![user_msg_1.clone()];

    let (response1, cache1) = complete_and_extract(&model, &messages_turn1, &settings).await;
    println!("Turn 1 response: {}", response1);
    assert!(!response1.is_empty(), "Turn 1 response should not be empty");
    assert!(cache1.is_some(), "Cache metrics should be present");
    assert!(
        !cache1.as_ref().unwrap().cache_hit,
        "Turn 1 should NOT be a cache hit"
    );
    assert_eq!(
        cache1.as_ref().unwrap().strategy.as_deref(),
        Some("Hybrid"),
        "Strategy should be Hybrid"
    );

    // Turn 2: Same message (cache hit expected for Hybrid prefix match)
    let messages_turn2 = vec![user_msg_1.clone()];

    let (response2, cache2) = complete_and_extract(&model, &messages_turn2, &settings).await;
    println!("Turn 2 response: {}", response2);
    assert!(!response2.is_empty(), "Turn 2 response should not be empty");
    assert!(cache2.is_some(), "Cache metrics should be present");
    assert!(
        cache2.as_ref().unwrap().cache_hit,
        "Turn 2 should be a cache hit"
    );

    // Turn 3: Multi-turn conversation (prefix match)
    let assistant_msg_1 = ChatMessage {
        sender: SenderType::Assistant,
        text: response1,
        ..Default::default()
    };
    let user_msg_2 = ChatMessage {
        sender: SenderType::User,
        text: "Who are you?".to_string(),
        ..Default::default()
    };
    let messages_turn3 = vec![user_msg_1.clone(), assistant_msg_1, user_msg_2];

    let (response3, _cache3) = complete_and_extract(&model, &messages_turn3, &settings).await;
    println!("Turn 3 response: {}", response3);
    assert!(!response3.is_empty(), "Turn 3 response should not be empty");

    // Turn 4: Different conversation (should NOT hit cache)
    let different_user_msg = ChatMessage {
        sender: SenderType::User,
        text: "Say 'goodbye' if you can hear me.".to_string(),
        ..Default::default()
    };
    let messages_turn4 = vec![different_user_msg];

    let (response4, cache4) = complete_and_extract(&model, &messages_turn4, &settings).await;
    println!("Turn 4 response: {}", response4);
    assert!(!response4.is_empty(), "Turn 4 response should not be empty");
    assert!(cache4.is_some(), "Cache metrics should be present");
    assert!(
        !cache4.as_ref().unwrap().cache_hit,
        "Turn 4 should NOT be a cache hit"
    );
}

#[tokio::test]
async fn test_gguf_hybrid_cache_with_output_fields() {
    init_tracing();

    let model_path = match get_model_path().await {
        Ok(path) => path,
        Err(e) => {
            panic!("Failed to download test model: {}", e);
        }
    };

    let model_config =
        get_model_config_with_strategy(&model_path, "test-gguf-hybrid-cache-fields", "hybrid");
    let model = GgufBaseModel::new(model_config).unwrap();

    let mut settings = HashMap::new();
    settings.insert("max_tokens".to_string(), "10".to_string());

    // Turn 1: First user message
    let user_msg_1 = ChatMessage {
        sender: SenderType::User,
        text: "Say 'hello' if you can hear me.".to_string(),
        ..Default::default()
    };
    let messages_turn1 = vec![user_msg_1.clone()];

    let (response1, _) = complete_and_extract(&model, &messages_turn1, &settings).await;
    println!("Turn 1 response: {}", response1);
    assert!(!response1.is_empty(), "Turn 1 response should not be empty");

    // Turn 2: Multi-turn with assistant message containing tools/metrics
    let assistant_msg_with_output = ChatMessage {
        sender: SenderType::Assistant,
        text: response1,
        tools: Some(vec![]),
        metrics: Some(arey_core::completion::CompletionMetrics {
            prompt_tokens: 22,
            prompt_eval_latency_ms: 100.0,
            completion_tokens: 10,
            completion_latency_ms: 500.0,
            thought: None,
            raw_chunk: None,
            cache_metrics: None,
        }),
        ..Default::default()
    };
    let user_msg_2 = ChatMessage {
        sender: SenderType::User,
        text: "Who are you?".to_string(),
        ..Default::default()
    };
    let messages_turn2 = vec![user_msg_1.clone(), assistant_msg_with_output, user_msg_2];

    let (response2, _) = complete_and_extract(&model, &messages_turn2, &settings).await;
    println!("Turn 2 response: {}", response2);
    assert!(!response2.is_empty(), "Turn 2 response should not be empty");
}

#[tokio::test]
async fn test_gguf_auto_detect_cache_strategy() {
    init_tracing();

    let model_path = match get_model_path().await {
        Ok(path) => path,
        Err(e) => {
            panic!("Failed to download test model: {}", e);
        }
    };

    // Don't set cache_strategy - should auto-detect based on model type
    let model_config = get_model_config(&model_path, "test-gguf-auto-detect");
    let model = GgufBaseModel::new(model_config).unwrap();

    let mut settings = HashMap::new();
    settings.insert("max_tokens".to_string(), "5".to_string());

    let messages = vec![ChatMessage {
        sender: SenderType::User,
        text: "Hello".to_string(),
        ..Default::default()
    }];

    let mut stream = model
        .complete(&messages, None, &settings, CancellationToken::new())
        .await;

    let mut completed = false;
    while let Some(result) = stream.next().await {
        match result {
            Ok(Completion::Response(_)) => {}
            Ok(Completion::Metrics(m)) => {
                println!(
                    "  Cache metrics: strategy={:?}",
                    m.cache_metrics.as_ref().map(|c| &c.strategy)
                );
                assert!(m.cache_metrics.is_some(), "Cache metrics should be present");
                let cache = m.cache_metrics.unwrap();
                assert!(cache.strategy.is_some(), "Strategy should be detected");
                let strategy = cache.strategy.unwrap();
                assert_eq!(
                    strategy, "Hybrid",
                    "Qwen3.5 should auto-detect as Hybrid strategy"
                );
                completed = true;
            }
            Err(e) => panic!("Stream error: {}", e),
        }
    }
    assert!(completed, "Should have received metrics");
}

#[tokio::test]
async fn test_gguf_cache_mechanism() {
    init_tracing();

    let model_path = match get_model_path().await {
        Ok(path) => path,
        Err(e) => {
            panic!("Failed to download test model: {}", e);
        }
    };

    let model_config = get_model_config_with_strategy(&model_path, "test-gguf-cache-perf", "auto");
    let model = GgufBaseModel::new(model_config).unwrap();

    let mut settings = HashMap::new();
    settings.insert("max_tokens".to_string(), "20".to_string());

    let long_prompt = "Explain the concept of recursion in computer science.";

    let user_msg = ChatMessage {
        sender: SenderType::User,
        text: long_prompt.to_string(),
        ..Default::default()
    };
    let messages = vec![user_msg.clone()];

    // First run: Create the cache snapshot
    let (first_response, first_metrics) = complete_and_extract(&model, &messages, &settings).await;

    assert!(
        !first_response.is_empty(),
        "First response should not be empty"
    );
    assert!(first_metrics.is_some(), "First metrics should be present");
    let first_cache = first_metrics.as_ref().unwrap();
    println!(
        "First run: prompt_tokens={}, cache_hit={}",
        first_cache.tokens_skipped.unwrap_or(0),
        first_cache.cache_hit
    );

    // Second run: Should hit the cache
    let (second_response, second_metrics) =
        complete_and_extract(&model, &messages, &settings).await;

    assert!(
        !second_response.is_empty(),
        "Second response should not be empty"
    );
    assert!(second_metrics.is_some(), "Second metrics should be present");

    let second_cache = second_metrics.as_ref().unwrap();
    println!("Second run: cache_hit={}", second_cache.cache_hit);

    // Verify cache hit occurred
    assert!(second_cache.cache_hit, "Second run should be a cache hit");
}

const TRANSFORMER_MODEL_URL: &str = "https://huggingface.co/bartowski/Qwen_Qwen3-0.6B-GGUF/resolve/main/Qwen_Qwen3-0.6B-Q4_K_M.gguf";
const TRANSFORMER_MODEL_FILENAME: &str = "Qwen_Qwen3-0.6B-Q4_K_M.gguf";

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

#[tokio::test]
async fn test_gguf_transformer_kv_cache() {
    init_tracing();

    let model_path = match get_transformer_model_path().await {
        Ok(path) => path,
        Err(e) => {
            panic!("Failed to download test model: {}", e);
        }
    };

    // Use KvCacheOnly strategy for regular transformer model
    let model_config =
        get_model_config_with_strategy(&model_path, "test-gguf-transformer-cache", "KvCacheOnly");
    let model = GgufBaseModel::new(model_config).unwrap();

    let mut settings = HashMap::new();
    settings.insert("max_tokens".to_string(), "10".to_string());

    // First request - use longer message (>8 tokens) to exceed cache threshold
    let user_msg_1 = ChatMessage {
        sender: SenderType::User,
        text: "Please say hello if you can hear me, this is a test message.".to_string(),
        ..Default::default()
    };
    let messages_turn1 = vec![user_msg_1.clone()];

    let (response1, cache1) = complete_and_extract(&model, &messages_turn1, &settings).await;
    println!("Turn 1 response: {}", response1);
    assert!(!response1.is_empty(), "Turn 1 response should not be empty");
    assert!(cache1.is_some(), "Cache metrics should be present");
    assert!(
        !cache1.as_ref().unwrap().cache_hit,
        "Turn 1 should NOT be a cache hit"
    );
    assert_eq!(
        cache1.as_ref().unwrap().strategy.as_deref(),
        Some("KvCacheOnly"),
        "Strategy should be KvCacheOnly"
    );

    // Second request with same message - should hit KV cache
    let messages_turn2 = vec![user_msg_1.clone()];

    let (response2, cache2) = complete_and_extract(&model, &messages_turn2, &settings).await;
    println!("Turn 2 response: {}", response2);
    assert!(!response2.is_empty(), "Turn 2 response should not be empty");
    assert!(cache2.is_some(), "Cache metrics should be present");
    assert!(
        cache2.as_ref().unwrap().cache_hit,
        "Turn 2 should be a cache hit (KvCacheOnly)"
    );

    // Verify tokens were skipped
    let tokens_skipped = cache2.as_ref().unwrap().tokens_skipped;
    assert!(
        tokens_skipped.is_some(),
        "Should have tokens_skipped for KvCacheOnly"
    );
    println!("Turn 2 tokens_skipped: {:?}", tokens_skipped);

    // Third request with different message - after overflow, should NOT hit cache
    let user_msg_2 = ChatMessage {
        sender: SenderType::User,
        text: "Please say goodbye if you can hear me, this is another test.".to_string(),
        ..Default::default()
    };
    let messages_turn3 = vec![user_msg_2];

    let (response3, cache3) = complete_and_extract(&model, &messages_turn3, &settings).await;
    println!("Turn 3 response: {}", response3);
    assert!(!response3.is_empty(), "Turn 3 response should not be empty");
    assert!(cache3.is_some(), "Cache metrics should be present");
    // After overflow, should be no cache hit
    assert!(
        !cache3.as_ref().unwrap().cache_hit,
        "Turn 3 should NOT be a cache hit after overflow"
    );

    // Fourth request with completely different message - should NOT hit cache
    let user_msg_3 = ChatMessage {
        sender: SenderType::User,
        text: "What is the capital of France? Please answer.".to_string(),
        ..Default::default()
    };
    let messages_turn4 = vec![user_msg_3];

    let (response4, cache4) = complete_and_extract(&model, &messages_turn4, &settings).await;
    println!("Turn 4 response: {}", response4);
    assert!(!response4.is_empty(), "Turn 4 response should not be empty");
    assert!(cache4.is_some(), "Cache metrics should be present");
    // After overflow, should be no cache hit
    println!("Turn 4 cache_hit={}", cache4.as_ref().unwrap().cache_hit);
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
async fn test_gguf_hybrid_cache_with_tool_calls() {
    init_tracing();

    let model_path = match get_model_path().await {
        Ok(path) => path,
        Err(e) => {
            panic!("Failed to download test model: {}", e);
        }
    };

    let mut model_config =
        get_model_config_with_template(&model_path, "test-gguf-hybrid-cache-tools", "qwen35");
    model_config
        .settings
        .insert("cache_strategy".to_string(), "hybrid".into());
    let model = GgufBaseModel::new(model_config).unwrap();

    let mut settings = HashMap::new();
    settings.insert("max_tokens".to_string(), "20".to_string());

    let tools: Vec<Arc<dyn Tool>> = vec![Arc::new(WeatherTool)];
    let mut messages: Vec<ChatMessage> = Vec::new();

    // Turn 1: First user message (no cache hit expected)
    let user_msg_1 = ChatMessage {
        sender: SenderType::User,
        text: "Hello, what is the weather like?".to_string(),
        ..Default::default()
    };
    messages.clear();
    messages.push(user_msg_1.clone());

    let (response1, thought1, _tool_calls1, cache1) =
        complete_with_tools(&model, &messages, &tools, &settings).await;
    println!("Turn 1 response: {}", response1);
    println!("Turn 1 thought: {:?}", thought1);
    assert!(!response1.is_empty(), "Turn 1 response should not be empty");
    assert!(cache1.is_some(), "Cache metrics should be present");
    assert!(
        !cache1.as_ref().unwrap().cache_hit,
        "Turn 1 should NOT be a cache hit"
    );

    // Turn 2: Same message (cache hit expected for Hybrid prefix match)
    messages.clear();
    messages.push(user_msg_1.clone());

    let (response2, _thought2, _tool_calls2, cache2) =
        complete_with_tools(&model, &messages, &tools, &settings).await;
    println!("Turn 2 response: {}", response2);
    println!(
        "Turn 2 cache_hit={}, transition={:?}",
        cache2.as_ref().map(|c| c.cache_hit).unwrap_or(false),
        cache2
            .as_ref()
            .and_then(|c| c.checkpoint_transition.clone())
    );
    assert!(!response2.is_empty(), "Turn 2 response should not be empty");
    assert!(cache2.is_some(), "Cache metrics should be present");
    assert!(
        cache2.as_ref().unwrap().cache_hit,
        "Turn 2 should be a cache hit"
    );

    // Turn 3: Multi-turn conversation (cache hit via TurnEnd checkpoint)
    let assistant_msg_1 = ChatMessage {
        sender: SenderType::Assistant,
        text: response1.clone(),
        thought: thought1.clone(),
        ..Default::default()
    };
    let user_msg_2 = ChatMessage {
        sender: SenderType::User,
        text: "Thanks, that's helpful!".to_string(),
        ..Default::default()
    };
    messages.clear();
    messages.push(user_msg_1.clone());
    messages.push(assistant_msg_1.clone());
    messages.push(user_msg_2.clone());

    let (response3, thought3, _tool_calls3, cache3) =
        complete_with_tools(&model, &messages, &tools, &settings).await;
    println!("Turn 3 response: {}", response3);
    println!("Turn 3 thought: {:?}", thought3);
    println!(
        "Turn 3 cache_hit={}, transition={:?}",
        cache3.as_ref().map(|c| c.cache_hit).unwrap_or(false),
        cache3
            .as_ref()
            .and_then(|c| c.checkpoint_transition.clone())
    );
    assert!(!response3.is_empty(), "Turn 3 response should not be empty");
    assert!(cache3.is_some(), "Cache metrics should be present");

    // Turn 4: Tool call scenario - user asks for weather, model makes tool call
    let user_msg_3 = ChatMessage {
        sender: SenderType::User,
        text: "What's the weather in Tokyo?".to_string(),
        ..Default::default()
    };
    let assistant_msg_3 = ChatMessage {
        sender: SenderType::Assistant,
        text: response3.clone(),
        thought: thought3.clone(),
        ..Default::default()
    };
    messages.clear();
    messages.push(user_msg_1.clone());
    messages.push(assistant_msg_1);
    messages.push(user_msg_2.clone());
    messages.push(assistant_msg_3);
    messages.push(user_msg_3.clone());

    let (response4, _thought4, tool_calls4, cache4) =
        complete_with_tools(&model, &messages, &tools, &settings).await;
    println!("Turn 4 response: {}", response4);
    println!("Turn 4 tool_calls: {:?}", tool_calls4);

    // The model might or might not make a tool call depending on its behavior
    // But we should still get a response
    if let Some(calls) = tool_calls4.filter(|c| !c.is_empty()) {
        println!(
            "Tool call detected: {}({:?})",
            calls[0].name, calls[0].arguments
        );

        // Execute the tool
        let tool = tools
            .iter()
            .find(|t| t.name() == calls[0].name)
            .expect("Tool not found");
        let args: serde_json::Value =
            serde_json::from_str(&calls[0].arguments).expect("Invalid tool arguments");
        let output = tool.execute(&args).await.expect("Tool execution failed");
        println!("Tool output: {}", output);

        // Add tool result message (this tests ToolResponse checkpoint transition)
        let tool_result_msg = ChatMessage {
            sender: SenderType::Tool,
            text: serde_json::to_string(&ToolResult {
                call: calls[0].clone(),
                output,
            })
            .unwrap(),
            ..Default::default()
        };
        messages.push(tool_result_msg);

        // Continue with tool results for final response
        let (response5, _thought5, _tool_calls5, cache5) =
            complete_with_tools(&model, &messages, &tools, &settings).await;
        println!("Turn 5 (after tool) response: {}", response5);
        assert!(!response5.is_empty(), "Turn 5 response should not be empty");
        assert!(cache5.is_some(), "Cache metrics should be present");
        println!(
            "Turn 5 cache_hit={}, transition={:?}",
            cache5.as_ref().unwrap().cache_hit,
            cache5.as_ref().unwrap().checkpoint_transition
        );
    }

    // Verify cache metrics for Turn 4
    assert!(
        cache4.is_some(),
        "Cache metrics should be present for Turn 4"
    );
    println!(
        "Turn 4 cache_hit={}, transition={:?}",
        cache4.as_ref().unwrap().cache_hit,
        cache4.as_ref().unwrap().checkpoint_transition
    );
}
