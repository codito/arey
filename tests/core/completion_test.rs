use arey::core::completion::{combine_metrics, CompletionMetrics, SenderType};
use arey::core::model::ModelMetrics;

#[test]
fn test_combine_metrics() {
    let metrics1 = CompletionMetrics {
        prompt_tokens: 100,
        prompt_eval_latency_ms: 50.0,
        completion_tokens: 20,
        completion_runs: 1,
        completion_latency_ms: 200.0,
    };

    let metrics2 = CompletionMetrics {
        prompt_tokens: 100,
        prompt_eval_latency_ms: 0.0,
        completion_tokens: 30,
        completion_runs: 1,
        completion_latency_ms: 300.0,
    };

    let combined = combine_metrics(&[metrics1, metrics2]);
    
    assert_eq!(combined.prompt_tokens, 100);
    assert_eq!(combined.prompt_eval_latency_ms, 50.0);
    assert_eq!(combined.completion_tokens, 50);
    assert_eq!(combined.completion_runs, 2);
    assert_eq!(combined.completion_latency_ms, 500.0);
}

#[test]
fn test_sender_type_roles() {
    assert_eq!(SenderType::System.role(), "system");
    assert_eq!(SenderType::Assistant.role(), "assistant");
    assert_eq!(SenderType::User.role(), "user");
}

// Mock implementation for testing
pub struct MockCompletionModel;

#[async_trait::async_trait]
impl arey::core::completion::CompletionModel for MockCompletionModel {
    fn context_size(&self) -> usize { 4096 }
    fn metrics(&self) -> ModelMetrics { ModelMetrics { init_latency_ms: 0.0 } }
    async fn load(&mut self, _text: &str) -> anyhow::Result<()> { Ok(()) }
    async fn complete(&mut self, _messages: &[arey::core::completion::ChatMessage], _settings: &std::collections::HashMap<String, String>) 
        -> futures::stream::BoxStream<'_, arey::core::completion::CompletionResponse> 
    {
        unimplemented!()
    }
    async fn count_tokens(&self, _text: &str) -> usize { 0 }
    async fn free(&mut self) {}
}

#[tokio::test]
async fn test_mock_completion_model() {
    let mut model = MockCompletionModel;
    model.load("test").await.unwrap();
    assert_eq!(model.context_size(), 4096);
}
