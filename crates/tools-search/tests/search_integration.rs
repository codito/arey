use arey_core::tools::Tool;
use arey_tools_search::{SearchResult, SearchTool};
use serde_json::json;

/// This test is marked as `ignore` because it makes a network request to a public service.
/// It should be run manually when testing the searxng provider integration.
/// To run this test: `cargo test --package arey-tools-search --test search_integration -- --ignored`
#[tokio::test]
#[ignore]
async fn test_searxng_public_instance() {
    // Using public instance https://search.codito.in
    let yaml_config = r#"
        provider: searxng
        base_url: "https://search.codito.in"
    "#;
    let config_value: serde_yaml::Value = serde_yaml::from_str(yaml_config).unwrap();

    let search_tool = SearchTool::from_config(&config_value).expect("Failed to create search tool");

    let arguments = json!({ "query": "rust language" });

    let result = search_tool.execute(&arguments).await;

    assert!(
        result.is_ok(),
        "Search execution failed: {:?}",
        result.err()
    );

    let results_value = result.unwrap();
    let search_results: Vec<SearchResult> =
        serde_json::from_value(results_value).expect("Failed to parse search results");

    assert!(!search_results.is_empty(), "Search should return results");

    let first_result = &search_results[0];
    assert!(!first_result.title.is_empty());
    assert!(!first_result.url.is_empty());
}
