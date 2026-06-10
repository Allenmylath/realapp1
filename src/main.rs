mod tools;

use std::sync::Arc;

use axum::{
    Router,
    extract::{State, WebSocketUpgrade, ws::WebSocket},
    response::IntoResponse,
    routing::get,
};
use sqlx::postgres::PgPoolOptions;
use tower_http::cors::CorsLayer;

use rustvani::{
    system_clock, SileroVadNative, VadParams,
    PipelineParams, PipelineTask,
};
use rustvani::turn::SmartTurnConfig;
use rustvani::adapters::schemas::ToolsSchema;
use rustvani::context::shared_context_with_tools;
use rustvani::observer::BaseObserver;
use rustvani::processors::{
    llm_assistant_aggregator::LLMAssistantAggregator,
    llm_user_aggregator::LLMUserAggregator,
};
use rustvani::ravi::{
    RaviObserverParams,
    processor::{RaviParams, RaviProcessor},
};
use rustvani::services::{
    OpenAILLMConfig, OpenAILLMHandler,
    SarvamSttConfig, SarvamSttHandler,
    DeepgramTtsConfig, DeepgramTtsHandler,
};
use rustvani::services::llm::FunctionRegistry;
use rustvani::transport::websocket::{WebSocketParams, WebSocketTransport};
use rustvani::transport::TransportParams;

use tools::{client_tool_schemas, register_client_tools};

// ---------------------------------------------------------------------------
// System prompt
// ---------------------------------------------------------------------------

const SYSTEM_PROMPT: &str = "\
You are a helpful real estate voice assistant with access to a client CRM database.

You can:
- Add new clients (requires: name, phone, email, and notes; budget and areas are optional)
- Search for clients by name — even with typos or voice transcription variants; \
  the search tries phonetic Soundex first, then trigram similarity, then substring matching
- Edit an existing client's details (name, phone, email, notes, budget, areas, or \
  active/inactive status) — first search by name to find the client, confirm with the \
  user if more than one matches, then update only the fields that should change
- List all active clients

You CANNOT delete clients under any circumstances. If asked to delete, \
politely explain that deletion is not supported, and offer to mark the client \
inactive instead.

When adding or editing a client, always read back every recorded detail so the user \
can confirm. \
When searching, tell the user how the match was found (e.g. 'found via phonetic match'). \
Keep responses concise and natural — this is a voice interface.";

// ---------------------------------------------------------------------------
// Connection counter
// ---------------------------------------------------------------------------

static CONN_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

fn next_conn_id() -> u64 {
    CONN_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

// ---------------------------------------------------------------------------
// Shared app state
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct AppState {
    sarvam_api_key:   String,
    openai_api_key:   String,
    deepgram_api_key: String,
    /// Shared DB pool — cloned (Arc) into each connection's tool closures.
    db_pool:          Arc<sqlx::PgPool>,
}

// ---------------------------------------------------------------------------
// WebSocket handler
// ---------------------------------------------------------------------------

async fn ws_handler(
    ws: WebSocketUpgrade,
    State(state): State<AppState>,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_connection(socket, state))
}

async fn handle_connection(socket: WebSocket, app_state: AppState) {
    let conn_id = next_conn_id();
    log::info!("[conn={}] connected", conn_id);

    let vad_analyzer = match SileroVadNative::new(16_000) {
        Ok(v)  => Arc::new(v),
        Err(e) => { log::error!("[conn={}] VAD init failed: {}", conn_id, e); return; }
    };

    let transport = WebSocketTransport::new(
        &format!("WsTransport-{}", conn_id),
        WebSocketParams {
            transport: TransportParams {
                audio_in_enabled:         true,
                audio_in_sample_rate:     Some(16_000),
                audio_in_channels:        1,
                audio_in_passthrough:     true,
                audio_in_stream_on_start: true,
                vad_analyzer:             Some(vad_analyzer),
                vad_params:               VadParams { confidence: 0.4, min_volume: 0.1, ..VadParams::default() },
                audio_out_enabled:        true,
                turn_config:              Some(SmartTurnConfig { stop_secs: 1.5, ..SmartTurnConfig::default() }),
                ..TransportParams::default()
            },
        },
    );

    // --- Build registry and register client tools ---
    let mut registry = FunctionRegistry::new();
    register_client_tools(&mut registry, app_state.db_pool.clone());

    // --- Build tool schemas for LLM context ---
    let tool_schemas = ToolsSchema::new(client_tool_schemas());

    // --- Shared context with system prompt and tools ---
    let context = shared_context_with_tools(Some(SYSTEM_PROMPT.into()), tool_schemas, None);

    // --- RAVI processor ---
    let ravi = RaviProcessor::new(RaviParams {
        context: Some(context.clone()),
        ..RaviParams::default()
    });
    let ravi_observer: Arc<dyn BaseObserver> = Arc::new(
        RaviProcessor::create_observer(&ravi, RaviObserverParams::default()),
    );

    // --- STT ---
    let stt = SarvamSttHandler::new(SarvamSttConfig {
        api_key:  app_state.sarvam_api_key.clone(),
        model:    "saaras:v3".to_string(),
        language: Some("en-IN".to_string()),
        mode:     Some("transcribe".to_string()),
        ..SarvamSttConfig::default()
    })
    .into_processor();

    // --- LLM with registry ---
    let llm = OpenAILLMHandler::with_registry(
        OpenAILLMConfig {
            api_key: app_state.openai_api_key.clone(),
            model:   "gpt-4o-mini".to_string(),
            ..OpenAILLMConfig::default()
        },
        registry,
    )
    .into_processor();

    // --- TTS ---
    let tts = match DeepgramTtsHandler::new(DeepgramTtsConfig {
        api_key: app_state.deepgram_api_key.clone(),
        ..DeepgramTtsConfig::default()
    }) {
        Ok(t)  => t.into_processor(),
        Err(e) => { log::error!("[conn={}] TTS init failed: {}", conn_id, e); return; }
    };

    // --- Aggregators ---
    let user_agg      = LLMUserAggregator::new(context.clone());
    let assistant_agg = LLMAssistantAggregator::new(context.clone());

    // --- Pipeline ---
    let task = PipelineTask::new(
        vec![
            transport.input(),
            ravi,
            stt,
            user_agg,
            llm,
            assistant_agg,
            tts,
            transport.output(),
        ],
        PipelineParams { allow_interruptions: true, ..PipelineParams::default() },
    );

    let push_tx = task.push_sender();

    tokio::join!(
        async { task.run(system_clock(), Some(ravi_observer)).await.ok(); },
        transport.run_socket(socket, push_tx),
    );

    log::info!("[conn={}] disconnected", conn_id);
}

// ---------------------------------------------------------------------------
// Health check
// ---------------------------------------------------------------------------

async fn health() -> &'static str { "ok" }

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("info"),
    )
    .init();

    // --- DB pool ---
    let database_url = std::env::var("DATABASE_URL")
        .expect("DATABASE_URL not set");

    let db_pool = PgPoolOptions::new()
        .max_connections(5)
        .connect(&database_url)
        .await
        .unwrap_or_else(|e| panic!("Failed to connect to database: {}", e));

    let db_pool = Arc::new(db_pool);
    log::info!("Database pool initialised");

    let app_state = AppState {
        sarvam_api_key:   std::env::var("SARVAM_API_KEY").expect("SARVAM_API_KEY not set"),
        openai_api_key:   std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not set"),
        deepgram_api_key: std::env::var("DEEPGRAM_API_KEY").expect("DEEPGRAM_API_KEY not set"),
        db_pool,
    };

    let app = Router::new()
        .route("/ws", get(ws_handler))
        .route("/health", get(health))
        .layer(CorsLayer::permissive())
        .with_state(app_state);

    let port = std::env::var("PORT").unwrap_or_else(|_| "8080".to_string());
    let addr = format!("0.0.0.0:{}", port);

    log::info!("rustvani voice agent on ws://{}/ws", addr);

    let listener = tokio::net::TcpListener::bind(&addr).await
        .unwrap_or_else(|e| panic!("Failed to bind {}: {}", addr, e));

    axum::serve(listener, app).await.unwrap();
}
