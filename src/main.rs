mod tools;

use std::sync::Arc;

use axum::{
    Router,
    Json,
    extract::{State, WebSocketUpgrade, ws::WebSocket},
    http::StatusCode,
    response::{Html, IntoResponse},
    routing::get,
};
use sqlx::postgres::PgPoolOptions;
use tower_http::cors::CorsLayer;

use rustvani::{
    system_clock, SileroVadNative, VadParams,
    PipelineParams, PipelineTask,
};
use rustvani::billing::{BillingStorage, PostgresBillingStorage, SessionBilling};
use uuid::Uuid;
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
use rustvani::transport::vaniwebrtc::{TurnServer, VaniWebRTCParams, VaniWebRTCTransport};
use rustvani::transport::TransportParams;

use tools::{client_tool_schemas, register_client_tools};

// ---------------------------------------------------------------------------
// System prompt
// ---------------------------------------------------------------------------

const SYSTEM_PROMPT: &str = "\
You are an experienced real estate agent's right hand — a sharp, warm colleague who \
knows the business inside out and has access to the client CRM database.

You can:
- Add new clients (requires: name, phone, email, and notes; budget and areas are optional)
- Search for clients by name — even with typos or voice transcription variants; \
  the search tries phonetic Soundex first, then trigram similarity, then substring matching. \
  Use get_client_by_name whenever the realtor gives you a name.
- Find a client when the realtor DOESN'T have the name — by what they're after, their \
  budget, or their area — using search_clients. Reach for this on requests like \
  'who was looking around 80 lakhs?', 'I forgot his name, the guy who needed a \
  wheelchair-accessible flat', or 'anyone interested in Bandra?'. The filters combine, \
  so 'someone under a crore wanting a villa in Andheri' is a single search.
- Prep the realtor for a client before a showing, call, or meeting — when they say \
  things like 'prep me for Priya', 'I've got a showing with the Sharmas, catch me up', \
  or 'what should I know before I see Rahul', use prep_client to pull the briefing
- Edit an existing client's details (name, phone, email, notes, budget, areas, or \
  active/inactive status) — first search to find the client, confirm with the \
  user if more than one matches, then update only the fields that should change
- List all active clients

Prepping the realtor for a client is one of your most important jobs, so make it \
feel effortless. When they ask to be prepped, call prep_client and turn what comes \
back into a few warm, natural sentences — like a sharp assistant catching them up in \
the hallway before they walk in. Lead with who the client is and what they're after, \
then budget and areas, and finish with one practical pointer or a detail worth \
confirming with them. Never read the briefing back as a list of fields.

You CANNOT delete clients under any circumstances. If asked to delete, \
politely explain that deletion is not supported, and offer to mark the client \
inactive instead.

This is a voice interface, so talk like a seasoned agent, not like a database readout. \
Be confident and deal-savvy: use natural, light real-estate phrasing, trust your read \
of the situation, and offer a quick instinct or the obvious next step ('Want me to pull \
her up?', 'I'd call him before the weekend if it were me'). Keep that energy but stay \
concise — don't lay on the jargon or ramble; one or two crisp sentences usually does it. \
Everything you say is spoken aloud by a text-to-speech engine, so output PLAIN SPOKEN \
WORDS ONLY. Never use any formatting or symbols: no asterisks, bold, bullet points, \
numbered lists, headings, markdown, emoji, or special characters. When you mention \
several clients, run through them in flowing speech — 'I've got three with garden space: \
Meera Reed, who's after energy-efficient features around Forest Heights; Ygritte, looking \
for greenhouse potential; and Leila Haddad, who wants a big kitchen' — not as a list. \
Say numbers and amounts the way a person speaks them — 'about nine and a half lakhs', \
'a budget up to one crore' — not as digits or with currency symbols, and don't read out \
phone numbers or emails unless the realtor asks for them. \
The detail lines you get back from the tools are for YOUR reference only — never \
recite them verbatim. Do not speak internal IDs, field labels, pipe characters, or \
how a match was found (phonetic, trigram, etc.) out loud. \
When you add or edit a client, confirm it naturally and mention only the couple of \
details that matter for them to catch a mistake — e.g. 'Got it, I've saved Priya with \
that mobile number and a budget up to one crore in Bandra. Sound right?' — rather than \
listing every field. \
When a search returns one obvious match, just use it and move on; only when several \
people match, briefly run through them so the realtor can pick.";

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
    /// Shared billing storage — created once, cloned (Arc) into each connection.
    billing_storage:  Arc<dyn BillingStorage>,
    /// STUN server URLs for the WebRTC transport; cloned per conn.
    ice_servers:      Vec<String>,
    /// Shared HTTP client for minting Cloudflare TURN credentials.
    http:             reqwest::Client,
    /// Cloudflare Realtime TURN key id + API token (from secrets). Used to mint
    /// short-lived TURN creds for both the server's peer connection and the
    /// browser (via `GET /ice`). TURN gives the server an IPv4 relay candidate
    /// an IPv4 browser can pair with — the bridge across Fly's IPv6-only NIC.
    turn_key_id:      String,
    turn_api_token:   String,
}

/// Mint short-lived ICE servers from Cloudflare. Returns the raw
/// `{ "iceServers": [...] }` body (STUN entry + a TURN entry with creds).
async fn mint_cloudflare_ice(state: &AppState) -> Result<serde_json::Value, String> {
    let url = format!(
        "https://rtc.live.cloudflare.com/v1/turn/keys/{}/credentials/generate-ice-servers",
        state.turn_key_id,
    );
    let resp = state
        .http
        .post(&url)
        .bearer_auth(&state.turn_api_token)
        .json(&serde_json::json!({ "ttl": 86_400 }))
        .send()
        .await
        .map_err(|e| format!("request failed: {e}"))?;
    if !resp.status().is_success() {
        return Err(format!("Cloudflare TURN API returned {}", resp.status()));
    }
    resp.json::<serde_json::Value>()
        .await
        .map_err(|e| format!("bad JSON: {e}"))
}

/// Pull the TURN entries (those carrying username/credential) out of a minted
/// Cloudflare ICE response into `rustvani` `TurnServer`s. The STUN-only entry,
/// which has no credentials, is naturally skipped.
fn turn_servers_from_ice(ice: &serde_json::Value) -> Vec<TurnServer> {
    ice.get("iceServers")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|e| {
                    let username = e.get("username")?.as_str()?.to_string();
                    let credential = e.get("credential")?.as_str()?.to_string();
                    let urls = e
                        .get("urls")?
                        .as_array()?
                        .iter()
                        .filter_map(|u| u.as_str().map(String::from))
                        .collect();
                    Some(TurnServer { urls, username, credential })
                })
                .collect()
        })
        .unwrap_or_default()
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

    // Mint fresh TURN creds for this connection's peer connection.
    let turn_servers = match mint_cloudflare_ice(&app_state).await {
        Ok(ice) => turn_servers_from_ice(&ice),
        Err(e) => {
            log::error!("[conn={}] TURN mint failed: {} — media may not connect", conn_id, e);
            vec![]
        }
    };

    let vad_analyzer = match SileroVadNative::new(16_000) {
        Ok(v)  => Arc::new(v),
        Err(e) => { log::error!("[conn={}] VAD init failed: {}", conn_id, e); return; }
    };

    let transport = VaniWebRTCTransport::new(
        &format!("VaniWebRTCTransport-{}", conn_id),
        VaniWebRTCParams {
            transport: TransportParams {
                audio_in_enabled:         true,
                audio_in_sample_rate:     Some(16_000),
                audio_in_channels:        1,
                audio_in_passthrough:     true,
                audio_in_stream_on_start: true,
                vad_analyzer:             Some(vad_analyzer),
                vad_params:               VadParams { confidence: 0.4, min_volume: 0.1, ..VadParams::default() },
                audio_out_enabled:        true,
                // Deepgram TTS emits 24 kHz PCM; the transport defaults to 16 kHz
                // when unset, which would play the voice ~1.5× slow (stretched/
                // pitched down). Match the TTS rate so Opus resamples correctly.
                audio_out_sample_rate:    Some(24_000),
                turn_config:              Some(SmartTurnConfig { stop_secs: 1.5, ..SmartTurnConfig::default() }),
                ..TransportParams::default()
            },
            // STUN + TURN. TURN gives the server an IPv4 relay candidate so an
            // IPv4 browser can pair with the IPv6-only Fly host. Opus tuning and
            // the optional 48 kHz denoiser hook keep their defaults.
            ice_servers:  app_state.ice_servers.clone(),
            turn_servers: turn_servers.clone(),
            ..VaniWebRTCParams::default()
        },
    );

    // --- Billing session (one per connection) ---
    let session_id = Uuid::new_v4();
    let (billing, billing_handle) =
        SessionBilling::new(session_id, app_state.billing_storage.clone(), 256);

    // Shared turn-id cells link transcript turns to the conversation.
    let active_user_turn_id = Arc::new(std::sync::Mutex::new(None));
    let active_bot_turn_id  = Arc::new(std::sync::Mutex::new(None));

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
    .with_billing(billing.clone())
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
    .with_billing(billing.clone())
    .into_processor();

    // --- TTS ---
    let tts = match DeepgramTtsHandler::new(DeepgramTtsConfig {
        api_key: app_state.deepgram_api_key.clone(),
        ..DeepgramTtsConfig::default()
    }) {
        Ok(t)  => t.with_billing(billing.clone()).into_processor(),
        Err(e) => { log::error!("[conn={}] TTS init failed: {}", conn_id, e); return; }
    };

    // --- Aggregators (these emit transcript turns) ---
    let user_agg = LLMUserAggregator::with_billing(
        context.clone(), billing.clone(), active_user_turn_id.clone(),
    );
    let assistant_agg = LLMAssistantAggregator::with_billing(
        context.clone(), billing.clone(), active_bot_turn_id.clone(),
    );

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
        PipelineParams {
            allow_interruptions:  true,
            enable_usage_metrics: true,
            billing_collector:    Some(billing.clone()),
            ..PipelineParams::default()
        },
    );

    let push_tx = task.push_sender();

    tokio::join!(
        async { task.run(system_clock(), Some(ravi_observer)).await.ok(); },
        transport.run(socket, push_tx),
    );

    // Drop the collector so the drain task can flush the final checkpoint +
    // finalize_session, then wait (bounded) for it to land before returning.
    drop(billing);
    let _ = tokio::time::timeout(
        std::time::Duration::from_secs(10), billing_handle,
    ).await;

    log::info!("[conn={}] disconnected", conn_id);
}

// ---------------------------------------------------------------------------
// Health check
// ---------------------------------------------------------------------------

async fn health() -> &'static str { "ok" }

// ---------------------------------------------------------------------------
// Static test client
// ---------------------------------------------------------------------------

/// Minimal browser WebRTC client for exercising the agent end-to-end.
/// Embedded at compile time so it ships inside the binary / Docker image.
async fn client() -> impl IntoResponse {
    Html(include_str!("../static/client.html"))
}

/// Mint short-lived ICE servers for the browser. The client fetches this before
/// creating its RTCPeerConnection so it has working (ephemeral) TURN creds.
async fn ice(State(state): State<AppState>) -> impl IntoResponse {
    match mint_cloudflare_ice(&state).await {
        Ok(body) => Json(body).into_response(),
        Err(e) => {
            log::error!("/ice mint failed: {}", e);
            (StatusCode::BAD_GATEWAY, "failed to mint ICE servers").into_response()
        }
    }
}

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

    // --- Billing storage (tokio_postgres, separate from the sqlx pool above) ---
    let connector = native_tls::TlsConnector::builder().build().unwrap();
    let tls = postgres_native_tls::MakeTlsConnector::new(connector);
    let (pg_client, pg_conn) = tokio_postgres::connect(&database_url, tls)
        .await
        .expect("billing DB connect failed");
    tokio::spawn(async move {
        if let Err(e) = pg_conn.await {
            log::error!("billing DB connection dropped: {e}");
        }
    });
    PostgresBillingStorage::run_migrations(&pg_client)
        .await
        .expect("billing migrations failed");
    let billing_storage: Arc<dyn BillingStorage> =
        Arc::new(PostgresBillingStorage::new(pg_client));
    log::info!("Billing storage initialised (billing_sessions + billing_events)");

    let ice_servers: Vec<String> = vec!["stun:stun.l.google.com:19302".to_string()];

    // Cloudflare Realtime TURN keys (set as Fly secrets). TURN creds are minted
    // per use; without these, media will NOT connect from Fly's IPv6-only host.
    let turn_key_id    = std::env::var("TURN_TOKEN_ID").expect("TURN_TOKEN_ID not set");
    let turn_api_token = std::env::var("TURN_API_TOKEN").expect("TURN_API_TOKEN not set");
    log::info!("ICE: stun={:?}, Cloudflare TURN key {}", ice_servers, turn_key_id);

    let app_state = AppState {
        sarvam_api_key:   std::env::var("SARVAM_API_KEY").expect("SARVAM_API_KEY not set"),
        openai_api_key:   std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not set"),
        deepgram_api_key: std::env::var("DEEPGRAM_API_KEY").expect("DEEPGRAM_API_KEY not set"),
        db_pool,
        billing_storage,
        ice_servers,
        http: reqwest::Client::new(),
        turn_key_id,
        turn_api_token,
    };

    let app = Router::new()
        .route("/", get(client))
        .route("/ice", get(ice))
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
