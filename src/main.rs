use rustvani::*;
use std::sync::Arc;

const SYSTEM_PROMPT: &str = "You are a helpful voice assistant.";

#[tokio::main]
async fn main() {
    // --- Environment ---
    let sarvam_api_key = std::env::var("SARVAM_API_KEY")
        .expect("SARVAM_API_KEY must be set");
    let openai_api_key = std::env::var("OPENAI_API_KEY")
        .expect("OPENAI_API_KEY must be set");
    let deepgram_api_key = std::env::var("DEEPGRAM_API_KEY")
        .expect("DEEPGRAM_API_KEY must be set");

    // --- Shared conversation context ---
    let context = shared_context(Some(SYSTEM_PROMPT.into()));

    // --- VAD: pure Rust Silero, zero external deps ---
    let vad = SileroVadNative::new(16_000).expect("VAD load failed");

    // --- Transport: WebSocket via axum ---
    let transport = BaseTransport::new(TransportParams {
        audio_in_enabled: true,
        audio_in_sample_rate: Some(16_000),
        vad_analyzer: Some(Arc::new(vad)),
        audio_out_enabled: true,
        ..Default::default()
    });

    // --- STT: Sarvam saaras:v3 ---
    let stt = SarvamSttHandler::new(SarvamSttConfig {
        api_key: sarvam_api_key,
        ..Default::default()
    })
    .into_processor();

    // --- LLM: OpenAI gpt-4o-mini ---
    let llm = OpenAILLMHandler::new(OpenAILLMConfig {
        api_key: openai_api_key,
        model: "gpt-4o-mini".into(),
        context: context.clone(),
        ..Default::default()
    })
    .into_processor();

    // --- TTS: Deepgram Aura-2 WebSocket streaming ---
    let tts = DeepgramTtsHandler::new(DeepgramTtsConfig {
        api_key: deepgram_api_key,
        ..Default::default()
    })
    .expect("Deepgram TTS init failed")
    .into_processor();

    // --- Aggregators: bridge VAD ↔ LLM ↔ TTS ---
    let user_agg = LLMUserAggregator::new(context.clone()).into_processor();
    let assistant_agg = LLMAssistantAggregator::new(context.clone()).into_processor();

    // --- Pipeline ---
    let task = PipelineTask::new(
        vec![
            transport.input(),
            stt,
            user_agg,
            llm,
            assistant_agg,
            tts,
            transport.output(),
        ],
        PipelineParams {
            allow_interruptions: true,
            ..Default::default()
        },
    );

    task.run(system_clock(), None).await.unwrap();
}
