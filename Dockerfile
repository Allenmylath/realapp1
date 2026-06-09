# ── Stage 1: Build static musl binary ────────────────────────────────────────
FROM rust:1.82-slim AS builder

# musl target for fully static binary
RUN rustup target add x86_64-unknown-linux-musl \
    && apt-get update \
    && apt-get install -y musl-tools \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY Cargo.toml ./
COPY src ./src

RUN cargo build --release --target x86_64-unknown-linux-musl

# ── Stage 2: Minimal runtime image (~5 MB) ───────────────────────────────────
FROM scratch

# CA certs needed for HTTPS calls to Sarvam / OpenAI / Deepgram
COPY --from=builder /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/

COPY --from=builder /app/target/x86_64-unknown-linux-musl/release/rustvani-fly /rustvani-fly

EXPOSE 8080

ENTRYPOINT ["/rustvani-fly"]
