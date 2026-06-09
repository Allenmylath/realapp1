# ── Stage 1: Build ────────────────────────────────────────────────────────────
FROM fedora:40 AS builder
WORKDIR /app
COPY Cargo.toml Cargo.lock* ./
COPY src/ src/
RUN dnf install -y gcc gcc-c++ make pkg-config openssl-devel && \
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable && \
    dnf clean all
ENV PATH="/root/.cargo/bin:${PATH}"
RUN cargo build --release --bin realapp1

# ── Stage 2: Runtime ──────────────────────────────────────────────────────────
FROM fedora:40
RUN dnf install -y ca-certificates openssl && dnf clean all
COPY --from=builder /app/target/release/realapp1 /usr/local/bin/realapp1
ENV PORT=8080
EXPOSE 8080
CMD ["realapp1"]
