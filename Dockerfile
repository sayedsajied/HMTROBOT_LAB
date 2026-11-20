# Stage 1: Build optimized Pure-Rust release binary
FROM rust:1-slim-bookworm AS builder

WORKDIR /app

# Copy manifest & source
COPY Cargo.toml ./
COPY src ./src

# Build release binary (matches [package] name = "hmtrobot")
RUN cargo build --release

# Stage 2: Minimal Linux runtime (<50MB RAM on Render)
FROM debian:bookworm-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /app/target/release/hmtrobot /usr/local/bin/hmtrobot

ENV PORT=10000
EXPOSE 10000

CMD ["hmtrobot"]
