#![allow(dead_code)]

mod ai_router;
mod config;
mod execution;
mod feeds;
mod math_kernel;
mod risk;
mod sniper_strategy;
mod telegram_hud;
mod types;
mod vault;

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

pub const NUM_PAIRS: usize = 3; // 0 = BTC, 1 = ETH, 2 = SOL

#[inline]
pub fn now_epoch_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Lock-free atomic telemetry exposed to Render's HTTP /health endpoint
pub struct HealthState {
    pub running: AtomicBool,
    pub safe_mode: AtomicBool,
    pub started_ms: AtomicU64,
    pub last_tick_ms: AtomicU64,
    pub ticks_processed: AtomicU64,
    pub equity_cents: AtomicU64,
    pub closed_trades: AtomicU64,
}

impl HealthState {
    pub fn new(initial_equity_usd: f64) -> Self {
        let now = now_epoch_ms();
        let cents = if initial_equity_usd.is_finite() && initial_equity_usd > 0.0 {
            (initial_equity_usd * 100.0) as u64
        } else {
            0
        };
        Self {
            running: AtomicBool::new(true),
            safe_mode: AtomicBool::new(false),
            started_ms: AtomicU64::new(now),
            last_tick_ms: AtomicU64::new(now),
            ticks_processed: AtomicU64::new(0),
            equity_cents: AtomicU64::new(cents),
            closed_trades: AtomicU64::new(0),
        }
    }

    #[inline]
    pub fn record_tick(&self, now_ms: u64, equity_usd: f64, closed_trades: u64, safe_mode: bool) {
        self.last_tick_ms.store(now_ms, Ordering::Relaxed);
        self.ticks_processed.fetch_add(1, Ordering::Relaxed);
        if equity_usd.is_finite() && equity_usd >= 0.0 {
            self.equity_cents
                .store((equity_usd * 100.0) as u64, Ordering::Relaxed);
        }
        self.closed_trades.store(closed_trades, Ordering::Relaxed);
        self.safe_mode.store(safe_mode, Ordering::Relaxed);
    }

    pub fn to_json(&self) -> String {
        let now = now_epoch_ms();
        let started = self.started_ms.load(Ordering::Relaxed);
        let uptime_sec = now.saturating_sub(started) / 1_000;
        let last_tick = self.last_tick_ms.load(Ordering::Relaxed);
        let tick_age_ms = now.saturating_sub(last_tick);
        let ticks = self.ticks_processed.load(Ordering::Relaxed);
        let eq_usd = (self.equity_cents.load(Ordering::Relaxed) as f64) / 100.0;
        let trades = self.closed_trades.load(Ordering::Relaxed);
        let safe = self.safe_mode.load(Ordering::Relaxed);
        let alive = self.running.load(Ordering::Relaxed);

        format!(
            "{{\"status\":\"{}\",\"pairs\":[\"BTC\",\"ETH\",\"SOL\"],\"uptime_sec\":{},\"ticks\":{},\"tick_age_ms\":{},\"equity_usd\":{:.2},\"closed_trades\":{},\"safe_mode\":{}}}",
            if alive { "ok" } else { "stopping" },
            uptime_sec,
            ticks,
            tick_age_ms,
            eq_usd,
            trades,
            safe
        )
    }
}

/// Ultra-lightweight pure-Tokio HTTP /health server for Render 24/7 keep-alive (Zero-Panic)
pub fn spawn_health_server(port: u16, health: Arc<HealthState>) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let bind_addr = format!("0.0.0.0:{}", port);
        let listener = match TcpListener::bind(&bind_addr).await {
            Ok(l) => l,
            Err(_) => return,
        };

        while health.running.load(Ordering::Relaxed) {
            let accept_res = tokio::time::timeout(Duration::from_secs(1), listener.accept()).await;
            let (mut stream, _) = match accept_res {
                Ok(Ok(conn)) => conn,
                _ => continue,
            };

            let body = health.to_json();
            tokio::spawn(async move {
                let mut buf = [0u8; 256];
                let _ = tokio::time::timeout(Duration::from_secs(2), stream.read(&mut buf)).await;
                let resp = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    body.len(),
                    body
                );
                let _ = stream.write_all(resp.as_bytes()).await;
                let _ = stream.shutdown().await;
            });
        }
    })
}

#[tokio::main]
async fn main() {
    // Full async supervisor wired in Chunk #2 & Chunk #3
}


