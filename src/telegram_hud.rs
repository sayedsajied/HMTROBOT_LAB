#![allow(dead_code)]

use crate::ai_router::AiRegime;
use crate::execution::ExecutionMode;
use crate::risk::RiskRejectReason;
use crate::sniper_strategy::TradeDirection;
use crate::types::SymbolId;
use std::fmt::Write as FmtWrite;

/// Runtime configuration for non-blocking Telegram HUD worker
#[derive(Debug, Clone)]
pub struct HudConfig {
    pub enabled: bool,
    pub bot_token: String,
    pub chat_id: String,
    pub min_alert_interval_ms: u64,
    pub heartbeat_interval_ms: u64,
    pub queue_capacity: usize,
}

impl HudConfig {
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            bot_token: String::new(),
            chat_id: String::new(),
            min_alert_interval_ms: 1_500,
            heartbeat_interval_ms: 300_000,
            queue_capacity: 64,
        }
    }

    #[inline]
    pub fn is_ready(&self) -> bool {
        self.enabled && !self.bot_token.trim().is_empty() && !self.chat_id.trim().is_empty()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HudSeverity {
    Info,
    Trade,
    Warning,
    Critical,
}

/// Zero-allocation stack event emitted from hot path to background Telegram worker
#[derive(Debug, Clone, Copy)]
pub enum HudEvent {
    Startup {
        mode: ExecutionMode,
        equity_usd: f64,
    },
    LiveOrder {
        symbol_id: SymbolId,
        direction: TradeDirection,
        size_contracts: u64,
        ok: bool,
        status_code: u16,
        exchange_order_id: u64,
    },
    TradeFill {
        symbol_id: SymbolId,
        direction: TradeDirection,
        fill_qty: f64,
        fill_price: f64,
        fee_usd: f64,
        realized_pnl_usd: f64,
        post_fee_pnl_bps: f64,
        is_maker: bool,
        is_hard_sl: bool,
        is_take_profit: bool,
        regime: AiRegime,
        top_engine_id: u16,
    },
    RiskAlert {
        symbol_id: SymbolId,
        reason: RiskRejectReason,
        safe_mode: bool,
        daily_pnl_pct: f64,
    },
    ReconcileAlert {
        balance_drift_usd: f64,
        positions_corrected: usize,
        severe: bool,
    },
    HeartbeatSummary {
        mode: ExecutionMode,
        equity_usd: f64,
        daily_pnl_pct: f64,
        gross_exposure_usd: f64,
        closed_trades: u64,
        winning_trades: u64,
        regime: AiRegime,
        top_engine_id: u16,
        safe_mode: bool,
    },
    Shutdown {
        equity_usd: f64,
        cancelled_orders: usize,
    },
}

impl HudEvent {
    #[inline]
    pub const fn severity(&self) -> HudSeverity {
        match self {
            Self::Startup { .. } | Self::HeartbeatSummary { .. } | Self::Shutdown { .. } => {
                HudSeverity::Info
            }
            Self::LiveOrder { .. } | Self::TradeFill { .. } => HudSeverity::Trade,
            Self::RiskAlert { safe_mode: true, .. }
            | Self::ReconcileAlert { severe: true, .. } => HudSeverity::Critical,
            Self::RiskAlert { .. } | Self::ReconcileAlert { .. } => HudSeverity::Warning,
        }
    }
}

#[inline]
pub const fn symbol_label(id: SymbolId) -> &'static str {
    match id.0 {
        0 => "BTC",
        1 => "ETH",
        2 => "SOL",
        _ => "ALT",
    }
}

#[inline]
pub const fn regime_label(r: AiRegime) -> &'static str {
    match r {
        AiRegime::CalmRange => "CALM",
        AiRegime::Trending => "TREND",
        AiRegime::VolatileBurst => "BURST",
        AiRegime::ToxicCrisis => "TOXIC",
    }
}

#[inline]
pub const fn mode_label(m: ExecutionMode) -> &'static str {
    match m {
        ExecutionMode::Paper => "PAPER",
        ExecutionMode::Live => "LIVE",
    }
}

/// Formats a HudEvent into a compact mobile-friendly Telegram message (background worker only)
pub fn format_hud_message(event: &HudEvent) -> String {
    let mut s = String::with_capacity(256);
    match *event {
        HudEvent::Startup { mode, equity_usd } => {
            let _ = write!(
                s,
                "HMTROBOT ONLINE [{}]\nPairs: BTC | ETH | SOL\nEquity: ${:.2}",
                mode_label(mode),
                equity_usd
            );
        }
        HudEvent::LiveOrder {
            symbol_id,
            direction,
            size_contracts,
            ok,
            status_code,
            exchange_order_id,
        } => {
            let side = match direction {
                TradeDirection::Long => "BUY",
                TradeDirection::Short => "SELL",
                TradeDirection::Flat => "FLAT",
            };
            let _ = write!(
                s,
                "DELTA ORDER {} {} x{}\nStatus: {} (HTTP {})\nOrder ID: {}",
                symbol_label(symbol_id),
                side,
                size_contracts,
                if ok { "ACCEPTED" } else { "REJECTED" },
                status_code,
                exchange_order_id
            );
        }
        HudEvent::TradeFill {
            symbol_id,
            direction,
            fill_qty,
            fill_price,
            fee_usd,
            realized_pnl_usd,
            post_fee_pnl_bps,
            is_maker,
            is_hard_sl,
            is_take_profit,
            regime,
            top_engine_id,
        } => {
            let side = match direction {
                TradeDirection::Long => "LONG",
                TradeDirection::Short => "SHORT",
                TradeDirection::Flat => "FLAT",
            };
            let tag = if is_hard_sl {
                "HARD-SL"
            } else if is_take_profit {
                "TP"
            } else if is_maker {
                "MAKER"
            } else {
                "TAKER"
            };
            let _ = write!(
                s,
                "FILL {} {} [{}]\nQty: {:.6} @ {:.2} | Fee: ${:.4}\nNet PnL: ${:+.4} ({:+.2} bps)\nAI: {} | Top Engine: E{:02}",
                symbol_label(symbol_id),
                side,
                tag,
                fill_qty,
                fill_price,
                fee_usd,
                realized_pnl_usd,
                post_fee_pnl_bps,
                regime_label(regime),
                top_engine_id
            );
        }
        HudEvent::RiskAlert {
            symbol_id,
            reason,
            safe_mode,
            daily_pnl_pct,
        } => {
            let _ = write!(
                s,
                "RISK ALERT [{}]\nReason: {:?}\nSafe-Mode: {} | Daily PnL: {:+.2}%",
                symbol_label(symbol_id),
                reason,
                if safe_mode { "ON" } else { "OFF" },
                daily_pnl_pct
            );
        }
        HudEvent::ReconcileAlert {
            balance_drift_usd,
            positions_corrected,
            severe,
        } => {
            let _ = write!(
                s,
                "EXCHANGE SYNC {}\nBal Drift: ${:+.4} | Pos Fixed: {}",
                if severe { "[DRIFT]" } else { "[OK]" },
                balance_drift_usd,
                positions_corrected
            );
        }
        HudEvent::HeartbeatSummary {
            mode,
            equity_usd,
            daily_pnl_pct,
            gross_exposure_usd,
            closed_trades,
            winning_trades,
            regime,
            top_engine_id,
            safe_mode,
        } => {
            let wr = if closed_trades > 0 {
                (winning_trades as f64 / closed_trades as f64) * 100.0
            } else {
                0.0
            };
            let _ = write!(
                s,
                "HUD [{}] {}\nEq: ${:.2} ({:+.2}%) | Exp: ${:.2}\nTrades: {} (WR: {:.1}%)\nAI: {} | Top: E{:02}",
                mode_label(mode),
                if safe_mode { "SAFE-MODE" } else { "ACTIVE" },
                equity_usd,
                daily_pnl_pct,
                gross_exposure_usd,
                closed_trades,
                wr,
                regime_label(regime),
                top_engine_id
            );
        }
        HudEvent::Shutdown {
            equity_usd,
            cancelled_orders,
        } => {
            let _ = write!(
                s,
                "HMTROBOT SHUTDOWN\nEq: ${:.2} | Cancelled: {}",
                equity_usd, cancelled_orders
            );
        }
    }
    s
}
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::mpsc;

/// Stack-only anti-flood rate limiter for background Telegram worker
#[derive(Debug, Clone, Copy)]
pub struct HudRateLimiter {
    pub min_interval_ms: u64,
    last_sent_ms: u64,
}

impl HudRateLimiter {
    pub const fn new(min_interval_ms: u64) -> Self {
        Self {
            min_interval_ms,
            last_sent_ms: 0,
        }
    }

    /// Critical & Trade events always pass; Info/Warning respect min_interval_ms
    pub fn should_send(&mut self, severity: HudSeverity, now_ms: u64) -> bool {
        if matches!(severity, HudSeverity::Critical | HudSeverity::Trade) {
            self.last_sent_ms = now_ms;
            return true;
        }
        if self.last_sent_ms == 0 || now_ms.saturating_sub(self.last_sent_ms) >= self.min_interval_ms {
            self.last_sent_ms = now_ms;
            return true;
        }
        false
    }
}

/// Non-blocking, zero-panic hot-path handle for pushing stack HudEvents
#[derive(Debug, Clone)]
pub struct TelegramHudHandle {
    tx: Option<mpsc::Sender<HudEvent>>,
    dropped: Arc<AtomicU64>,
}

impl TelegramHudHandle {
    pub fn disabled() -> Self {
        Self {
            tx: None,
            dropped: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Never blocks the trading loop; drops cleanly if background queue is full
    #[inline]
    pub fn try_send(&self, event: HudEvent) -> bool {
        match &self.tx {
            Some(sender) => match sender.try_send(event) {
                Ok(()) => true,
                Err(_) => {
                    self.dropped.fetch_add(1, Ordering::Relaxed);
                    false
                }
            },
            None => false,
        }
    }

    #[inline]
    pub fn dropped_count(&self) -> u64 {
        self.dropped.load(Ordering::Relaxed)
    }

    #[inline]
    pub fn is_enabled(&self) -> bool {
        self.tx.is_some()
    }
}

fn escape_json_str(input: &str) -> String {
    let mut out = String::with_capacity(input.len() + 16);
    for ch in input.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {}
            c => out.push(c),
        }
    }
    out
}

#[inline]
fn now_epoch_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Spawns the non-blocking async Telegram worker (returns disabled handle if unconfigured)
pub fn spawn_telegram_hud(cfg: HudConfig) -> (TelegramHudHandle, Option<tokio::task::JoinHandle<()>>) {
    if !cfg.is_ready() {
        eprintln!("[TG_HUD] DISABLED: TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID missing in Render ENV");
        return (TelegramHudHandle::disabled(), None);
    }

    let client = match reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()
    {
        Ok(c) => c,
        Err(_) => {
            eprintln!("[TG_HUD] FAILED to build HTTP client");
            return (TelegramHudHandle::disabled(), None);
        }
    };

    let cap = cfg.queue_capacity.clamp(8, 512);
    let (tx, mut rx) = mpsc::channel::<HudEvent>(cap);
    let handle = TelegramHudHandle {
        tx: Some(tx),
        dropped: Arc::new(AtomicU64::new(0)),
    };

    let url = format!("https://api.telegram.org/bot{}/sendMessage", cfg.bot_token.trim());
    let chat_id = escape_json_str(cfg.chat_id.trim());
    let mut limiter = HudRateLimiter::new(cfg.min_alert_interval_ms);

    eprintln!("[TG_HUD] WORKER STARTED");

    let join = tokio::spawn(async move {
        while let Some(event) = rx.recv().await {
            let now_ms = now_epoch_ms();
            let bypass = matches!(event, HudEvent::Startup { .. } | HudEvent::Shutdown { .. });
            if !bypass && !limiter.should_send(event.severity(), now_ms) {
                continue;
            }

            let text = escape_json_str(&format_hud_message(&event));
            let payload = format!(
                "{{\"chat_id\":\"{}\",\"text\":\"{}\",\"disable_web_page_preview\":true}}",
                chat_id, text
            );

            match client
                .post(&url)
                .header("Content-Type", "application/json")
                .body(payload)
                .send()
                .await
            {
                Ok(resp) => {
                    let status = resp.status();
                    if status.is_success() {
                        eprintln!("[TG_HUD] SEND OK (HTTP {})", status.as_u16());
                    } else {
                        let body = resp.text().await.unwrap_or_default();
                        let snippet: String = body.chars().take(180).collect();
                        eprintln!(
                            "[TG_HUD] SEND FAILED HTTP {} body={}",
                            status.as_u16(),
                            snippet
                        );
                    }
                }
                Err(e) => {
                    eprintln!("[TG_HUD] NETWORK ERROR: {}", e);
                }
            }
        }
    });

    (handle, Some(join))
}

#[cfg(test)]
mod telegram_hud_tests {
    use super::*;

    #[test]
    fn test_format_hud_messages_btc_eth_sol() {
        let ev = HudEvent::TradeFill {
            symbol_id: SymbolId(2),
            direction: TradeDirection::Long,
            fill_qty: 1.5,
            fill_price: 180.25,
            fee_usd: 0.054,
            realized_pnl_usd: 4.25,
            post_fee_pnl_bps: 15.72,
            is_maker: true,
            is_hard_sl: false,
            is_take_profit: false,
            regime: AiRegime::Trending,
            top_engine_id: 9,
        };
        let msg = format_hud_message(&ev);
        assert!(msg.contains("SOL LONG [MAKER]"));
        assert!(msg.contains("TREND"));
        assert!(msg.contains("E09"));

        let live_ord = format_hud_message(&HudEvent::LiveOrder {
            symbol_id: SymbolId(0),
            direction: TradeDirection::Long,
            size_contracts: 2,
            ok: true,
            status_code: 200,
            exchange_order_id: 424242,
        });
        assert!(live_ord.contains("DELTA ORDER BTC BUY x2"));
        assert!(live_ord.contains("ACCEPTED"));

        let start = format_hud_message(&HudEvent::Startup {
            mode: ExecutionMode::Paper,
            equity_usd: 1_000.0,
        });
        assert!(start.contains("BTC | ETH | SOL"));
        assert!(start.contains("[PAPER]"));
    }

    #[test]
    fn test_rate_limiter_allows_critical_and_throttles_warnings() {
        let mut rl = HudRateLimiter::new(2_000);
        assert!(rl.should_send(HudSeverity::Warning, 10_000));
        assert!(!rl.should_send(HudSeverity::Warning, 11_000));
        assert!(rl.should_send(HudSeverity::Critical, 11_100));
        assert!(rl.should_send(HudSeverity::Trade, 11_200));
        assert!(rl.should_send(HudSeverity::Info, 13_500));
    }

    #[test]
    fn test_json_escape_and_disabled_handle() {
        let escaped = escape_json_str("Line1\n\"Quote\"\\Slash");
        assert_eq!(escaped, "Line1\\n\\\"Quote\\\"\\\\Slash");

        let h = TelegramHudHandle::disabled();
        assert!(!h.is_enabled());
        assert!(!h.try_send(HudEvent::Shutdown {
            equity_usd: 1_000.0,
            cancelled_orders: 0,
        }));
        assert_eq!(h.dropped_count(), 0);
    }
}
