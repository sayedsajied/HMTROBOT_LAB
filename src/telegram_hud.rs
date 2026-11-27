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
