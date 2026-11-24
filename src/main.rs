#![allow(dead_code)]

mod ai_router;
mod config;
mod delta_rest;
mod execution;
mod feeds;
mod math_kernel;
mod risk;
mod sniper_strategy;
mod telegram_hud;
mod types;
mod vault;

use crate::delta_rest::{
    DeltaOrderRequest, DeltaOrderSide, DeltaOrderType, DeltaRestHandle,
};
use crate::execution::{LiveCommandKind, LiveDispatchCommand};
use crate::sniper_strategy::TradeDirection;
use crate::types::SymbolId;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

pub const NUM_PAIRS: usize = 3; // 0 = BTC, 1 = ETH, 2 = SOL (STRICTLY NO GOLD)

#[inline]
pub fn now_epoch_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Converts base coin qty into Delta Exchange contract count (BTC: 0.001, ETH: 0.01, SOL: 1.0)
#[inline]
pub fn qty_to_delta_contracts(symbol_id: SymbolId, qty: f64) -> u64 {
    if !qty.is_finite() || qty <= 0.0 {
        return 0;
    }
    let mult = match symbol_id.0 {
        0 => 1_000.0, // BTCUSD: 1 contract = 0.001 BTC
        1 => 100.0,   // ETHUSD: 1 contract = 0.01 ETH
        2 => 1.0,     // SOLUSD: 1 contract = 1 SOL
        _ => return 0,
    };
    let contracts = (qty * mult).round();
    if !contracts.is_finite() || contracts < 1.0 {
        1
    } else {
        (contracts as u64).min(10_000)
    }
}

/// Non-blocking bridge from ExecutionEngine LiveDispatchCommand -> DeltaRestHandle
pub fn dispatch_live_command(rest: &DeltaRestHandle, cmd: &LiveDispatchCommand) {
    if !rest.is_enabled() {
        return;
    }
    match cmd.kind {
        LiveCommandKind::CancelAllGlobal => {
            let _ = rest.try_cancel_all(None);
        }
        LiveCommandKind::CancelAllSymbol => {
            let _ = rest.try_cancel_all(Some(cmd.symbol_id));
        }
        LiveCommandKind::PlaceLimitPostOnly | LiveCommandKind::PlaceMarketIoc => {
            let side = match cmd.direction {
                TradeDirection::Long => DeltaOrderSide::Buy,
                TradeDirection::Short => DeltaOrderSide::Sell,
                TradeDirection::Flat => return,
            };
            let size_contracts = qty_to_delta_contracts(cmd.symbol_id, cmd.qty);
            if size_contracts == 0 {
                return;
            }
            let is_limit = cmd.kind == LiveCommandKind::PlaceLimitPostOnly;
            let req = DeltaOrderRequest {
                symbol_id: cmd.symbol_id,
                side,
                order_type: if is_limit {
                    DeltaOrderType::Limit
                } else {
                    DeltaOrderType::Market
                },
                size_contracts,
                limit_price: if is_limit { Some(cmd.price) } else { None },
                post_only: is_limit,
                reduce_only: cmd.reduce_only,
                client_order_id: format!("hmt_{}", cmd.client_order_id),
            };
            let _ = rest.try_place_order(req);
        }
        LiveCommandKind::PlaceHardStop | LiveCommandKind::CancelOrder => {}
    }
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

fn main() {}
use crate::ai_router::{AiRegime, AiRouter};
use crate::execution::{
    ExecutionEngine, ExecutionMode, ExecutionOutput, FillEvent, PaperSimConfig,
};
use crate::math_kernel::MathKernel;
use crate::risk::{RiskConfig, RiskEngine, RiskRejectReason};
use crate::sniper_strategy::{SniperStrategy, StrategyConfig};
use crate::telegram_hud::{HudEvent, TelegramHudHandle};
use crate::types::{FundingBasisSnapshot, L2OrderBook};
use crate::vault::{ExchangeAccountTruth, ExchangeReconciler, VaultLedger};
use std::path::PathBuf;

/// Master Zero-Panic Orchestrator for BTC (0), ETH (1), and SOL (2)
pub struct BotEngine {
    pub kernels: [MathKernel; NUM_PAIRS],
    pub routers: [AiRouter; NUM_PAIRS],
    pub strategies: [SniperStrategy; NUM_PAIRS],
    pub risk: RiskEngine,
    pub exec: ExecutionEngine,
    pub vault: VaultLedger,
    pub reconciler: ExchangeReconciler,
    pub hud: TelegramHudHandle,
    pub delta_rest: DeltaRestHandle,
    pub health: Arc<HealthState>,
    pub vault_path: PathBuf,
    pub journal_path: PathBuf,
    entry_regimes: [AiRegime; NUM_PAIRS],
    last_feed_ms: [u64; NUM_PAIRS],
    last_heartbeat_hud_ms: u64,
    last_utc_day: u64,
}

impl BotEngine {
    pub fn new(
        mode: ExecutionMode,
        initial_capital_usd: f64,
        vault_path: PathBuf,
        journal_path: PathBuf,
        hud: TelegramHudHandle,
        delta_rest: DeltaRestHandle,
        health: Arc<HealthState>,
    ) -> Self {
        let vault = VaultLedger::load_or_init(&vault_path, initial_capital_usd);
        let risk_cfg = RiskConfig::default_crypto();
        let strat_cfg = StrategyConfig::default_crypto();
        let now = now_epoch_ms();

        Self {
            kernels: [MathKernel::new(); NUM_PAIRS],
            routers: [AiRouter::new(3_000, risk_cfg.max_daily_loss_pct); NUM_PAIRS],
            strategies: [SniperStrategy::new(strat_cfg); NUM_PAIRS],
            risk: RiskEngine::new(risk_cfg),
            exec: ExecutionEngine::new(mode, PaperSimConfig::default_crypto()),
            vault,
            reconciler: ExchangeReconciler::default_30s(),
            hud,
            delta_rest,
            health,
            vault_path,
            journal_path,
            entry_regimes: [AiRegime::CalmRange; NUM_PAIRS],
            last_feed_ms: [now; NUM_PAIRS],
            last_heartbeat_hud_ms: now,
            last_utc_day: now / 86_400_000,
        }
    }

    fn process_fill(&mut self, fill: &FillEvent, idx: usize) {
        let outcome = self.vault.apply_fill(fill);
        let regime = self.entry_regimes[idx];

        if outcome.closed_position {
            self.routers[idx].on_trade_closed(
                regime,
                outcome.post_fee_pnl_bps,
                outcome.daily_pnl_pct,
                outcome.is_hard_sl,
                outcome.is_take_profit,
            );
        }

        let _ = self.vault.save_atomic(&self.vault_path);
        if let Some(entry) = self.vault.latest_journal_entry() {
            let _ = VaultLedger::append_journal_csv(&self.journal_path, entry);
        }

        self.hud.try_send(HudEvent::TradeFill {
            symbol_id: fill.symbol_id,
            direction: fill.direction,
            fill_qty: fill.fill_qty,
            fill_price: fill.fill_price,
            fee_usd: fill.fee_usd,
            realized_pnl_usd: outcome.realized_pnl_usd,
            post_fee_pnl_bps: outcome.post_fee_pnl_bps,
            is_maker: fill.is_maker,
            is_hard_sl: fill.is_hard_sl,
            is_take_profit: fill.is_take_profit,
            regime,
            top_engine_id: self.routers[idx].top_engine_id(),
        });
    }

    /// Processes fills (Paper) and dispatches live wire commands (Delta REST)
    fn handle_exec_output(&mut self, exec_out: &ExecutionOutput, idx: usize) {
        for i in 0..exec_out.fill_count {
            if let Some(fill) = exec_out.fills[i] {
                self.process_fill(&fill, idx);
            }
        }
        for i in 0..exec_out.cmd_count {
            if let Some(cmd) = exec_out.live_cmds[i] {
                dispatch_live_command(&self.delta_rest, &cmd);
            }
        }
    }

    /// Single-symbol hot-path tick: L2 + Funding -> 86 Engines -> 3-Layer AI -> Strategy -> Risk -> Exec
    pub fn on_market_snapshot(
        &mut self,
        symbol_id: SymbolId,
        book: &L2OrderBook,
        fb: &FundingBasisSnapshot,
        now_ms: u64,
    ) {
        let idx = symbol_id.0 as usize;
        if idx >= NUM_PAIRS || !book.is_valid() {
            return;
        }
        let (bid, ask) = match (book.best_bid(), book.best_ask()) {
            (Some(b), Some(a))
                if b.price.is_finite()
                    && a.price.is_finite()
                    && b.price > 0.0
                    && a.price > b.price =>
            {
                (b.price, a.price)
            }
            _ => return,
        };

        let utc_day = now_ms / 86_400_000;
        if utc_day > self.last_utc_day {
            self.last_utc_day = utc_day;
            self.vault.on_utc_day_rollover();
            self.risk.on_utc_day_rollover();
        }

        let mid = (bid + ask) * 0.5;
        let spread_bps = ((ask - bid) / mid) * 10_000.0;
        self.last_feed_ms[idx] = now_ms;
        self.risk.on_heartbeat(now_ms);
        self.vault.update_mark_price(symbol_id, mid, now_ms);

        let tick_exec = self.exec.on_market_tick(symbol_id, bid, ask, now_ms);
        self.handle_exec_output(&tick_exec, idx);

        let btc_ret = self.kernels[0].last_ret_bps;
        let batch = self.kernels[idx].on_snapshot(symbol_id, book, fb, btc_ret, now_ms);
        let vol_bps = self.kernels[idx].ewma_vol_bps;
        let mtf_trend = self.kernels[idx].mtf_trend_alignment();
        let ret_bps = self.kernels[idx].last_ret_bps;

        let decision = self.routers[idx].route(
            &batch,
            vol_bps,
            mtf_trend,
            spread_bps,
            ret_bps,
            now_ms,
            self.last_feed_ms[idx],
        );

        let has_open = self.exec.osm.active_count_for(symbol_id) > 0;
        let inv = self.vault.inventory_state(symbol_id, has_open);
        let intent = self.strategies[idx].evaluate(&decision, &batch, book, &inv, vol_bps);

        let snap = self.vault.to_risk_snapshot(self.exec.osm.total_active_count() as u32);
        let verdict = self.risk.evaluate(&intent, &inv, &snap, now_ms);

        if !verdict.approved
            && matches!(
                verdict.reject_reason,
                RiskRejectReason::KillSwitchActive
                    | RiskRejectReason::DeadManSwitchTripped
                    | RiskRejectReason::DailyLossLimitBreached
                    | RiskRejectReason::MaxDrawdownBreached
                    | RiskRejectReason::ConsecutiveRejectsTripped
            )
        {
            self.hud.try_send(HudEvent::RiskAlert {
                symbol_id,
                reason: verdict.reject_reason,
                safe_mode: true,
                daily_pnl_pct: self.vault.daily_pnl_pct(),
            });
        }

        let is_reduce = RiskEngine::is_risk_reducing(intent.direction, &inv);
        if verdict.approved && !is_reduce {
            self.entry_regimes[idx] = decision.regime;
        }
        let exec_out = self.exec.execute_intent(&intent, &verdict, bid, ask, is_reduce, now_ms);
        self.handle_exec_output(&exec_out, idx);

        let any_safe = self.routers[0].is_safe_mode()
            || self.routers[1].is_safe_mode()
            || self.routers[2].is_safe_mode()
            || self.risk.breaker.is_halted();
        self.health.record_tick(
            now_ms,
            self.vault.total_equity_usd(),
            self.vault.total_closed_trades,
            any_safe,
        );
    }

    /// Periodic 30s Exchange-vs-Vault Reconciliation hook (Wintermute Rule #11)
    pub fn on_exchange_sync(&mut self, truth: &ExchangeAccountTruth, now_ms: u64) {
        if !self.reconciler.is_sync_due(now_ms) {
            return;
        }
        let rep = self.reconciler.reconcile(&mut self.vault, truth, now_ms);
        let _ = self.vault.save_atomic(&self.vault_path);
        if rep.severe_drift_detected || rep.positions_corrected > 0 {
            self.hud.try_send(HudEvent::ReconcileAlert {
                balance_drift_usd: rep.balance_drift_usd,
                positions_corrected: rep.positions_corrected,
                severe: rep.severe_drift_detected,
            });
        }
    }

    /// Periodic Telegram HUD summary (every heartbeat_interval_ms)
    pub fn maybe_emit_hud_heartbeat(&mut self, now_ms: u64, interval_ms: u64) {
        if now_ms.saturating_sub(self.last_heartbeat_hud_ms) < interval_ms {
            return;
        }
        self.last_heartbeat_hud_ms = now_ms;
        let any_safe = self.routers[0].is_safe_mode() || self.risk.breaker.is_halted();
        self.hud.try_send(HudEvent::HeartbeatSummary {
            mode: self.exec.mode,
            equity_usd: self.vault.total_equity_usd(),
            daily_pnl_pct: self.vault.daily_pnl_pct(),
            gross_exposure_usd: self.vault.total_gross_exposure_usd(),
            closed_trades: self.vault.total_closed_trades,
            winning_trades: self.vault.winning_trades,
            regime: self.routers[0].hmm.current,
            top_engine_id: self.routers[0].top_engine_id(),
            safe_mode: any_safe,
        });
    }

    /// Graceful SIGTERM shutdown: Cancel-All active orders + Atomic Vault Save + HUD Notice
    pub fn graceful_shutdown(&mut self, now_ms: u64) {
        self.health.running.store(false, Ordering::Relaxed);
        let out = self.exec.cancel_all_global(now_ms);
        for i in 0..out.cmd_count {
            if let Some(cmd) = out.live_cmds[i] {
                dispatch_live_command(&self.delta_rest, &cmd);
            }
        }
        let _ = self.vault.save_atomic(&self.vault_path);
        self.hud.try_send(HudEvent::Shutdown {
            equity_usd: self.vault.total_equity_usd(),
            cancelled_orders: out.cancelled_count,
        });
    }
}

fn main() {}
