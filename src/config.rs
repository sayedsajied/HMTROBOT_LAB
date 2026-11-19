#![allow(dead_code)]

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExchangeId {
    DeltaExchange,
    Bybit,
    Binance,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TradingMode {
    Paper,
    Live,
}

#[derive(Debug, Clone, Copy)]
pub struct FeeConfig {
    pub maker_fee_bps: f64,
    pub taker_fee_bps: f64,
    pub slippage_bps: f64,
}

impl FeeConfig {
    /// Total round-trip taker cost in basis points (entry + exit + 2x slippage)
    #[inline]
    pub fn round_trip_taker_bps(&self) -> f64 {
        (self.taker_fee_bps * 2.0) + (self.slippage_bps * 2.0)
    }

    /// Total round-trip maker cost in basis points (entry + exit + 1x slippage buffer)
    #[inline]
    pub fn round_trip_maker_bps(&self) -> f64 {
        (self.maker_fee_bps * 2.0) + self.slippage_bps
    }

    /// Minimum expected move (bps) required to clear the Jump-style edge-to-fee gate
    #[inline]
    pub fn min_required_edge_bps(&self, edge_ratio: f64, is_maker: bool) -> f64 {
        let base_cost = if is_maker {
            self.round_trip_maker_bps()
        } else {
            self.round_trip_taker_bps()
        };
        base_cost * edge_ratio.max(1.0)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct RiskConfig {
    pub max_daily_loss_pct: f64,
    pub max_position_usd: f64,
    pub max_leverage: f64,
    pub max_spread_bps: f64,
    pub stale_feed_ms: u64,
    pub trade_cooldown_ms: u64,
    pub min_edge_to_fee_ratio: f64,
    pub max_consecutive_losses: u32,
}

#[derive(Debug, Clone)]
pub struct BotConfig {
    pub exchange: ExchangeId,
    pub mode: TradingMode,
    pub symbols: Vec<String>,
    pub initial_capital_usd: f64,
    pub fees: FeeConfig,
    pub risk: RiskConfig,
    pub api_key: String,
    pub api_secret: String,
    pub telegram_bot_token: String,
    pub telegram_chat_id: String,
    pub health_port: u16,
}

impl Default for BotConfig {
    fn default() -> Self {
        Self {
            exchange: ExchangeId::DeltaExchange,
            mode: TradingMode::Paper,
            symbols: vec!["BTCUSD".to_string(), "PAXGUSD".to_string()],
            initial_capital_usd: 100.0,
            fees: FeeConfig {
                maker_fee_bps: 2.0, // 0.02%
                taker_fee_bps: 5.0, // 0.05%
                slippage_bps: 1.5,  // 0.015% realistic live buffer
            },
            risk: RiskConfig {
                max_daily_loss_pct: 3.0,
                max_position_usd: 25.0,
                max_leverage: 3.0,
                max_spread_bps: 4.5,
                stale_feed_ms: 3000,
                trade_cooldown_ms: 15_000,
                min_edge_to_fee_ratio: 3.0,
                max_consecutive_losses: 4,
            },
            api_key: String::new(),
            api_secret: String::new(),
            telegram_bot_token: String::new(),
            telegram_chat_id: String::new(),
            health_port: 10000,
        }
    }
}
impl ExchangeId {
    pub fn ws_url(&self) -> &'static str {
        match self {
            Self::DeltaExchange => "wss://socket.india.delta.exchange",
            Self::Bybit => "wss://stream.bybit.com/v5/public/linear",
            Self::Binance => "wss://fstream.binance.com/ws",
        }
    }

    pub fn rest_url(&self) -> &'static str {
        match self {
            Self::DeltaExchange => "https://api.india.delta.exchange",
            Self::Bybit => "https://api.bybit.com",
            Self::Binance => "https://fapi.binance.com",
        }
    }
}

fn env_f64(key: &str, default: f64) -> f64 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.trim().parse::<f64>().ok())
        .filter(|v| v.is_finite())
        .unwrap_or(default)
}

fn env_u64(key: &str, default: u64) -> u64 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .unwrap_or(default)
}

impl BotConfig {
    /// Zero-panic loader from Render / OS Environment Variables
    pub fn from_env() -> Result<Self, String> {
        let mut cfg = Self::default();

        if let Ok(ex) = std::env::var("EXCHANGE") {
            cfg.exchange = match ex.trim().to_uppercase().as_str() {
                "BYBIT" => ExchangeId::Bybit,
                "BINANCE" => ExchangeId::Binance,
                _ => ExchangeId::DeltaExchange,
            };
        }

        if let Ok(mode) = std::env::var("TRADING_MODE") {
            cfg.mode = match mode.trim().to_uppercase().as_str() {
                "LIVE" => TradingMode::Live,
                _ => TradingMode::Paper,
            };
        }

        if let Ok(syms) = std::env::var("SYMBOLS") {
            let parsed: Vec<String> = syms
                .split(',')
                .map(|s| s.trim().to_uppercase())
                .filter(|s| !s.is_empty())
                .collect();
            if !parsed.is_empty() {
                cfg.symbols = parsed;
            }
        }

        cfg.initial_capital_usd = env_f64("INITIAL_CAPITAL_USD", cfg.initial_capital_usd);
        cfg.risk.max_daily_loss_pct = env_f64("MAX_DAILY_LOSS_PCT", cfg.risk.max_daily_loss_pct);
        cfg.risk.max_position_usd = env_f64("MAX_POSITION_USD", cfg.risk.max_position_usd);
        cfg.risk.max_leverage = env_f64("MAX_LEVERAGE", cfg.risk.max_leverage);
        cfg.risk.max_spread_bps = env_f64("MAX_SPREAD_BPS", cfg.risk.max_spread_bps);
        cfg.risk.stale_feed_ms = env_u64("STALE_FEED_MS", cfg.risk.stale_feed_ms);
        cfg.health_port = env_u64("PORT", cfg.health_port as u64).clamp(1, 65535) as u16;

        cfg.api_key = std::env::var("API_KEY").unwrap_or_default().trim().to_string();
        cfg.api_secret = std::env::var("API_SECRET").unwrap_or_default().trim().to_string();
        cfg.telegram_bot_token = std::env::var("TELEGRAM_BOT_TOKEN").unwrap_or_default().trim().to_string();
        cfg.telegram_chat_id = std::env::var("TELEGRAM_CHAT_ID").unwrap_or_default().trim().to_string();

        cfg.validate()?;
        Ok(cfg)
    }

    /// Hard institutional pre-flight safety validator
    pub fn validate(&self) -> Result<(), String> {
        if self.symbols.is_empty() {
            return Err("Config Error: symbols list cannot be empty".into());
        }
        if !self.initial_capital_usd.is_finite() || self.initial_capital_usd <= 0.0 {
            return Err("Config Error: initial_capital_usd must be > 0".into());
        }
        if self.risk.max_daily_loss_pct <= 0.0 || self.risk.max_daily_loss_pct > 20.0 {
            return Err("Config Error: max_daily_loss_pct must be in (0, 20]".into());
        }
        if self.risk.max_leverage < 1.0 || self.risk.max_leverage > 20.0 {
            return Err("Config Error: max_leverage must be in [1.0, 20.0]".into());
        }
        if self.risk.max_position_usd <= 0.0
            || self.risk.max_position_usd > (self.initial_capital_usd * self.risk.max_leverage)
        {
            return Err("Config Error: max_position_usd exceeds capital * leverage".into());
        }
        if self.risk.stale_feed_ms < 500 || self.risk.stale_feed_ms > 10_000 {
            return Err("Config Error: stale_feed_ms must be between 500ms and 10000ms".into());
        }
        if self.risk.min_edge_to_fee_ratio < 1.5 {
            return Err("Config Error: min_edge_to_fee_ratio must be >= 1.5".into());
        }
        if self.mode == TradingMode::Live
            && (self.api_key.is_empty() || self.api_secret.is_empty())
        {
            return Err("FATAL SAFETY BLOCK: LIVE mode requires non-empty API_KEY and API_SECRET".into());
        }
        Ok(())
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config_passes_validation() {
        let cfg = BotConfig::default();
        assert!(cfg.validate().is_ok());
        assert_eq!(cfg.mode, TradingMode::Paper);
    }

    #[test]
    fn test_fee_and_jump_edge_math() {
        let cfg = BotConfig::default();
        let rt_maker = cfg.fees.round_trip_maker_bps();
        let rt_taker = cfg.fees.round_trip_taker_bps();
        let min_maker_edge = cfg.fees.min_required_edge_bps(3.0, true);
        let min_taker_edge = cfg.fees.min_required_edge_bps(3.0, false);

        assert!((rt_maker - 5.5).abs() < 1e-9);
        assert!((rt_taker - 13.0).abs() < 1e-9);
        assert!((min_maker_edge - 16.5).abs() < 1e-9);
        assert!((min_taker_edge - 39.0).abs() < 1e-9);
    }

    #[test]
    fn test_live_mode_blocks_empty_api_keys() {
        let mut cfg = BotConfig::default();
        cfg.mode = TradingMode::Live;
        assert!(cfg.validate().is_err());

        cfg.api_key = "test_key".to_string();
        cfg.api_secret = "test_secret".to_string();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_blocks_excessive_position_and_nan_capital() {
        let mut cfg = BotConfig::default();
        cfg.risk.max_position_usd = 500.0; // Exceeds 100 * 3x = 300
        assert!(cfg.validate().is_err());

        let mut nan_cfg = BotConfig::default();
        nan_cfg.initial_capital_usd = f64::NAN;
        assert!(nan_cfg.validate().is_err());
    }

    #[test]
    fn test_blocks_unsafe_stale_feed_and_daily_loss() {
        let mut cfg = BotConfig::default();
        cfg.risk.stale_feed_ms = 100; // Too low (< 500ms)
        assert!(cfg.validate().is_err());

        let mut loss_cfg = BotConfig::default();
        loss_cfg.risk.max_daily_loss_pct = 25.0; // Exceeds 20% hard cap
        assert!(loss_cfg.validate().is_err());
    }

    #[test]
    fn test_exchange_endpoints_valid() {
        for ex in [ExchangeId::DeltaExchange, ExchangeId::Bybit, ExchangeId::Binance] {
            assert!(ex.ws_url().starts_with("wss://"));
            assert!(ex.rest_url().starts_with("https://"));
        }
    }
}
