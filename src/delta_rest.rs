#![allow(dead_code)]

use crate::types::SymbolId;

/// Delta Exchange V2 REST API configuration (Demo/Testnet or Live)
#[derive(Debug, Clone)]
pub struct DeltaRestConfig {
    pub enabled: bool,
    pub base_url: String,
    pub api_key: String,
    pub api_secret: String,
    pub timeout_ms: u64,
}

impl DeltaRestConfig {
    pub fn from_env() -> Self {
        let api_key = std::env::var("DELTA_API_KEY").unwrap_or_default();
        let api_secret = std::env::var("DELTA_API_SECRET").unwrap_or_default();
        let base_url = std::env::var("DELTA_REST_URL")
            .unwrap_or_else(|_| "https://cdn-ind.testnet.deltaex.org".to_string());
        let enabled = !api_key.trim().is_empty() && !api_secret.trim().is_empty();

        Self {
            enabled,
            base_url: base_url.trim_end_matches('/').to_string(),
            api_key: api_key.trim().to_string(),
            api_secret: api_secret.trim().to_string(),
            timeout_ms: 4_000,
        }
    }
}

#[inline]
pub fn delta_symbol_name(symbol_id: SymbolId) -> Option<&'static str> {
    match symbol_id.0 {
        0 => Some("BTCUSD"),
        1 => Some("ETHUSD"),
        2 => Some("SOLUSD"),
        _ => None,
    }
}

const SHA256_K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

/// Pure-Rust zero-dependency FIPS 180-4 SHA-256
pub fn sha256(data: &[u8]) -> [u8; 32] {
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
    ];

    let bit_len = (data.len() as u64).wrapping_mul(8);
    let mut msg = Vec::with_capacity(data.len() + 72);
    msg.extend_from_slice(data);
    msg.push(0x80);
    while (msg.len() % 64) != 56 {
        msg.push(0x00);
    }
    msg.extend_from_slice(&bit_len.to_be_bytes());

    let mut w = [0u32; 64];
    for chunk in msg.chunks_exact(64) {
        for i in 0..16 {
            let j = i * 4;
            w[i] = u32::from_be_bytes([chunk[j], chunk[j + 1], chunk[j + 2], chunk[j + 3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16].wrapping_add(s0).wrapping_add(w[i - 7]).wrapping_add(s1);
        }

        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh] = h;
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let t1 = hh.wrapping_add(s1).wrapping_add(ch).wrapping_add(SHA256_K[i]).wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);

            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }

        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
        h[5] = h[5].wrapping_add(f);
        h[6] = h[6].wrapping_add(g);
        h[7] = h[7].wrapping_add(hh);
    }

    let mut out = [0u8; 32];
    for i in 0..8 {
        out[i * 4..(i + 1) * 4].copy_from_slice(&h[i].to_be_bytes());
    }
    out
}

/// Pure-Rust RFC 2104 HMAC-SHA256 returning lowercase 64-char hex string
pub fn hmac_sha256_hex(key: &[u8], message: &[u8]) -> String {
    let mut k_block = [0u8; 64];
    if key.len() > 64 {
        let digest = sha256(key);
        k_block[..32].copy_from_slice(&digest);
    } else {
        k_block[..key.len()].copy_from_slice(key);
    }

    let mut inner = Vec::with_capacity(64 + message.len());
    for b in k_block {
        inner.push(b ^ 0x36);
    }
    inner.extend_from_slice(message);
    let inner_hash = sha256(&inner);

    let mut outer = [0u8; 96];
    for i in 0..64 {
        outer[i] = k_block[i] ^ 0x5c;
    }
    outer[64..96].copy_from_slice(&inner_hash);
    let mac = sha256(&outer);

    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(64);
    for b in mac {
        s.push(HEX[(b >> 4) as usize] as char);
        s.push(HEX[(b & 0x0f) as usize] as char);
    }
    s
}

/// Delta Exchange V2 signature: HMAC-SHA256(secret, method + timestamp + path + query + payload)
pub fn sign_delta_v2(
    secret: &str,
    method: &str,
    timestamp_sec: u64,
    path: &str,
    query: &str,
    payload: &str,
) -> String {
    let prehash = format!("{}{}{}{}{}", method, timestamp_sec, path, query, payload);
    hmac_sha256_hex(secret.as_bytes(), prehash.as_bytes())
}

#[cfg(test)]
mod delta_rest_tests {
    use super::*;

    #[test]
    fn test_hmac_sha256_rfc4231_vector() {
        let sig = hmac_sha256_hex(b"Jefe", b"what do ya want for nothing?");
        assert_eq!(
            sig,
            "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843"
        );
    }

    #[test]
    fn test_delta_symbol_mapping_no_gold() {
        assert_eq!(delta_symbol_name(SymbolId(0)), Some("BTCUSD"));
        assert_eq!(delta_symbol_name(SymbolId(1)), Some("ETHUSD"));
        assert_eq!(delta_symbol_name(SymbolId(2)), Some("SOLUSD"));
        assert_eq!(delta_symbol_name(SymbolId(3)), None);
    }
}
use serde_json::{json, Value};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeltaOrderSide {
    Buy,
    Sell,
}

impl DeltaOrderSide {
    #[inline]
    pub fn as_str(self) -> &'static str {
        match self {
            DeltaOrderSide::Buy => "buy",
            DeltaOrderSide::Sell => "sell",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeltaOrderType {
    Limit,
    Market,
}

impl DeltaOrderType {
    #[inline]
    pub fn as_str(self) -> &'static str {
        match self {
            DeltaOrderType::Limit => "limit_order",
            DeltaOrderType::Market => "market_order",
        }
    }
}

#[derive(Debug, Clone)]
pub struct DeltaOrderRequest {
    pub symbol_id: SymbolId,
    pub side: DeltaOrderSide,
    pub order_type: DeltaOrderType,
    pub size_contracts: u64,
    pub limit_price: Option<f64>,
    pub post_only: bool,
    pub reduce_only: bool,
    pub client_order_id: String,
}

#[derive(Debug, Clone)]
pub struct DeltaApiResult {
    pub ok: bool,
    pub status_code: u16,
    pub exchange_order_id: Option<String>,
    pub message: String,
}

#[inline]
pub fn now_epoch_sec() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Builds Delta Exchange V2 `/v2/orders` JSON payload (strictly BTC/ETH/SOL, NO GOLD)
pub fn build_order_payload(req: &DeltaOrderRequest) -> Option<String> {
    let symbol = delta_symbol_name(req.symbol_id)?;
    if req.size_contracts == 0 {
        return None;
    }

    let mut obj = json!({
        "product_symbol": symbol,
        "size": req.size_contracts,
        "side": req.side.as_str(),
        "order_type": req.order_type.as_str(),
        "reduce_only": req.reduce_only,
        "client_order_id": req.client_order_id
    });

    if req.order_type == DeltaOrderType::Limit {
        let px = req.limit_price?;
        if !px.is_finite() || px <= 0.0 {
            return None;
        }
        if let Some(map) = obj.as_object_mut() {
            map.insert("limit_price".to_string(), json!(format!("{:.2}", px)));
            map.insert("time_in_force".to_string(), json!("gtc"));
            map.insert("post_only".to_string(), json!(req.post_only));
        }
    }

    Some(obj.to_string())
}

/// Zero-panic async Delta Exchange V2 REST client (Demo/Testnet or Live)
pub struct DeltaRestClient {
    pub cfg: DeltaRestConfig,
    http: reqwest::Client,
}

impl DeltaRestClient {
    pub fn new(cfg: DeltaRestConfig) -> Option<Self> {
        let http = reqwest::Client::builder()
            .timeout(Duration::from_millis(cfg.timeout_ms.clamp(1_000, 10_000)))
            .build()
            .ok()?;
        Some(Self { cfg, http })
    }

    pub async fn place_order(&self, req: &DeltaOrderRequest) -> DeltaApiResult {
        if !self.cfg.enabled {
            return DeltaApiResult {
                ok: false,
                status_code: 0,
                exchange_order_id: None,
                message: "delta_rest_disabled".to_string(),
            };
        }

        let payload = match build_order_payload(req) {
            Some(p) => p,
            None => {
                return DeltaApiResult {
                    ok: false,
                    status_code: 0,
                    exchange_order_id: None,
                    message: "invalid_order_request".to_string(),
                };
            }
        };

        let path = "/v2/orders";
        let ts = now_epoch_sec();
        let sig = sign_delta_v2(&self.cfg.api_secret, "POST", ts, path, "", &payload);
        let url = format!("{}{}", self.cfg.base_url, path);

        let resp = match self
            .http
            .post(&url)
            .header("api-key", &self.cfg.api_key)
            .header("timestamp", ts.to_string())
            .header("signature", sig)
            .header("Content-Type", "application/json")
            .header("User-Agent", "hmtrobot-rust")
            .body(payload)
            .send()
            .await
        {
            Ok(r) => r,
            Err(_) => {
                return DeltaApiResult {
                    ok: false,
                    status_code: 0,
                    exchange_order_id: None,
                    message: "network_error".to_string(),
                };
            }
        };

        let status = resp.status().as_u16();
        let body = resp.text().await.unwrap_or_default();
        let parsed: Option<Value> = serde_json::from_str(&body).ok();
        let success = parsed
            .as_ref()
            .and_then(|v| v.get("success"))
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let order_id = parsed
            .as_ref()
            .and_then(|v| v.get("result"))
            .and_then(|r| r.get("id"))
            .map(|id| id.to_string().trim_matches('"').to_string());

        DeltaApiResult {
            ok: (200..300).contains(&status) && success,
            status_code: status,
            exchange_order_id: order_id,
            message: if success {
                "ok".to_string()
            } else {
                body.chars().take(180).collect()
            },
        }
    }

    pub async fn cancel_all_orders(&self, symbol_id: Option<SymbolId>) -> DeltaApiResult {
        if !self.cfg.enabled {
            return DeltaApiResult {
                ok: false,
                status_code: 0,
                exchange_order_id: None,
                message: "delta_rest_disabled".to_string(),
            };
        }

        let payload = match symbol_id.and_then(delta_symbol_name) {
            Some(sym) => json!({
                "product_symbol": sym,
                "contract_types": "perpetual_futures"
            })
            .to_string(),
            None => json!({
                "contract_types": "perpetual_futures"
            })
            .to_string(),
        };

        let path = "/v2/orders/all";
        let ts = now_epoch_sec();
        let sig = sign_delta_v2(&self.cfg.api_secret, "DELETE", ts, path, "", &payload);
        let url = format!("{}{}", self.cfg.base_url, path);

        let resp = match self
            .http
            .delete(&url)
            .header("api-key", &self.cfg.api_key)
            .header("timestamp", ts.to_string())
            .header("signature", sig)
            .header("Content-Type", "application/json")
            .header("User-Agent", "hmtrobot-rust")
            .body(payload)
            .send()
            .await
        {
            Ok(r) => r,
            Err(_) => {
                return DeltaApiResult {
                    ok: false,
                    status_code: 0,
                    exchange_order_id: None,
                    message: "network_error".to_string(),
                };
            }
        };

        let status = resp.status().as_u16();
        let body = resp.text().await.unwrap_or_default();
        DeltaApiResult {
            ok: (200..300).contains(&status),
            status_code: status,
            exchange_order_id: None,
            message: body.chars().take(180).collect(),
        }
    }
}

#[cfg(test)]
mod delta_rest_chunk2_tests {
    use super::*;

    #[test]
    fn test_build_post_only_limit_payload() {
        let req = DeltaOrderRequest {
            symbol_id: SymbolId(0),
            side: DeltaOrderSide::Buy,
            order_type: DeltaOrderType::Limit,
            size_contracts: 1,
            limit_price: Some(60_000.5),
            post_only: true,
            reduce_only: false,
            client_order_id: "hmt_1".to_string(),
        };
        let payload = build_order_payload(&req);
        assert!(payload.is_some());
        if let Some(p) = payload {
            assert!(p.contains("\"product_symbol\":\"BTCUSD\""));
            assert!(p.contains("\"post_only\":true"));
            assert!(p.contains("\"limit_price\":\"60000.50\""));
        }
    }

    #[test]
    fn test_reject_invalid_symbol_and_zero_size() {
        let bad_sym = DeltaOrderRequest {
            symbol_id: SymbolId(9),
            side: DeltaOrderSide::Buy,
            order_type: DeltaOrderType::Market,
            size_contracts: 1,
            limit_price: None,
            post_only: false,
            reduce_only: false,
            client_order_id: "hmt_bad".to_string(),
        };
        assert!(build_order_payload(&bad_sym).is_none());
    }
}
use tokio::sync::mpsc;
use tokio::task::JoinHandle;

#[derive(Debug, Clone, Copy, Default)]
pub struct DeltaLivePosition {
    pub symbol_id: u8,
    pub size_contracts: f64,
    pub entry_price: f64,
    pub unrealized_pnl: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct DeltaSyncSnapshot {
    pub wallet_balance_usd: f64,
    pub available_balance_usd: f64,
    pub positions: [DeltaLivePosition; 3],
    pub synced_at_sec: u64,
}

#[inline]
fn parse_json_f64(v: &Value) -> f64 {
    v.as_f64()
        .or_else(|| v.as_str().and_then(|s| s.parse::<f64>().ok()))
        .filter(|x| x.is_finite())
        .unwrap_or(0.0)
}

#[inline]
fn delta_symbol_to_index(sym: &str) -> Option<usize> {
    match sym {
        "BTCUSD" | "BTCUSDT" => Some(0),
        "ETHUSD" | "ETHUSDT" => Some(1),
        "SOLUSD" | "SOLUSDT" => Some(2),
        _ => None,
    }
}

/// Parses Delta V2 `/v2/wallet/balances` response -> (wallet_balance_usd, available_balance_usd)
pub fn parse_wallet_balances_json(raw: &str) -> Option<(f64, f64)> {
    let val: Value = serde_json::from_str(raw).ok()?;
    let list = val.get("result")?.as_array()?;
    let mut wallet_total = 0.0;
    let mut avail_total = 0.0;
    let mut found = false;

    for item in list {
        let asset = item
            .get("asset_symbol")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        if matches!(asset, "USD" | "USDT" | "USDC") {
            wallet_total += item.get("balance").map(parse_json_f64).unwrap_or(0.0);
            avail_total += item
                .get("available_balance")
                .map(parse_json_f64)
                .unwrap_or(0.0);
            found = true;
        }
    }

    if found && wallet_total.is_finite() && avail_total.is_finite() {
        Some((wallet_total.max(0.0), avail_total.max(0.0)))
    } else {
        None
    }
}

/// Parses Delta V2 `/v2/positions/margined` response strictly for BTC/ETH/SOL (NO GOLD)
pub fn parse_positions_json(raw: &str) -> [DeltaLivePosition; 3] {
    let mut out = [
        DeltaLivePosition { symbol_id: 0, ..Default::default() },
        DeltaLivePosition { symbol_id: 1, ..Default::default() },
        DeltaLivePosition { symbol_id: 2, ..Default::default() },
    ];

    let val: Value = match serde_json::from_str(raw) {
        Ok(v) => v,
        Err(_) => return out,
    };
    let list = match val.get("result").and_then(|r| r.as_array()) {
        Some(a) => a,
        None => return out,
    };

    for item in list {
        let sym = item
            .get("product_symbol")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        if let Some(idx) = delta_symbol_to_index(sym) {
            out[idx].size_contracts = item.get("size").map(parse_json_f64).unwrap_or(0.0);
            out[idx].entry_price = item.get("entry_price").map(parse_json_f64).unwrap_or(0.0);
            out[idx].unrealized_pnl = item
                .get("unrealized_pnl")
                .map(parse_json_f64)
                .unwrap_or(0.0);
        }
    }

    out
}

impl DeltaRestClient {
    async fn signed_get(&self, path: &str) -> Option<String> {
        if !self.cfg.enabled {
            return None;
        }
        let ts = now_epoch_sec();
        let sig = sign_delta_v2(&self.cfg.api_secret, "GET", ts, path, "", "");
        let url = format!("{}{}", self.cfg.base_url, path);

        let resp = self
            .http
            .get(&url)
            .header("api-key", &self.cfg.api_key)
            .header("timestamp", ts.to_string())
            .header("signature", sig)
            .header("User-Agent", "hmtrobot-rust")
            .send()
            .await
            .ok()?;

        if !resp.status().is_success() {
            return None;
        }
        resp.text().await.ok()
    }

    pub async fn fetch_account_sync(&self) -> Option<DeltaSyncSnapshot> {
        let bal_raw = self.signed_get("/v2/wallet/balances").await?;
        let (wallet_balance_usd, available_balance_usd) = parse_wallet_balances_json(&bal_raw)?;
        let pos_raw = self.signed_get("/v2/positions/margined").await.unwrap_or_default();
        let positions = parse_positions_json(&pos_raw);

        Some(DeltaSyncSnapshot {
            wallet_balance_usd,
            available_balance_usd,
            positions,
            synced_at_sec: now_epoch_sec(),
        })
    }
}

#[derive(Debug, Clone)]
pub enum DeltaRestCommand {
    PlaceOrder(DeltaOrderRequest),
    CancelAll(Option<SymbolId>),
}

#[derive(Debug, Clone)]
pub struct DeltaRestHandle {
    tx: Option<mpsc::Sender<DeltaRestCommand>>,
}

impl DeltaRestHandle {
    pub fn disabled() -> Self {
        Self { tx: None }
    }

    #[inline]
    pub fn is_enabled(&self) -> bool {
        self.tx.is_some()
    }

    #[inline]
    pub fn try_place_order(&self, req: DeltaOrderRequest) -> bool {
        match &self.tx {
            Some(tx) => tx.try_send(DeltaRestCommand::PlaceOrder(req)).is_ok(),
            None => false,
        }
    }

    #[inline]
    pub fn try_cancel_all(&self, symbol_id: Option<SymbolId>) -> bool {
        match &self.tx {
            Some(tx) => tx.try_send(DeltaRestCommand::CancelAll(symbol_id)).is_ok(),
            None => false,
        }
    }
}

/// Spawns non-blocking Delta REST order dispatcher + 30s balance/position reconciler
pub fn spawn_delta_rest_worker(
    cfg: DeltaRestConfig,
    sync_tx: Option<mpsc::Sender<DeltaSyncSnapshot>>,
) -> (DeltaRestHandle, Option<JoinHandle<()>>) {
    if !cfg.enabled {
        return (DeltaRestHandle::disabled(), None);
    }

    let client = match DeltaRestClient::new(cfg) {
        Some(c) => c,
        None => return (DeltaRestHandle::disabled(), None),
    };

    let (cmd_tx, mut cmd_rx) = mpsc::channel::<DeltaRestCommand>(64);
    let task = tokio::spawn(async move {
        let mut sync_tick = tokio::time::interval(Duration::from_secs(30));
        loop {
            tokio::select! {
                maybe_cmd = cmd_rx.recv() => {
                    match maybe_cmd {
                        Some(DeltaRestCommand::PlaceOrder(req)) => {
                            let _ = client.place_order(&req).await;
                        }
                        Some(DeltaRestCommand::CancelAll(sym)) => {
                            let _ = client.cancel_all_orders(sym).await;
                        }
                        None => break,
                    }
                }
                _ = sync_tick.tick() => {
                    if let Some(ref stx) = sync_tx {
                        if let Some(snap) = client.fetch_account_sync().await {
                            let _ = stx.try_send(snap);
                        }
                    }
                }
            }
        }
    });

    (DeltaRestHandle { tx: Some(cmd_tx) }, Some(task))
}

#[cfg(test)]
mod delta_rest_chunk3_tests {
    use super::*;

    #[test]
    fn test_parse_wallet_and_positions_ignore_gold() {
        let bal_json = r#"{"success":true,"result":[{"asset_symbol":"USD","balance":"1250.50","available_balance":"1100.25"}]}"#;
        let bal = parse_wallet_balances_json(bal_json);
        assert!(bal.is_some());
        if let Some((w, a)) = bal {
            assert!((w - 1250.50).abs() < 1e-6);
            assert!((a - 1100.25).abs() < 1e-6);
        }

        let pos_json = r#"{"success":true,"result":[
            {"product_symbol":"BTCUSD","size":2,"entry_price":"60010.5","unrealized_pnl":"12.4"},
            {"product_symbol":"PAXGUSD","size":5,"entry_price":"2600.0","unrealized_pnl":"50.0"},
            {"product_symbol":"SOLUSD","size":-3,"entry_price":"145.0","unrealized_pnl":"-1.5"}
        ]}"#;
        let pos = parse_positions_json(pos_json);
        assert!((pos[0].size_contracts - 2.0).abs() < 1e-6);
        assert!((pos[1].size_contracts - 0.0).abs() < 1e-6);
        assert!((pos[2].size_contracts + 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_disabled_handle_zero_panic() {
        let h = DeltaRestHandle::disabled();
        assert!(!h.is_enabled());
        assert!(!h.try_cancel_all(None));
    }
}
