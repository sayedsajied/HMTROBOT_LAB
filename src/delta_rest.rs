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
