mod config;

fn main() {
    let cfg = config::BotConfig::default();
    println!("HMTROBOT_LAB v2 | {:?}", cfg);
}
