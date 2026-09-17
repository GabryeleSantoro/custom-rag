//! API keys live in the OS keychain and nowhere else.
//!
//! They are never written to the index, to settings, or to logs, and they never
//! cross into the webview: the UI can ask whether a key exists and can replace
//! it, but cannot read it back.

use keyring::Entry;

const SERVICE: &str = "com.customrag.desktop";

fn entry(connection_id: &str) -> Result<Entry, String> {
    Entry::new(SERVICE, connection_id).map_err(|e| e.to_string())
}

pub fn set(connection_id: &str, secret: &str) -> Result<(), String> {
    entry(connection_id)?
        .set_password(secret)
        .map_err(|e| e.to_string())
}

pub fn get(connection_id: &str) -> Result<Option<String>, String> {
    match entry(connection_id)?.get_password() {
        Ok(secret) => Ok(Some(secret)),
        Err(keyring::Error::NoEntry) => Ok(None),
        Err(e) => Err(e.to_string()),
    }
}

pub fn delete(connection_id: &str) -> Result<(), String> {
    match entry(connection_id)?.delete_credential() {
        Ok(()) | Err(keyring::Error::NoEntry) => Ok(()),
        Err(e) => Err(e.to_string()),
    }
}
