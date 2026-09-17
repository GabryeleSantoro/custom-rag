//! The Rust shell.
//!
//! Responsibilities, in order of importance: keep the sidecars alive, keep the
//! session token and API keys out of the webview, and proxy everything the UI
//! asks for to `ragcore`.

mod hardware;
mod keychain;
mod proxy;
mod sidecars;

use std::collections::HashSet;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use tauri::{Manager, RunEvent};

use hardware::HardwareInfo;
use sidecars::{SidecarStatus, Supervisor};

pub struct AppState {
    pub supervisor: Arc<Supervisor>,
    pub http: reqwest::Client,
    pub cancels: proxy::CancelSet,
    pub hardware: HardwareInfo,
}

#[tauri::command]
fn hardware_info(state: tauri::State<'_, AppState>) -> HardwareInfo {
    state.hardware.clone()
}

#[tauri::command]
fn sidecar_status(state: tauri::State<'_, AppState>) -> Vec<SidecarStatus> {
    // Only ragcore is supervised so far. The model servers join this list
    // unchanged once they are spawned here too.
    vec![state.supervisor.status()]
}

#[tauri::command]
fn sidecar_logs(state: tauri::State<'_, AppState>) -> Vec<String> {
    state.supervisor.logs()
}

/// Kill the sidecar and let the supervisor restart it. Used by Diagnostics, and
/// the fastest way to recover from a wedged child.
#[tauri::command]
fn sidecar_restart(state: tauri::State<'_, AppState>) {
    state.supervisor.kill_child();
}

#[tauri::command]
fn keychain_set(connection_id: String, secret: String) -> Result<(), String> {
    keychain::set(&connection_id, &secret)
}

#[tauri::command]
fn keychain_has(connection_id: String) -> Result<bool, String> {
    keychain::get(&connection_id).map(|secret| secret.is_some())
}

#[tauri::command]
fn keychain_delete(connection_id: String) -> Result<(), String> {
    keychain::delete(&connection_id)
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let hardware = hardware::detect();
    let supervisor = Supervisor::new().expect("could not reserve a port for ragcore");
    let http = reqwest::Client::builder()
        .timeout(Duration::from_secs(300))
        .connect_timeout(Duration::from_secs(5))
        .build()
        .expect("could not build the HTTP client");

    let state = AppState {
        supervisor: Arc::clone(&supervisor),
        http: http.clone(),
        cancels: Arc::new(Mutex::new(HashSet::new())),
        hardware: hardware.clone(),
    };

    let app = tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_os::init())
        .manage(state)
        .setup(move |app| {
            supervisor.spawn(app.handle().clone(), hardware.clone(), http.clone());
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            hardware_info,
            sidecar_status,
            sidecar_logs,
            sidecar_restart,
            keychain_set,
            keychain_has,
            keychain_delete,
            proxy::api_request,
            proxy::api_stream,
            proxy::api_cancel,
        ])
        .build(tauri::generate_context!())
        .expect("error while building the application");

    app.run(|app_handle, event| {
        // Closing the window must not leave an orphaned Python process behind.
        if let RunEvent::ExitRequested { .. } | RunEvent::Exit = event {
            let state = app_handle.state::<AppState>();
            state.supervisor.request_shutdown();
            state.supervisor.kill_child();
        }
    });
}
