//! The HTTP boundary between the webview and `ragcore`.
//!
//! Every call from the UI goes through here so the session token and the port
//! stay on the Rust side. Streaming answers arrive as SSE and leave as frames
//! on a Tauri `Channel`.

use std::collections::HashSet;
use std::sync::{Arc, Mutex};

use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tauri::ipc::Channel;
use tauri::State;

use crate::AppState;

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum StreamFrame {
    /// One `event:`/`data:` pair from the sidecar, forwarded verbatim.
    Event { event: String, data: Value },
    /// The stream ended cleanly, or was cancelled from the UI.
    Closed { reason: String },
    /// The transport itself failed. Errors the sidecar reports arrive as an
    /// `Event` named "error" instead.
    Failed { message: String },
}

#[derive(Debug, Deserialize)]
pub struct ApiRequest {
    pub method: String,
    pub path: String,
    #[serde(default)]
    pub body: Option<Value>,
}

/// Non-streaming call. Returns the parsed JSON body, or an error string that is
/// safe to show in the UI.
#[tauri::command]
pub async fn api_request(state: State<'_, AppState>, req: ApiRequest) -> Result<Value, String> {
    let url = format!("{}{}", state.supervisor.base_url(), req.path);
    let method = reqwest::Method::from_bytes(req.method.to_uppercase().as_bytes())
        .map_err(|_| format!("unsupported method {}", req.method))?;

    let mut builder = state
        .http
        .request(method, &url)
        .bearer_auth(&state.supervisor.token);
    if let Some(body) = req.body {
        builder = builder.json(&body);
    }

    let response = builder
        .send()
        .await
        .map_err(|e| format!("ragcore unreachable: {e}"))?;

    let status = response.status();
    let text = response.text().await.unwrap_or_default();

    if !status.is_success() {
        let detail = serde_json::from_str::<Value>(&text)
            .ok()
            .and_then(|v| v.get("detail").and_then(|d| d.as_str().map(String::from)))
            .unwrap_or_else(|| text.clone());
        return Err(format!("{}: {detail}", status.as_u16()));
    }

    if text.is_empty() {
        return Ok(Value::Null);
    }
    serde_json::from_str(&text).map_err(|e| format!("invalid JSON from ragcore: {e}"))
}

/// Open an SSE stream and forward each frame to the channel.
///
/// `stream_id` is chosen by the caller so it can cancel before the first frame
/// arrives.
#[tauri::command]
pub async fn api_stream(
    state: State<'_, AppState>,
    stream_id: String,
    method: String,
    path: String,
    body: Option<Value>,
    channel: Channel<StreamFrame>,
) -> Result<(), String> {
    let url = format!("{}{}", state.supervisor.base_url(), path);
    let http_method = reqwest::Method::from_bytes(method.to_uppercase().as_bytes())
        .map_err(|_| format!("unsupported method {method}"))?;

    let cancels = Arc::clone(&state.cancels);
    cancels
        .lock()
        .expect("cancel set poisoned")
        .remove(&stream_id);

    let mut builder = state
        .http
        .request(http_method, &url)
        .bearer_auth(&state.supervisor.token)
        .header("Accept", "text/event-stream");
    if let Some(body) = body {
        builder = builder.json(&body);
    }

    let response = match builder.send().await {
        Ok(response) => response,
        Err(err) => {
            let _ = channel.send(StreamFrame::Failed {
                message: format!("ragcore unreachable: {err}"),
            });
            return Ok(());
        }
    };

    if !response.status().is_success() {
        let status = response.status().as_u16();
        let text = response.text().await.unwrap_or_default();
        let _ = channel.send(StreamFrame::Failed {
            message: format!("{status}: {text}"),
        });
        return Ok(());
    }

    let mut stream = response.bytes_stream();
    let mut buffer = String::new();
    let mut reason = "complete";

    while let Some(chunk) = stream.next().await {
        if cancels
            .lock()
            .expect("cancel set poisoned")
            .contains(&stream_id)
        {
            reason = "cancelled";
            break;
        }

        let bytes = match chunk {
            Ok(bytes) => bytes,
            Err(err) => {
                let _ = channel.send(StreamFrame::Failed {
                    message: format!("stream broken: {err}"),
                });
                reason = "failed";
                break;
            }
        };
        buffer.push_str(&String::from_utf8_lossy(&bytes));

        // Frames are separated by a blank line; anything after the last one is
        // a partial frame and stays in the buffer.
        while let Some(split) = buffer.find("\n\n") {
            let raw = buffer[..split].to_string();
            buffer.drain(..split + 2);
            if let Some(frame) = parse_frame(&raw) {
                let _ = channel.send(frame);
            }
        }
    }

    cancels
        .lock()
        .expect("cancel set poisoned")
        .remove(&stream_id);
    let _ = channel.send(StreamFrame::Closed {
        reason: reason.to_string(),
    });
    Ok(())
}

/// Stop forwarding, and tell the sidecar to stop generating.
#[tauri::command]
pub async fn api_cancel(
    state: State<'_, AppState>,
    stream_id: String,
    cancel_path: Option<String>,
) -> Result<(), String> {
    state
        .cancels
        .lock()
        .expect("cancel set poisoned")
        .insert(stream_id);

    if let Some(path) = cancel_path {
        let url = format!("{}{}", state.supervisor.base_url(), path);
        let _ = state
            .http
            .post(url)
            .bearer_auth(&state.supervisor.token)
            .send()
            .await;
    }
    Ok(())
}

fn parse_frame(raw: &str) -> Option<StreamFrame> {
    let mut event = "message".to_string();
    let mut data = String::new();

    for line in raw.lines() {
        if let Some(rest) = line.strip_prefix("event:") {
            event = rest.trim().to_string();
        } else if let Some(rest) = line.strip_prefix("data:") {
            if !data.is_empty() {
                data.push('\n');
            }
            data.push_str(rest.trim_start());
        }
        // Anything else is a comment (": heartbeat") and is ignored.
    }

    if data.is_empty() {
        return None;
    }
    let parsed = serde_json::from_str(&data).unwrap_or(Value::String(data));
    Some(StreamFrame::Event {
        event,
        data: parsed,
    })
}

pub type CancelSet = Arc<Mutex<HashSet<String>>>;
