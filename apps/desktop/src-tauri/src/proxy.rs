//! The HTTP boundary between the webview and `ragcore`.
//!
//! Every call from the UI goes through here so the session token and the port
//! stay on the Rust side. Streaming answers arrive as SSE and leave as frames
//! on a Tauri `Channel`.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tauri::ipc::Channel;
use tauri::State;
use tokio::sync::Notify;

use crate::AppState;

/// Plain calls only. Streams have no overall deadline: a deep conversion can run
/// for many minutes, and a timeout on the client would cut it off mid-body.
const REQUEST_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(300);

/// An error with its causes: reqwest's top line ("error decoding response
/// body") hides the part that explains it ("operation timed out").
fn describe(err: &dyn std::error::Error) -> String {
    let mut text = err.to_string();
    let mut source = err.source();
    while let Some(cause) = source {
        text.push_str(": ");
        text.push_str(&cause.to_string());
        source = cause.source();
    }
    text
}

/// An error the UI can translate: a stable `code`, the English `message`, and
/// the values the sentence interpolates. Plain strings still work; the UI falls
/// back to showing them as they are.
fn shell_error(code: &str, message: String, params: Value) -> String {
    serde_json::json!({ "code": code, "message": message, "params": params }).to_string()
}

/// ragcore's coded error body (`{detail, code, params}`) in the UI's shape.
/// `None` for anything without a string `code`, which keeps the old text path.
fn coded_error(body: &Value) -> Option<String> {
    let code = body.get("code")?.as_str()?;
    let message = body.get("detail").and_then(Value::as_str).unwrap_or(code);
    let params = body.get("params").cloned().unwrap_or_else(|| serde_json::json!({}));
    Some(shell_error(code, message.to_string(), params))
}

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
    let label = format!("{} {}", req.method.to_uppercase(), req.path);
    let result = forward(&state, req).await;
    if let Err(err) = &result {
        state.supervisor.log(format!("[shell] {label} failed: {err}"));
    }
    result
}

/// Paths are appended to the sidecar's base URL; one that doesn't start with
/// `/` (e.g. `@evil.host/x`) would change the host and leak the bearer token.
fn check_path(path: &str) -> Result<(), String> {
    if path.starts_with('/') && !path.starts_with("//") {
        Ok(())
    } else {
        Err(shell_error(
            "invalid_path",
            format!("invalid path {path}"),
            serde_json::json!({ "path": path }),
        ))
    }
}

async fn forward(state: &State<'_, AppState>, req: ApiRequest) -> Result<Value, String> {
    check_path(&req.path)?;
    state.supervisor.wait_ready().await;
    let url = format!("{}{}", state.supervisor.base_url(), req.path);
    let method = reqwest::Method::from_bytes(req.method.to_uppercase().as_bytes())
        .map_err(|_| {
            shell_error(
                "unsupported_method",
                format!("unsupported method {}", req.method),
                serde_json::json!({ "method": req.method }),
            )
        })?;

    let mut builder = state
        .http
        .request(method, &url)
        .bearer_auth(&state.supervisor.token)
        .timeout(REQUEST_TIMEOUT);
    if let Some(body) = req.body {
        builder = builder.json(&body);
    }

    let response = builder
        .send()
        .await
        .map_err(|e| {
            let reason = describe(&e);
            shell_error(
                "ragcore_unreachable",
                format!("ragcore unreachable: {reason}"),
                serde_json::json!({ "reason": reason }),
            )
        })?;

    let status = response.status();
    let text = response.text().await.unwrap_or_default();

    if !status.is_success() {
        if let Some(coded) = serde_json::from_str::<Value>(&text)
            .ok()
            .and_then(|body| coded_error(&body))
        {
            return Err(coded);
        }
        let detail = serde_json::from_str::<Value>(&text)
            .ok()
            .and_then(|v| v.get("detail").and_then(|d| d.as_str().map(String::from)))
            .unwrap_or_else(|| text.clone());
        return Err(format!("{}: {detail}", status.as_u16()));
    }

    if text.is_empty() {
        return Ok(Value::Null);
    }
    serde_json::from_str(&text).map_err(|e| {
        shell_error(
            "invalid_response",
            format!("invalid JSON from ragcore: {e}"),
            serde_json::json!({ "reason": e.to_string() }),
        )
    })
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
    check_path(&path)?;
    let http_method = reqwest::Method::from_bytes(method.to_uppercase().as_bytes())
        .map_err(|_| {
            shell_error(
                "unsupported_method",
                format!("unsupported method {method}"),
                serde_json::json!({ "method": method }),
            )
        })?;

    // Registered before any await so a cancel that lands while we wait for the
    // sidecar is not lost; the guard removes it on every exit path.
    let notify = Arc::new(Notify::new());
    state
        .cancels
        .lock()
        .expect("cancel set poisoned")
        .insert(stream_id.clone(), Arc::clone(&notify));
    let _registration = Registration {
        set: Arc::clone(&state.cancels),
        id: stream_id,
    };

    tokio::select! {
        _ = state.supervisor.wait_ready() => {}
        _ = notify.notified() => {
            let _ = channel.send(StreamFrame::Closed { reason: "cancelled".to_string() });
            return Ok(());
        }
    }
    let url = format!("{}{}", state.supervisor.base_url(), path);

    let mut builder = state
        .http
        .request(http_method, &url)
        .bearer_auth(&state.supervisor.token)
        .header("Accept", "text/event-stream");
    if let Some(body) = body {
        builder = builder.json(&body);
    }

    let label = format!("{} {path}", method.to_uppercase());
    let started = std::time::Instant::now();
    state.supervisor.log(format!("[shell] stream {label} opened"));

    let sent = tokio::select! {
        sent = builder.send() => sent,
        _ = notify.notified() => {
            let _ = channel.send(StreamFrame::Closed { reason: "cancelled".to_string() });
            return Ok(());
        }
    };
    let response = match sent {
        Ok(response) => response,
        Err(err) => {
            let reason = describe(&err);
            let message = shell_error(
                "ragcore_unreachable",
                format!("ragcore unreachable: {reason}"),
                serde_json::json!({ "reason": reason }),
            );
            state.supervisor.log(format!("[shell] stream {label} failed: {message}"));
            let _ = channel.send(StreamFrame::Failed { message });
            return Ok(());
        }
    };

    if !response.status().is_success() {
        let status = response.status().as_u16();
        let text = response.text().await.unwrap_or_default();
        let message = serde_json::from_str::<Value>(&text)
            .ok()
            .and_then(|body| coded_error(&body))
            .unwrap_or_else(|| format!("{status}: {text}"));
        state.supervisor.log(format!("[shell] stream {label} failed: {message}"));
        let _ = channel.send(StreamFrame::Failed { message });
        return Ok(());
    }

    let mut stream = response.bytes_stream();
    let mut buffer: Vec<u8> = Vec::new();
    let mut reason = "complete";

    loop {
        // Cancel must interrupt an idle stream, not wait for the next chunk.
        let chunk = tokio::select! {
            chunk = stream.next() => chunk,
            _ = notify.notified() => {
                reason = "cancelled";
                break;
            }
        };
        let Some(chunk) = chunk else { break };

        let bytes = match chunk {
            Ok(bytes) => bytes,
            Err(err) => {
                let why = describe(&err);
                let message = shell_error(
                    "stream_broken",
                    format!("stream broken: {why}"),
                    serde_json::json!({ "reason": why }),
                );
                state.supervisor.log(format!(
                    "[shell] stream {label} broken after {:.0}s: {message}",
                    started.elapsed().as_secs_f64()
                ));
                let _ = channel.send(StreamFrame::Failed { message });
                reason = "failed";
                break;
            }
        };
        buffer.extend_from_slice(&bytes);

        // Frames are separated by a blank line; anything after the last one is
        // a partial frame and stays in the buffer.
        // Split on bytes and decode whole frames only: a multi-byte character
        // can straddle two network chunks.
        while let Some(split) = buffer.windows(2).position(|w| w == b"\n\n") {
            let raw = String::from_utf8_lossy(&buffer[..split]).into_owned();
            buffer.drain(..split + 2);
            if let Some(frame) = parse_frame(&raw) {
                let _ = channel.send(frame);
            }
        }
    }

    state.supervisor.log(format!(
        "[shell] stream {label} closed ({reason}) after {:.0}s",
        started.elapsed().as_secs_f64()
    ));
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
    // No entry means the stream already ended (or never started): nothing to
    // stop, and nothing to record, so the set cannot grow.
    let notify = state
        .cancels
        .lock()
        .expect("cancel set poisoned")
        .get(&stream_id)
        .cloned();
    if let Some(notify) = notify {
        notify.notify_one();
    }

    if let Some(path) = cancel_path {
        check_path(&path)?;
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

pub type CancelSet = Arc<Mutex<HashMap<String, Arc<Notify>>>>;

/// Removes a stream's cancel entry when `api_stream` exits, however it exits.
struct Registration {
    set: CancelSet,
    id: String,
}

impl Drop for Registration {
    fn drop(&mut self) {
        if let Ok(mut set) = self.set.lock() {
            set.remove(&self.id);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn event(raw: &str) -> (String, Value) {
        match parse_frame(raw) {
            Some(StreamFrame::Event { event, data }) => (event, data),
            other => panic!("expected an Event frame, got {other:?}"),
        }
    }

    #[test]
    fn an_event_and_data_pair_becomes_a_typed_frame() {
        let (name, data) = event("event: token\ndata: {\"text\":\"hello\"}");

        assert_eq!(name, "token");
        assert_eq!(data["text"], "hello");
    }

    #[test]
    fn a_frame_without_an_event_line_defaults_to_message() {
        assert_eq!(event("data: {\"text\":\"hi\"}").0, "message");
    }

    #[test]
    fn whitespace_after_the_field_name_is_trimmed() {
        let (name, data) = event("event:   token   \ndata:   {\"text\":\"hi\"}");

        assert_eq!(name, "token");
        assert_eq!(data["text"], "hi");
    }

    #[test]
    fn a_comment_only_frame_carries_no_data() {
        // The heartbeat on /jobs/stream. Forwarding it would confuse the UI.
        assert!(parse_frame(": heartbeat").is_none());
    }

    #[test]
    fn an_event_line_with_no_data_is_dropped() {
        assert!(parse_frame("event: token").is_none());
    }

    #[test]
    fn an_empty_frame_is_dropped() {
        assert!(parse_frame("").is_none());
    }

    #[test]
    fn several_data_lines_are_rejoined_with_newlines() {
        let (_, data) = event("event: token\ndata: {\"text\":\ndata: \"hi\"}");

        assert_eq!(data["text"], "hi");
    }

    #[test]
    fn data_that_is_not_json_is_forwarded_as_a_string() {
        let (name, data) = event("event: note\ndata: plain text");

        assert_eq!(name, "note");
        assert_eq!(data, Value::String("plain text".into()));
    }

    #[test]
    fn leading_whitespace_inside_the_payload_survives() {
        // Token frames carry the space between words; trimming it would glue
        // the answer together.
        let (_, data) = event("event: token\ndata: {\"text\":\" world\"}");

        assert_eq!(data["text"], " world");
    }

    #[test]
    fn the_error_frame_is_an_ordinary_event_not_a_transport_failure() {
        let (name, data) = event("event: error\ndata: {\"message\":\"boom\",\"retryable\":true}");

        assert_eq!(name, "error");
        assert_eq!(data["retryable"], true);
    }

    #[test]
    fn frames_serialise_with_a_kind_tag_the_ui_can_switch_on() {
        let event = serde_json::to_value(StreamFrame::Event {
            event: "token".into(),
            data: Value::Null,
        })
        .unwrap();
        let closed = serde_json::to_value(StreamFrame::Closed {
            reason: "cancelled".into(),
        })
        .unwrap();
        let failed = serde_json::to_value(StreamFrame::Failed {
            message: "ragcore unreachable".into(),
        })
        .unwrap();

        assert_eq!(event["kind"], "event");
        assert_eq!(closed["kind"], "closed");
        assert_eq!(failed["kind"], "failed");
        assert_eq!(failed["message"], "ragcore unreachable");
    }

    #[test]
    fn an_api_request_body_is_optional() {
        let without: ApiRequest = serde_json::from_str(r#"{"method":"GET","path":"/health"}"#)
            .expect("body must be optional");

        assert!(without.body.is_none());
        assert_eq!(without.path, "/health");
    }

    #[test]
    fn a_finished_stream_leaves_nothing_in_the_cancel_set() {
        let set: CancelSet = Arc::default();
        set.lock().unwrap().insert("s_1".into(), Arc::new(Notify::new()));

        drop(Registration { set: Arc::clone(&set), id: "s_1".into() });

        assert!(set.lock().unwrap().is_empty());
    }

    #[test]
    fn a_shell_error_is_json_with_a_code_and_params() {
        let text = shell_error(
            "invalid_path",
            "invalid path x".into(),
            serde_json::json!({ "path": "x" }),
        );
        let parsed: Value = serde_json::from_str(&text).unwrap();

        assert_eq!(parsed["code"], "invalid_path");
        assert_eq!(parsed["message"], "invalid path x");
        assert_eq!(parsed["params"]["path"], "x");
    }

    #[test]
    fn a_coded_ragcore_body_becomes_the_ui_error_shape() {
        let body = serde_json::json!({
            "detail": "model not installed",
            "code": "model_not_installed",
            "params": { "id": "m" }
        });
        let parsed: Value = serde_json::from_str(&coded_error(&body).unwrap()).unwrap();

        assert_eq!(parsed["code"], "model_not_installed");
        assert_eq!(parsed["message"], "model not installed");
        assert_eq!(parsed["params"]["id"], "m");
    }

    #[test]
    fn a_body_without_a_code_is_not_coded() {
        assert!(coded_error(&serde_json::json!({ "detail": "boom" })).is_none());
        assert!(coded_error(&serde_json::json!({ "detail": { "x": 1 } })).is_none());
    }

    #[test]
    fn paths_must_stay_on_the_sidecar_host() {
        assert!(check_path("/health").is_ok());
        assert!(check_path("@evil.host/x").is_err());
        assert!(check_path("//evil.host/x").is_err());
        assert!(check_path("").is_err());
    }

    #[test]
    fn a_character_split_across_chunks_survives() {
        let full = "data: \"è\"\n\n".as_bytes();
        let cut = full.iter().position(|&b| b == 0xC3).unwrap() + 1; // mid-"è"
        let mut buffer: Vec<u8> = Vec::new();
        let mut out = Vec::new();
        for chunk in [&full[..cut], &full[cut..]] {
            buffer.extend_from_slice(chunk);
            while let Some(split) = buffer.windows(2).position(|w| w == b"\n\n") {
                let raw = String::from_utf8_lossy(&buffer[..split]).into_owned();
                buffer.drain(..split + 2);
                out.push(event(&raw).1);
            }
        }

        assert_eq!(out, [Value::String("è".into())]);
    }

    #[test]
    fn a_buffer_yields_whole_frames_and_keeps_the_partial_tail() {
        // Mirrors the split loop in api_stream: a chunk boundary must not eat
        // half a frame.
        let mut buffer =
            String::from("event: a\ndata: 1\n\nevent: b\ndata: 2\n\nevent: c\ndata: ");
        let mut names = Vec::new();

        while let Some(split) = buffer.find("\n\n") {
            let raw = buffer[..split].to_string();
            buffer.drain(..split + 2);
            if let Some(StreamFrame::Event { event, .. }) = parse_frame(&raw) {
                names.push(event);
            }
        }

        assert_eq!(names, ["a", "b"]);
        assert_eq!(buffer, "event: c\ndata: ");
    }
}
