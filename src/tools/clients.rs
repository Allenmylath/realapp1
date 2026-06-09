//! Client management tool functions for the LLM function registry.
//!
//! Exposes three tools:
//!   - `add_client`        — insert a new client row
//!   - `get_client_by_name` — fuzzy search by name (ILIKE)
//!   - `list_clients`      — list all active clients
//!
//! Register all three with a single call to [`register_client_tools`].

use std::sync::Arc;

use serde_json::{json, Value};
use sqlx::PgPool;

use rustvani::adapters::schemas::FunctionSchema;
use rustvani::services::llm::FunctionRegistry;
use rustvani::services::llm::function_registry::ToolCallOutput;

// ---------------------------------------------------------------------------
// Schema definitions — what the LLM sees
// ---------------------------------------------------------------------------

/// Returns the `FunctionSchema` list for all client tools.
/// Pass these to `ToolsSchema::new(...)` when building context.
pub fn client_tool_schemas() -> Vec<FunctionSchema> {
    vec![
        FunctionSchema::new("add_client", "Add a new client to the database")
            .with_parameters(json!({
                "type": "object",
                "properties": {
                    "name":            { "type": "string",  "description": "Full name of the client" },
                    "phone":           { "type": "string",  "description": "Phone number" },
                    "email":           { "type": "string",  "description": "Email address" },
                    "notes":           { "type": "string",  "description": "Any notes about this client" },
                    "budget_min":      { "type": "integer", "description": "Minimum budget in INR (optional)" },
                    "budget_max":      { "type": "integer", "description": "Maximum budget in INR (optional)" },
                    "preferred_areas": { "type": "string",  "description": "Preferred areas or localities (optional)" }
                },
                "required": ["name", "phone", "email", "notes"]
            })),

        FunctionSchema::new(
            "get_client_by_name",
            "Search for clients by name. Uses fuzzy matching so partial names work.",
        )
        .with_parameters(json!({
            "type": "object",
            "properties": {
                "name": { "type": "string", "description": "Name or partial name to search for" }
            },
            "required": ["name"]
        })),

        FunctionSchema::new("list_clients", "List all active clients with their key details.")
            .with_parameters(json!({
                "type": "object",
                "properties": {},
                "required": []
            })),
    ]
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

/// Register all client tool handlers into `registry`.
///
/// Call this once in `main` after creating the registry:
/// ```rust,ignore
/// let pool = Arc::new(PgPool::connect(&db_url).await?);
/// register_client_tools(&mut registry, pool);
/// ```
pub fn register_client_tools(registry: &mut FunctionRegistry, pool: Arc<PgPool>) {
    register_add_client(registry, pool.clone());
    register_get_client_by_name(registry, pool.clone());
    register_list_clients(registry, pool.clone());
    log::info!("ClientTools: registered add_client, get_client_by_name, list_clients");
}

// ---------------------------------------------------------------------------
// add_client
// ---------------------------------------------------------------------------

fn register_add_client(registry: &mut FunctionRegistry, pool: Arc<PgPool>) {
    registry.register_data("add_client", move |args: String| {
        let pool = pool.clone();
        async move {
            let v: Value = match serde_json::from_str(&args) {
                Ok(v) => v,
                Err(e) => {
                    return ToolCallOutput::summary_only(
                        format!("Failed to parse arguments: {}", e),
                    );
                }
            };

            // --- Mandatory fields ---
            let name = match v["name"].as_str() {
                Some(s) if !s.trim().is_empty() => s.trim().to_string(),
                _ => return ToolCallOutput::summary_only("Missing required field: name"),
            };
            let phone = match v["phone"].as_str() {
                Some(s) if !s.trim().is_empty() => s.trim().to_string(),
                _ => return ToolCallOutput::summary_only("Missing required field: phone"),
            };
            let email = match v["email"].as_str() {
                Some(s) if !s.trim().is_empty() => s.trim().to_string(),
                _ => return ToolCallOutput::summary_only("Missing required field: email"),
            };
            let notes = match v["notes"].as_str() {
                Some(s) if !s.trim().is_empty() => s.trim().to_string(),
                _ => return ToolCallOutput::summary_only("Missing required field: notes"),
            };

            // --- Optional fields ---
            let budget_min      = v["budget_min"].as_i64();
            let budget_max      = v["budget_max"].as_i64();
            let preferred_areas = v["preferred_areas"].as_str().map(|s| s.trim().to_string());

            let result = sqlx::query!(
                r#"
                INSERT INTO clients
                    (name, email, phone, budget_min, budget_max, preferred_areas, notes, status)
                VALUES
                    ($1, $2, $3, $4, $5, $6, $7, 'active')
                RETURNING id, created_at
                "#,
                name,
                email,
                phone,
                budget_min,
                budget_max,
                preferred_areas,
                notes,
            )
            .fetch_one(pool.as_ref())
            .await;

            match result {
                Ok(row) => {
                    let summary = format!(
                        "Client '{}' added successfully with ID {}.",
                        name, row.id
                    );
                    let data = json!({
                        "id":         row.id,
                        "name":       name,
                        "email":      email,
                        "phone":      phone,
                        "created_at": row.created_at.to_string(),
                    });
                    ToolCallOutput::with_data(summary, data)
                }
                Err(e) => {
                    log::error!("add_client DB error: {}", e);
                    ToolCallOutput::summary_only(format!("Failed to add client: {}", e))
                }
            }
        }
    });
}

// ---------------------------------------------------------------------------
// get_client_by_name
// ---------------------------------------------------------------------------

fn register_get_client_by_name(registry: &mut FunctionRegistry, pool: Arc<PgPool>) {
    registry.register_data("get_client_by_name", move |args: String| {
        let pool = pool.clone();
        async move {
            let v: Value = match serde_json::from_str(&args) {
                Ok(v) => v,
                Err(e) => {
                    return ToolCallOutput::summary_only(
                        format!("Failed to parse arguments: {}", e),
                    );
                }
            };

            let raw_name = match v["name"].as_str() {
                Some(s) if !s.trim().is_empty() => s.trim().to_string(),
                _ => return ToolCallOutput::summary_only("Missing required field: name"),
            };

            // Fuzzy: wrap in % for ILIKE, handles voice transcription fuzziness
            let pattern = format!("%{}%", raw_name);

            let result = sqlx::query!(
                r#"
                SELECT id, name, email, phone, budget_min, budget_max,
                       preferred_areas, status, notes, created_at
                FROM clients
                WHERE name ILIKE $1
                ORDER BY name ASC
                LIMIT 10
                "#,
                pattern,
            )
            .fetch_all(pool.as_ref())
            .await;

            match result {
                Ok(rows) if rows.is_empty() => {
                    ToolCallOutput::summary_only(format!(
                        "No clients found matching '{}'.",
                        raw_name
                    ))
                }
                Ok(rows) => {
                    let count = rows.len();
                    let names: Vec<&str> = rows.iter().map(|r| r.name.as_str()).collect();
                    let summary = format!(
                        "Found {} client(s) matching '{}': {}.",
                        count,
                        raw_name,
                        names.join(", ")
                    );

                    let data: Vec<Value> = rows
                        .iter()
                        .map(|r| {
                            json!({
                                "id":              r.id,
                                "name":            r.name,
                                "email":           r.email,
                                "phone":           r.phone,
                                "budget_min":      r.budget_min,
                                "budget_max":      r.budget_max,
                                "preferred_areas": r.preferred_areas,
                                "status":          r.status,
                                "notes":           r.notes,
                                "created_at":      r.created_at.to_string(),
                            })
                        })
                        .collect();

                    ToolCallOutput::with_data(summary, json!(data))
                }
                Err(e) => {
                    log::error!("get_client_by_name DB error: {}", e);
                    ToolCallOutput::summary_only(format!("Database error: {}", e))
                }
            }
        }
    });
}

// ---------------------------------------------------------------------------
// list_clients
// ---------------------------------------------------------------------------

fn register_list_clients(registry: &mut FunctionRegistry, pool: Arc<PgPool>) {
    registry.register_data("list_clients", move |_args: String| {
        let pool = pool.clone();
        async move {
            let result = sqlx::query!(
                r#"
                SELECT id, name, email, phone, budget_min, budget_max,
                       preferred_areas, status, notes, created_at
                FROM clients
                WHERE status = 'active'
                ORDER BY created_at DESC
                LIMIT 50
                "#,
            )
            .fetch_all(pool.as_ref())
            .await;

            match result {
                Ok(rows) if rows.is_empty() => {
                    ToolCallOutput::summary_only("No active clients found.")
                }
                Ok(rows) => {
                    let count = rows.len();
                    let names: Vec<&str> = rows.iter().map(|r| r.name.as_str()).collect();
                    let summary = format!(
                        "{} active client(s): {}.",
                        count,
                        names.join(", ")
                    );

                    let data: Vec<Value> = rows
                        .iter()
                        .map(|r| {
                            json!({
                                "id":              r.id,
                                "name":            r.name,
                                "email":           r.email,
                                "phone":           r.phone,
                                "budget_min":      r.budget_min,
                                "budget_max":      r.budget_max,
                                "preferred_areas": r.preferred_areas,
                                "status":          r.status,
                                "notes":           r.notes,
                                "created_at":      r.created_at.to_string(),
                            })
                        })
                        .collect();

                    ToolCallOutput::with_data(summary, json!(data))
                }
                Err(e) => {
                    log::error!("list_clients DB error: {}", e);
                    ToolCallOutput::summary_only(format!("Database error: {}", e))
                }
            }
        }
    });
}
