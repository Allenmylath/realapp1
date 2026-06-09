//! Client management tools.
//!
//! Search stages used by `get_client_by_name`:
//!   1. ILIKE     — case-insensitive substring match (fast, exact)
//!   2. Trigram   — pg_trgm similarity > 0.3 (handles typos, partial words)
//!   3. Soundex   — fuzzystrmatch phonetic match (handles voice transcription variants)
//!
//! Stages 2 and 3 require the PostgreSQL extensions:
//!   CREATE EXTENSION IF NOT EXISTS pg_trgm;
//!   CREATE EXTENSION IF NOT EXISTS fuzzystrmatch;

use std::sync::Arc;

use chrono::{DateTime, Utc};
use serde_json::{json, Value};
use sqlx::PgPool;
use uuid::Uuid;

use rustvani::adapters::schemas::FunctionSchema;
use rustvani::services::llm::FunctionRegistry;
use rustvani::services::llm::function_registry::ToolCallOutput;

// ── Shared row type ──────────────────────────────────────────────────────────

#[derive(sqlx::FromRow)]
struct ClientRow {
    id:              Uuid,
    name:            String,
    email:           String,
    phone:           Option<String>,
    budget_min:      Option<i64>,
    budget_max:      Option<i64>,
    preferred_areas: Option<String>,
    status:          String,
    notes:           Option<String>,
    created_at:      DateTime<Utc>,
}

impl ClientRow {
    fn to_json(&self) -> Value {
        json!({
            "id":              self.id,
            "name":            self.name,
            "email":           self.email,
            "phone":           self.phone,
            "budget_min":      self.budget_min,
            "budget_max":      self.budget_max,
            "preferred_areas": self.preferred_areas,
            "status":          self.status,
            "notes":           self.notes,
            "created_at":      self.created_at.to_string(),
        })
    }
}

// ── Three-stage fuzzy search ─────────────────────────────────────────────────

/// Searches clients progressively through three fallback stages.
/// Returns the first non-empty result set and a label for the stage that matched.
async fn do_search(pool: &PgPool, raw_name: &str) -> (Vec<ClientRow>, &'static str) {
    const COLS: &str =
        "SELECT id, name, email, phone, budget_min, budget_max, \
         preferred_areas, status, notes, created_at FROM clients";

    // Stage 1: Soundex phonetic match (fuzzystrmatch extension)
    let soundex_sql = format!(
        "{} WHERE soundex(name) = soundex($1) ORDER BY name ASC LIMIT 10",
        COLS
    );
    match sqlx::query_as::<_, ClientRow>(&soundex_sql)
        .bind(raw_name)
        .fetch_all(pool)
        .await
    {
        Ok(rows) if !rows.is_empty() => return (rows, "phonetic soundex match"),
        Ok(_) => {}
        Err(e) => log::warn!("Soundex search failed (fuzzystrmatch installed?): {}", e),
    }

    // Stage 2: trigram similarity (pg_trgm extension)
    let trigram_sql = format!(
        "{} WHERE similarity(name, $1) > 0.3 \
         ORDER BY similarity(name, $1) DESC LIMIT 10",
        COLS
    );
    match sqlx::query_as::<_, ClientRow>(&trigram_sql)
        .bind(raw_name)
        .fetch_all(pool)
        .await
    {
        Ok(rows) if !rows.is_empty() => return (rows, "fuzzy trigram match"),
        Ok(_) => {}
        Err(e) => log::warn!("Trigram search failed (pg_trgm installed?): {}", e),
    }

    // Stage 3: case-insensitive substring (ILIKE)
    let ilike_sql = format!("{} WHERE name ILIKE $1 ORDER BY name ASC LIMIT 10", COLS);
    let pattern = format!("%{}%", raw_name);
    match sqlx::query_as::<_, ClientRow>(&ilike_sql)
        .bind(&pattern)
        .fetch_all(pool)
        .await
    {
        Ok(rows) if !rows.is_empty() => return (rows, "substring match"),
        Ok(_) => {}
        Err(e) => log::warn!("ILIKE search error: {}", e),
    }

    (vec![], "no match")
}

// ── Schema definitions — what the LLM sees ──────────────────────────────────

/// Returns the `FunctionSchema` list for all client tools.
pub fn client_tool_schemas() -> Vec<FunctionSchema> {
    vec![
        FunctionSchema::new(
            "add_client",
            "Add a new real estate prospect to the CRM database. \
             Required fields: full name, phone number, email address, and notes summarising \
             their property requirements. Optional: budget range in INR and preferred \
             areas/localities. After adding, always read back the recorded details to the \
             user so they can confirm everything was captured correctly.",
        )
        .with_parameters(json!({
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Full name of the client (first and last name)"
                },
                "phone": {
                    "type": "string",
                    "description": "Mobile or landline number, including country code if provided"
                },
                "email": {
                    "type": "string",
                    "description": "Email address of the client"
                },
                "notes": {
                    "type": "string",
                    "description": "Property requirements, preferences, timeline, or any other relevant notes"
                },
                "budget_min": {
                    "type": "integer",
                    "description": "Minimum budget in INR (e.g. 5000000 for 50 lakhs). Omit if not mentioned."
                },
                "budget_max": {
                    "type": "integer",
                    "description": "Maximum budget in INR (e.g. 10000000 for 1 crore). Omit if not mentioned."
                },
                "preferred_areas": {
                    "type": "string",
                    "description": "Comma-separated preferred localities or areas (e.g. 'Bandra, Andheri'). Omit if not mentioned."
                }
            },
            "required": ["name", "phone", "email", "notes"]
        })),

        FunctionSchema::new(
            "get_client_by_name",
            "Search for one or more clients by name. \
             Uses three progressive stages so names are found even with voice transcription \
             errors or spelling variants: \
             (1) Soundex phonetic match — primary, finds names that sound alike but are spelled differently; \
             (2) trigram similarity — fallback for typos and partial words; \
             (3) case-insensitive substring match — final fallback for exact partial names. \
             Returns up to 10 matching clients with full details: contact info, budget range, \
             preferred areas, status, and notes. \
             Always summarise clearly who was found and their key details.",
        )
        .with_parameters(json!({
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Name or partial name to search for. Phonetically similar or misspelled names are also found."
                }
            },
            "required": ["name"]
        })),

        FunctionSchema::new(
            "list_clients",
            "Retrieve all active clients from the CRM, ordered by most recently added. \
             Returns up to 50 clients, each with full contact details, budget range, \
             preferred areas, and notes. \
             Use this when the user wants an overview of all clients, asks 'who are my clients', \
             or does not have a specific name to search for.",
        )
        .with_parameters(json!({
            "type": "object",
            "properties": {},
            "required": []
        })),
    ]
}

// ── Registration ─────────────────────────────────────────────────────────────

/// Register all client tool handlers into `registry`.
pub fn register_client_tools(registry: &mut FunctionRegistry, pool: Arc<PgPool>) {
    register_add_client(registry, pool.clone());
    register_get_client_by_name(registry, pool.clone());
    register_list_clients(registry, pool.clone());
    log::info!("ClientTools: registered add_client, get_client_by_name, list_clients");
}

// ── add_client ───────────────────────────────────────────────────────────────

fn register_add_client(registry: &mut FunctionRegistry, pool: Arc<PgPool>) {
    registry.register_data("add_client", move |args: String| {
        let pool = pool.clone();
        async move {
            let v: Value = match serde_json::from_str(&args) {
                Ok(v) => v,
                Err(e) => {
                    return ToolCallOutput::summary_only(format!(
                        "Failed to parse arguments: {}",
                        e
                    ));
                }
            };

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
                    let budget_str = match (budget_min, budget_max) {
                        (Some(mn), Some(mx)) => format!(" | Budget: ₹{} – ₹{}", mn, mx),
                        (Some(mn), None)     => format!(" | Budget from: ₹{}", mn),
                        (None, Some(mx))     => format!(" | Budget up to: ₹{}", mx),
                        (None, None)         => String::new(),
                    };
                    let areas_str = preferred_areas
                        .as_deref()
                        .map(|a| format!(" | Areas: {}", a))
                        .unwrap_or_default();

                    let summary = format!(
                        "Client '{}' added (ID: {}). Phone: {} | Email: {}{}{}. Notes: {}",
                        name, row.id, phone, email, budget_str, areas_str, notes
                    );
                    let data = json!({
                        "id":              row.id,
                        "name":            name,
                        "email":           email,
                        "phone":           phone,
                        "budget_min":      budget_min,
                        "budget_max":      budget_max,
                        "preferred_areas": preferred_areas,
                        "notes":           notes,
                        "created_at":      row.created_at.to_string(),
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

// ── get_client_by_name ───────────────────────────────────────────────────────

fn register_get_client_by_name(registry: &mut FunctionRegistry, pool: Arc<PgPool>) {
    registry.register_data("get_client_by_name", move |args: String| {
        let pool = pool.clone();
        async move {
            let v: Value = match serde_json::from_str(&args) {
                Ok(v) => v,
                Err(e) => {
                    return ToolCallOutput::summary_only(format!(
                        "Failed to parse arguments: {}",
                        e
                    ));
                }
            };

            let raw_name = match v["name"].as_str() {
                Some(s) if !s.trim().is_empty() => s.trim().to_string(),
                _ => return ToolCallOutput::summary_only("Missing required field: name"),
            };

            let (rows, method) = do_search(pool.as_ref(), &raw_name).await;

            if rows.is_empty() {
                return ToolCallOutput::summary_only(format!(
                    "No clients found matching '{}'. \
                     Tried substring, trigram similarity, and phonetic Soundex matching.",
                    raw_name
                ));
            }

            let count = rows.len();
            let names: Vec<&str> = rows.iter().map(|r| r.name.as_str()).collect();
            let summary = format!(
                "Found {} client(s) for '{}' via {}: {}.",
                count,
                raw_name,
                method,
                names.join(", ")
            );

            let data: Vec<Value> = rows
                .iter()
                .map(|r| {
                    let mut obj = r.to_json();
                    obj["match_method"] = json!(method);
                    obj
                })
                .collect();

            ToolCallOutput::with_data(summary, json!(data))
        }
    });
}

// ── list_clients ─────────────────────────────────────────────────────────────

fn register_list_clients(registry: &mut FunctionRegistry, pool: Arc<PgPool>) {
    registry.register_data("list_clients", move |_args: String| {
        let pool = pool.clone();
        async move {
            let sql = "SELECT id, name, email, phone, budget_min, budget_max, \
                       preferred_areas, status, notes, created_at \
                       FROM clients WHERE status = 'active' \
                       ORDER BY created_at DESC LIMIT 50";

            match sqlx::query_as::<_, ClientRow>(sql)
                .fetch_all(pool.as_ref())
                .await
            {
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
                    let data: Vec<Value> = rows.iter().map(|r| r.to_json()).collect();
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
