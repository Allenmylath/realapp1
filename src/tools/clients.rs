//! Client management tools.
//!
//! Search stages used by `get_client_by_name`:
//!   1. Soundex   — fuzzystrmatch phonetic match (handles voice transcription variants)
//!   2. Trigram   — pg_trgm similarity > 0.3 (handles typos, partial words)
//!   3. ILIKE     — case-insensitive substring match (final fallback)
//!
//! Stages 2 and 3 require the PostgreSQL extensions:
//!   CREATE EXTENSION IF NOT EXISTS pg_trgm;
//!   CREATE EXTENSION IF NOT EXISTS fuzzystrmatch;

use std::sync::Arc;

use chrono::{DateTime, Utc};
use serde_json::{json, Value};
use sqlx::{PgPool, Postgres, QueryBuilder};
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

    /// Detail string for the LLM summary. The LLM only ever sees the summary
    /// (full_data goes downstream, not into model context), so every field the
    /// model may be asked about must appear here. Phrased as natural prose
    /// rather than a pipe-delimited record so that, even though the model is
    /// told not to recite it, anything that does leak still sounds human. The
    /// internal id is kept (needed for update_client) but bracketed and flagged
    /// so the model treats it as reference data, not something to say aloud.
    fn detail_line(&self) -> String {
        let budget = match (self.budget_min, self.budget_max) {
            (Some(mn), Some(mx)) => format!("a budget of ₹{} to ₹{}", mn, mx),
            (Some(mn), None)     => format!("a budget from ₹{}", mn),
            (None, Some(mx))     => format!("a budget up to ₹{}", mx),
            (None, None)         => "no budget on file".to_string(),
        };
        let areas = match self.preferred_areas.as_deref() {
            Some(a) => format!("interested in {}", a),
            None    => "no preferred areas yet".to_string(),
        };
        let notes = self.notes.as_deref().unwrap_or("no notes");
        format!(
            "{} ({}) — phone {}, email {}, {}, {}. Notes: {}. \
             [internal, do not say aloud: id {}]",
            self.name,
            self.status,
            self.phone.as_deref().unwrap_or("no phone on file"),
            self.email,
            budget,
            areas,
            notes,
            self.id,
        )
    }

    /// Briefing string for prep_client — the raw material for a spoken pre-showing
    /// brief. Ordered the way a realtor wants to hear it: who, how long they've
    /// been a prospect, what they're after (notes lead), budget, areas, contact.
    /// The model turns this into a few warm sentences; it must not be recited.
    fn briefing(&self) -> String {
        let age = match (Utc::now() - self.created_at).num_days() {
            d if d <= 0  => "added today".to_string(),
            1            => "a prospect since yesterday".to_string(),
            d if d < 14  => format!("a prospect for {} days", d),
            d if d < 60  => format!("a prospect for about {} weeks", d / 7),
            d            => format!("a prospect for about {} months", d / 30),
        };
        let wants = self.notes.as_deref()
            .filter(|n| !n.trim().is_empty())
            .unwrap_or("no requirements captured yet — worth asking what they're after");
        let budget = match (self.budget_min, self.budget_max) {
            (Some(mn), Some(mx)) => format!("budget ₹{} to ₹{}", mn, mx),
            (Some(mn), None)     => format!("budget from ₹{}", mn),
            (None, Some(mx))     => format!("budget up to ₹{}", mx),
            (None, None)         => "no budget on file yet".to_string(),
        };
        let areas = match self.preferred_areas.as_deref() {
            Some(a) => format!("prefers {}", a),
            None    => "no preferred areas noted".to_string(),
        };
        format!(
            "Prep for {} — {}, currently {}. What they want: {}. {}. {}. \
             Reach them on {} or {}. \
             [internal, do not say aloud: id {}]",
            self.name,
            age,
            self.status,
            wants,
            budget,
            areas,
            self.phone.as_deref().unwrap_or("no phone on file"),
            self.email,
            self.id,
        )
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
            "search_clients",
            "Find a client when you DON'T have their name — search by what they're \
             looking for, by budget, or by area. Use this for requests like \
             'who was looking around 80 lakhs?', 'I forgot his name, the guy who needs \
             a wheelchair-accessible flat', or 'anyone interested in Bandra?'. \
             All three filters are optional and combine with AND, so you can ask for \
             'someone under 1 crore wanting a villa in Andheri' in one call. \
             Provide at least one filter. \
             Budget mapping: 'up to 1 crore' → budget_max only; 'around 80 lakhs' → \
             set both budget_min and budget_max to 8000000; 'between 50 lakhs and 1 \
             crore' → budget_min 5000000 and budget_max 10000000. A client matches a \
             budget query when their own budget range overlaps the range you give. \
             Returns up to 10 matching clients with full details. \
             NOTE: if the realtor gives a NAME, use get_client_by_name instead — this \
             tool is only for nameless searches.",
        )
        .with_parameters(json!({
            "type": "object",
            "properties": {
                "requirements": {
                    "type": "string",
                    "description": "What the client wants or a distinguishing characteristic, matched against their notes and preferred areas — e.g. 'wheelchair accessible', 'villa with garden', 'near a good school'. Pass the meaningful words; common filler words are ignored automatically."
                },
                "budget_min": {
                    "type": "integer",
                    "description": "Lower bound of the budget being searched, in INR. See the budget mapping in the tool description. Omit if no lower bound."
                },
                "budget_max": {
                    "type": "integer",
                    "description": "Upper bound of the budget being searched, in INR. See the budget mapping in the tool description. Omit if no upper bound."
                },
                "area": {
                    "type": "string",
                    "description": "Locality or area to match against the client's preferred areas, e.g. 'Bandra'. Omit if not searching by area."
                }
            },
            "required": []
        })),

        FunctionSchema::new(
            "prep_client",
            "Brief the realtor on a client before a showing, call, or meeting. \
             Use this whenever the realtor wants to be prepped or briefed — e.g. \
             'prep me for Priya', 'I've got a showing with the Sharmas, catch me up', \
             'what do I need to know before I call Rahul'. \
             Looks the client up by name (same fuzzy matching as get_client_by_name) and \
             returns a briefing: how long they've been a prospect, what they're looking \
             for, their budget, preferred areas, and contact details. \
             Deliver it as a short, natural spoken brief — like a colleague catching the \
             realtor up before they walk in — not as a list of fields.",
        )
        .with_parameters(json!({
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Name or partial name of the client to be briefed on. Phonetically similar or misspelled names are also found."
                }
            },
            "required": ["name"]
        })),

        FunctionSchema::new(
            "update_client",
            "Update an existing client's details in the CRM. \
             First locate the client with get_client_by_name to obtain their ID, \
             confirm with the user which client is meant if several match, then call \
             this tool with the ID and ONLY the fields that should change — omitted \
             fields keep their current values. \
             Clients can be marked inactive via the status field, but records are \
             never deleted. \
             After updating, always read back the updated details so the user can confirm.",
        )
        .with_parameters(json!({
            "type": "object",
            "properties": {
                "client_id": {
                    "type": "string",
                    "description": "UUID of the client to update, as returned by get_client_by_name or list_clients"
                },
                "name": {
                    "type": "string",
                    "description": "New full name. Omit to keep current."
                },
                "phone": {
                    "type": "string",
                    "description": "New phone number. Omit to keep current."
                },
                "email": {
                    "type": "string",
                    "description": "New email address. Omit to keep current."
                },
                "notes": {
                    "type": "string",
                    "description": "Replacement notes text. Omit to keep current."
                },
                "budget_min": {
                    "type": "integer",
                    "description": "New minimum budget in INR. Omit to keep current."
                },
                "budget_max": {
                    "type": "integer",
                    "description": "New maximum budget in INR. Omit to keep current."
                },
                "preferred_areas": {
                    "type": "string",
                    "description": "Comma-separated preferred localities. Omit to keep current."
                },
                "status": {
                    "type": "string",
                    "enum": ["active", "inactive"],
                    "description": "Set 'inactive' to hide a client from the active list (records are never deleted). Omit to keep current."
                }
            },
            "required": ["client_id"]
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
    register_search_clients(registry, pool.clone());
    register_prep_client(registry, pool.clone());
    register_update_client(registry, pool.clone());
    register_list_clients(registry, pool.clone());
    log::info!(
        "ClientTools: registered add_client, get_client_by_name, search_clients, prep_client, update_client, list_clients"
    );
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
                        (Some(mn), Some(mx)) => format!(", budget ₹{} to ₹{}", mn, mx),
                        (Some(mn), None)     => format!(", budget from ₹{}", mn),
                        (None, Some(mx))     => format!(", budget up to ₹{}", mx),
                        (None, None)         => String::new(),
                    };
                    let areas_str = preferred_areas
                        .as_deref()
                        .map(|a| format!(", interested in {}", a))
                        .unwrap_or_default();

                    let summary = format!(
                        "Saved {} — phone {}, email {}{}{}. Notes: {}. \
                         [internal, do not say aloud: id {}]",
                        name, phone, email, budget_str, areas_str, notes, row.id
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
            let details: Vec<String> = rows.iter().map(|r| r.detail_line()).collect();
            let summary = format!(
                "Found {} client(s) matching '{}'. \
                 [internal, do not say aloud: matched via {}]\n{}",
                count,
                raw_name,
                method,
                details.join("\n")
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

// ── search_clients ─────────────────────────────────────────────────────────

/// Builds the dynamic search query from whichever filters were supplied.
/// `use_fulltext` controls how `requirements` is matched: full-text
/// (websearch_to_tsquery, the primary path) or a plain ILIKE substring fallback.
fn build_search_query<'a>(
    requirements: Option<&'a str>,
    budget_min: Option<i64>,
    budget_max: Option<i64>,
    area: Option<&'a str>,
    use_fulltext: bool,
) -> QueryBuilder<'a, Postgres> {
    let mut qb = QueryBuilder::new(
        "SELECT id, name, email, phone, budget_min, budget_max, \
         preferred_areas, status, notes, created_at FROM clients WHERE 1=1",
    );

    if let Some(req) = requirements {
        if use_fulltext {
            // websearch_to_tsquery drops stopwords and stems, so a phrase like
            // "guy on a wheelchair" reduces to `wheelchair` and matches notes
            // containing "wheelchair accessible". No extension required.
            qb.push(
                " AND to_tsvector('english', coalesce(notes,'') || ' ' || \
                 coalesce(preferred_areas,'')) @@ websearch_to_tsquery('english', ",
            );
            qb.push_bind(req);
            qb.push(")");
        } else {
            qb.push(" AND notes ILIKE ");
            qb.push_bind(format!("%{}%", req));
        }
    }

    if budget_min.is_some() || budget_max.is_some() {
        // A client matches when their own budget range overlaps the queried
        // range; NULL bounds on the client are treated as open-ended.
        let qmin = budget_min.unwrap_or(0);
        let qmax = budget_max.unwrap_or(i64::MAX);
        qb.push(" AND (budget_max IS NULL OR budget_max >= ");
        qb.push_bind(qmin);
        qb.push(") AND (budget_min IS NULL OR budget_min <= ");
        qb.push_bind(qmax);
        qb.push(")");
    }

    if let Some(a) = area {
        qb.push(" AND preferred_areas ILIKE ");
        qb.push_bind(format!("%{}%", a));
    }

    qb.push(" ORDER BY created_at DESC LIMIT 10");
    qb
}

fn register_search_clients(registry: &mut FunctionRegistry, pool: Arc<PgPool>) {
    registry.register_data("search_clients", move |args: String| {
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

            let requirements = v["requirements"].as_str()
                .map(|s| s.trim())
                .filter(|s| !s.is_empty());
            let budget_min = v["budget_min"].as_i64();
            let budget_max = v["budget_max"].as_i64();
            let area = v["area"].as_str()
                .map(|s| s.trim())
                .filter(|s| !s.is_empty());

            if requirements.is_none() && budget_min.is_none()
                && budget_max.is_none() && area.is_none()
            {
                return ToolCallOutput::summary_only(
                    "No search filters given. Ask the realtor what to go on — what the \
                     client's after, roughly what budget, or which area.",
                );
            }

            // Plain-language description of the search, for the no-match message.
            let mut criteria: Vec<String> = Vec::new();
            if let Some(req) = requirements {
                criteria.push(format!("looking for '{}'", req));
            }
            match (budget_min, budget_max) {
                (Some(mn), Some(mx)) if mn == mx => criteria.push(format!("budget around ₹{}", mn)),
                (Some(mn), Some(mx)) => criteria.push(format!("budget between ₹{} and ₹{}", mn, mx)),
                (Some(mn), None)     => criteria.push(format!("budget from ₹{}", mn)),
                (None, Some(mx))     => criteria.push(format!("budget up to ₹{}", mx)),
                (None, None)         => {}
            }
            if let Some(a) = area {
                criteria.push(format!("area '{}'", a));
            }
            let criteria_str = criteria.join(", ");

            // Primary: full-text on requirements. If nothing matches and the
            // ONLY filter was requirements, retry once with an ILIKE fallback
            // for short or oddly-phrased terms full-text might miss.
            let mut qb = build_search_query(requirements, budget_min, budget_max, area, true);
            let mut rows = match qb.build_query_as::<ClientRow>().fetch_all(pool.as_ref()).await {
                Ok(rows) => rows,
                Err(e) => {
                    log::error!("search_clients DB error: {}", e);
                    return ToolCallOutput::summary_only(format!("Database error: {}", e));
                }
            };

            let requirements_only =
                requirements.is_some() && budget_min.is_none()
                && budget_max.is_none() && area.is_none();
            if rows.is_empty() && requirements_only {
                let mut qb = build_search_query(requirements, budget_min, budget_max, area, false);
                match qb.build_query_as::<ClientRow>().fetch_all(pool.as_ref()).await {
                    Ok(r) => rows = r,
                    Err(e) => log::warn!("search_clients ILIKE fallback failed: {}", e),
                }
            }

            if rows.is_empty() {
                return ToolCallOutput::summary_only(format!(
                    "No clients found matching {}.",
                    criteria_str
                ));
            }

            let count = rows.len();
            let details: Vec<String> = rows.iter().map(|r| r.detail_line()).collect();
            let summary = format!(
                "Found {} client(s) matching {}.\n{}",
                count,
                criteria_str,
                details.join("\n")
            );
            let data: Vec<Value> = rows.iter().map(|r| r.to_json()).collect();
            ToolCallOutput::with_data(summary, json!(data))
        }
    });
}

// ── prep_client ──────────────────────────────────────────────────────────────

fn register_prep_client(registry: &mut FunctionRegistry, pool: Arc<PgPool>) {
    registry.register_data("prep_client", move |args: String| {
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

            let (rows, _method) = do_search(pool.as_ref(), &raw_name).await;

            match rows.len() {
                0 => ToolCallOutput::summary_only(format!(
                    "No client found matching '{}', so there's nothing to brief on. \
                     Ask the realtor to confirm the name.",
                    raw_name
                )),
                // Several people match — don't risk briefing the wrong one. Hand
                // back a short disambiguation list so the model can ask which.
                1 => {
                    let row = &rows[0];
                    ToolCallOutput::with_data(row.briefing(), row.to_json())
                }
                _ => {
                    let options: Vec<String> = rows
                        .iter()
                        .map(|r| {
                            let area = r.preferred_areas.as_deref().unwrap_or("no area on file");
                            format!("{} ({})", r.name, area)
                        })
                        .collect();
                    let summary = format!(
                        "Several clients match '{}' — ask the realtor which one before briefing: {}.",
                        raw_name,
                        options.join("; ")
                    );
                    let data: Vec<Value> = rows.iter().map(|r| r.to_json()).collect();
                    ToolCallOutput::with_data(summary, json!(data))
                }
            }
        }
    });
}

// ── update_client ────────────────────────────────────────────────────────────

fn register_update_client(registry: &mut FunctionRegistry, pool: Arc<PgPool>) {
    registry.register_data("update_client", move |args: String| {
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

            let client_id = match v["client_id"].as_str().map(Uuid::parse_str) {
                Some(Ok(id)) => id,
                Some(Err(_)) => {
                    return ToolCallOutput::summary_only(
                        "Invalid client_id: not a valid UUID. \
                         Use get_client_by_name first to obtain the client's ID.",
                    );
                }
                None => {
                    return ToolCallOutput::summary_only("Missing required field: client_id");
                }
            };

            let name            = v["name"].as_str().map(|s| s.trim().to_string());
            let email           = v["email"].as_str().map(|s| s.trim().to_string());
            let phone           = v["phone"].as_str().map(|s| s.trim().to_string());
            let notes           = v["notes"].as_str().map(|s| s.trim().to_string());
            let budget_min      = v["budget_min"].as_i64();
            let budget_max      = v["budget_max"].as_i64();
            let preferred_areas = v["preferred_areas"].as_str().map(|s| s.trim().to_string());
            let status          = v["status"].as_str().map(|s| s.trim().to_string());

            if let Some(s) = status.as_deref() {
                if s != "active" && s != "inactive" {
                    return ToolCallOutput::summary_only(
                        "Invalid status: must be 'active' or 'inactive'.",
                    );
                }
            }

            if name.is_none() && email.is_none() && phone.is_none() && notes.is_none()
                && budget_min.is_none() && budget_max.is_none()
                && preferred_areas.is_none() && status.is_none()
            {
                return ToolCallOutput::summary_only(
                    "No fields to update: provide at least one of name, phone, email, \
                     notes, budget_min, budget_max, preferred_areas, or status.",
                );
            }

            // NULL binds leave the column unchanged via COALESCE; fields can
            // therefore be updated but never cleared, and rows never deleted.
            let sql = "UPDATE clients SET \
                       name            = COALESCE($2, name), \
                       email           = COALESCE($3, email), \
                       phone           = COALESCE($4, phone), \
                       budget_min      = COALESCE($5, budget_min), \
                       budget_max      = COALESCE($6, budget_max), \
                       preferred_areas = COALESCE($7, preferred_areas), \
                       notes           = COALESCE($8, notes), \
                       status          = COALESCE($9, status) \
                       WHERE id = $1 \
                       RETURNING id, name, email, phone, budget_min, budget_max, \
                                 preferred_areas, status, notes, created_at";

            match sqlx::query_as::<_, ClientRow>(sql)
                .bind(client_id)
                .bind(name)
                .bind(email)
                .bind(phone)
                .bind(budget_min)
                .bind(budget_max)
                .bind(preferred_areas)
                .bind(notes)
                .bind(status)
                .fetch_optional(pool.as_ref())
                .await
            {
                Ok(Some(row)) => {
                    let summary = format!("Client updated. {}", row.detail_line());
                    let data = row.to_json();
                    ToolCallOutput::with_data(summary, data)
                }
                Ok(None) => ToolCallOutput::summary_only(format!(
                    "No client found with ID {}. \
                     Use get_client_by_name to look up the correct ID.",
                    client_id
                )),
                Err(e) => {
                    log::error!("update_client DB error: {}", e);
                    ToolCallOutput::summary_only(format!("Failed to update client: {}", e))
                }
            }
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
                    let details: Vec<String> = rows.iter().map(|r| r.detail_line()).collect();
                    let summary = format!(
                        "{} active client(s):\n{}",
                        count,
                        details.join("\n")
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
