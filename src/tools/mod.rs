//! LLM tool function modules.
//!
//! Each module registers its own handlers into a `FunctionRegistry`
//! and exposes its `FunctionSchema` list for context setup.

pub mod clients;

pub use clients::{client_tool_schemas, register_client_tools};
