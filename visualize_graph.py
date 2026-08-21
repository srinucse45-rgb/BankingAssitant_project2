"""
Generate visualization of the Banking Assistant LangGraph structure.

This creates Bank_assistant.png showing the complete RAG + SQL agent flow.
"""

import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.graph.rag_graph import app


def generate_graph_visualization():
    """
    Generate and save the LangGraph visualization as Bank_assistant.png
    """
    try:
        graph = app

        # Generate Mermaid diagram
        graph_png = graph.get_graph().draw_mermaid_png()

        # Save to file
        output_path = PROJECT_ROOT / "Bank_assistant.png"
        with open(output_path, "wb") as f:
            f.write(graph_png)

        print(f"✅ Graph visualization saved: {output_path}")
        return str(output_path)

    except Exception as e:
        print(f"❌ Error generating visualization: {e}")
        print("\nAlternative: Using fallback ASCII visualization...")
        return None


def generate_ascii_diagram():
    """
    Fallback: Generate ASCII diagram of the graph structure
    """
    ascii_diagram = """
╔════════════════════════════════════════════════════════════════════════════╗
║                  BANKING ASSISTANT - MULTIAGENT FLOW                       ║
╚════════════════════════════════════════════════════════════════════════════╝

                                    START
                                      │
                                      ▼
                        ┌─────────────────────────┐
                        │  Input Guardrail Check  │ (Validate PII/Security)
                        └────┬────────────┬────────┘
                             │            │
                    (Block)   │            │   (Pass)
                             ▼            ▼
                        ┌────────┐   ┌──────────────────┐
                        │Finalize│   │Query Classifier  │ (Route: General/RAG/SQL/Hybrid)
                        └────┬───┘   └────┬──┬──┬────┬──┘
                             │            │  │  │    │
                        END  │      General│  │  │SQL │  Hybrid
                             │      Chat  │  │  │    │
                             │            │  │  │    │
                        ┌────▼─────┐ ┌───▼─┐│  │    │ ┌──────────────┐
                        │  Finalize │ │General└─┘  │  │ RAG Retriever│
                        │  (Output) │ │Chat       SQL│  │ + Reranker  │
                        └─────┬─────┘ │  │         │  └────┬────────┬┘
                              │       │  │         │       │        │
                              │       │  │    ┌────▼──┐    │   ┌────▼────┐
                              │       │  │    │ SQL   │    │   │Hybrid   │
                              │       │  │    │Execute│    │   │ SQL +   │
                              │       │  │    │       │    │   │ Retrieval
                              │       │  │    └────┬──┘    │   └────┬────┘
                              │       │  │         │       │        │
                              │       └──┼────┬────┴─┬─────┴────┬───┘
                              │          │    │      │         │
                              │   ┌──────▼────▼──────▼─────────▼──┐
                              │   │  Response Generator           │
                              │   │  (LLM + Retrieved Context)    │
                              │   └──────────────┬────────────────┘
                              │                  │
                              └──────────────────┴────────────────┘
                                                 │
                                          ┌──────▼──────┐
                                          │   Finalize  │
                                          │   & Output  │
                                          └──────┬──────┘
                                                 │
                                                 ▼
                                               END


═══════════════════════════════════════════════════════════════════════════════
                            NODE DESCRIPTIONS
═══════════════════════════════════════════════════════════════════════════════

1. INPUT_GUARDRAIL
   └─ Validates question for PII/sensitive data
   └─ Blocks harmful requests
   └─ Routes to finalize if guardrail triggered

2. QUERY_CLASSIFIER
   └─ Analyzes question intent
   └─ Routes to: GENERAL, RAG, SQL, or HYBRID path
   └─ Decision based on banking terminology detection

3. GENERAL_CHAT
   └─ Handles conversational questions
   └─ Uses LLM with memory
   └─ No document/database access needed

4. RAG_RETRIEVER
   └─ Hybrid search (vector + full-text)
   └─ Retrieves from uploaded banking documents
   └─ Returns top-K relevant passages

5. RERANKER
   └─ Re-ranks retrieved documents
   └─ Uses cross-encoder model
   └─ Routes to HYBRID_SQL or RESPONSE_GENERATOR

6. SQL_EXECUTOR
   └─ Translates questions to SQL queries
   └─ Executes read-only queries
   └─ Safe banking account/transaction access

7. HYBRID_SQL_EXECUTOR
   └─ Combines RAG + SQL results
   └─ Merges document and database context
   └─ Unified retrieval for complex queries

8. RESPONSE_GENERATOR
   └─ Generates final answer
   └─ Uses LLM with context from retrievers
   └─ Validates banking-specific guardrails

9. FINALIZE
   └─ Formats and returns response
   └─ Saves to conversation history
   └─ Stores memory (non-sensitive only)

═══════════════════════════════════════════════════════════════════════════════
                            ROUTING LOGIC
═══════════════════════════════════════════════════════════════════════════════

GUARDRAIL CHECK
  ├─ Detects PII (credit card, account number, SSN, password, etc.)
  ├─ Blocks harmful requests
  └─ If triggered → FINALIZE (explain why blocked)

QUERY CLASSIFICATION
  ├─ General Banking Chat
  │  └─ "Hello", "How are you?", "What time is it?"
  │  └─ Routes to: GENERAL_CHAT
  │
  ├─ Document-based (RAG)
  │  └─ "Tell me about fixed deposits", "What are the fees?"
  │  └─ Requires active uploaded document
  │  └─ Routes to: RAG_RETRIEVER
  │
  ├─ SQL Query (Database)
  │  └─ "Check my balance", "List my transactions"
  │  └─ Requires account data
  │  └─ Routes to: SQL_EXECUTOR
  │
  └─ Hybrid (RAG + SQL)
     └─ "Compare my loan EMI with policies in the document"
     └─ Routes to: RAG_RETRIEVER → RERANKER → HYBRID_SQL_EXECUTOR

═══════════════════════════════════════════════════════════════════════════════
    """
    print(ascii_diagram)
    return ascii_diagram


if __name__ == "__main__":
    print("Generating Bank Assistant Visualization...\n")

    # Try to generate PNG
    png_path = generate_graph_visualization()

    # Always show ASCII diagram
    print("\n" + "=" * 80 + "\n")
    generate_ascii_diagram()

    if png_path:
        print(f"\n✅ Visualization created at: {png_path}")
    else:
        print("\n⚠️  PNG generation failed. ASCII diagram generated instead.")
