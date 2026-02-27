# 🤖 Candidates FTE — Digital Full-Time Employee

A multi-agent autonomous HR system that acts as your personal job search assistant.
**Not a SaaS tool** — a delegated digital worker you manage.

---

## Architecture

```
User Chat → Orchestrator Agent (LangGraph)
               ├── Job Search Agent   → Apify API (LinkedIn/Indeed/Glassdoor)
               ├── Resume Agent       → Claude + BowJob-style CV tailoring
               ├── Apply Agent        → Gmail API + HR email finder
               └── Interview Agent    → Mock interviews + skill gap analysis
                        ↕
              LangSmith + Langfuse (full observability)
                        ↕
              PostgreSQL (long-term) + Redis (short-term)
```

---

## Quick Start

### 1. Prerequisites

- Python 3.12+
- Docker + Docker Compose
- Node.js 18+ (for frontend)

### 2. Clone & Configure

```bash
git clone <your-repo>
cd candidates-fte

# Backend config
cp backend/.env.example backend/.env
# Fill in your API keys in backend/.env
```

### 3. API Keys You Need

| Service | Where to get | Required? |
|---------|-------------|-----------|
| Anthropic API | console.anthropic.com | ✅ Yes |
| Apify API | apify.com | ✅ Yes (job scraping) |
| Gmail OAuth2 | console.developers.google.com | ✅ Yes (sending emails) |
| LangSmith | smith.langchain.com | Recommended |
| Langfuse | cloud.langfuse.com | Recommended |
| Hunter.io | hunter.io | Optional (HR email lookup) |

### 4. Gmail Setup

1. Go to [Google Cloud Console](https://console.cloud.google.com)
2. Create a new project
3. Enable **Gmail API**
4. Create OAuth2 credentials (Web Application type)
5. Add `http://localhost:8000/auth/gmail/callback` as redirect URI
6. Copy Client ID + Secret to your `.env`

### 5. Apify Setup

1. Sign up at [apify.com](https://apify.com)
2. Get your API key from Settings → Integrations
3. Add to `.env` as `APIFY_API_KEY`

### 6. BowJob Integration (CV Building)

The Resume Agent is inspired by [BowJob](https://github.com/rurahim/BowJob).
Study their CV parsing prompts for reference:

```bash
git clone https://github.com/rurahim/BowJob.git bowjob-reference
# Review: bowjob-reference/prompts/ and bowjob-reference/cv_parser/
```

### 7. Start Services

```bash
# Start PostgreSQL + Redis + API + Frontend
docker-compose up -d

# Or run backend directly:
cd backend
pip install -r requirements.txt
uvicorn api.main:app --reload --port 8000
```

---

## API Endpoints

### Chat (Orchestrator)
```
POST /chat                          → Start workflow, returns workflow_id
GET  /chat/stream/{workflow_id}     → SSE stream of agent events
GET  /chat/history/{session_id}     → Chat history
GET  /workflow/{workflow_id}/status → Current workflow state
POST /workflow/{workflow_id}/interrupt → Pause running workflow
```

### Profile & CV Upload
```
POST /profile                       → Create/update user profile
GET  /profile/{user_id}
POST /profile/{user_id}/upload-cv   → Upload PDF/DOCX CV for parsing
```

### Jobs
```
GET  /jobs/{user_id}                → All jobs (filter by ?status=shortlisted)
PATCH /jobs/{job_id}                → Update job status
POST /jobs/{user_id}/shortlist      → Shortlist specific jobs
```

### CVs (HITL)
```
GET  /cvs/{user_id}                 → All tailored CVs
POST /cvs/edit                      → Edit CV section (inline or AI-assisted)
POST /cvs/approve                   → Approve CVs for applying
POST /hitl/{checkpoint_id}/resolve  → Resolve any HITL checkpoint
```

### Applications (HITL)
```
GET  /applications/{user_id}        → All email drafts + sent applications
POST /applications/edit             → Edit email/cover letter
POST /applications/approve          → Mark emails as approved
POST /applications/send             → Send approved emails via Gmail
```

### Gmail Auth
```
GET  /auth/gmail                    → Get OAuth2 URL
GET  /auth/gmail/callback           → OAuth2 callback (redirect here from Google)
GET  /auth/gmail/status/{user_id}   → Check if Gmail is connected
```

### Interview Prep
```
GET  /interview/{user_id}/prep      → All interview prep materials
POST /interview/chat                → Mock interview chat message
```

### Observability
```
GET  /observability/{workflow_id}/traces → LangSmith traces for a workflow
```

---

## Example Workflows

### Full Pipeline
```
User: "Find senior Python engineer jobs in Pakistan and apply to the top 5"

Orchestrator activates:
1. Job Search Agent → scrapes 150 jobs → deduplicates → scores → saves top 50
2. Resume Agent → tailors CV for each of top 5 → triggers HITL
   ⏸ PAUSED → user reviews CVs in CVs tab, edits if needed, approves
3. Apply Agent → finds HR emails → drafts emails → triggers HITL
   ⏸ PAUSED → user reviews emails in Applications tab, approves
4. Emails sent via Gmail ✅
```

### Single Task
```
User: "Help me prepare for my Stripe interview next week"
Orchestrator activates only Interview Agent
→ Generates Stripe-specific questions → skill gap analysis
```

---

## Project Structure

```
candidates-fte/
├── backend/
│   ├── agents/
│   │   ├── orchestrator.py      # LangGraph central brain
│   │   ├── job_search_agent.py  # Apify scraping + deduplication
│   │   ├── resume_agent.py      # CV tailoring + HITL editing
│   │   ├── apply_agent.py       # Gmail sending + HITL
│   │   └── interview_agent.py   # Mock interviews + skill gaps
│   ├── tools/
│   │   ├── apify_scraper.py     # Job scraping with dedup
│   │   ├── cv_parser.py         # PDF/DOCX parsing
│   │   └── gmail_sender.py      # Gmail OAuth2 + sending
│   ├── memory/
│   │   └── store.py             # Redis + PostgreSQL
│   ├── api/
│   │   └── main.py              # FastAPI + SSE endpoints
│   ├── observability/
│   │   └── config.py            # LangSmith + Langfuse
│   └── schemas/
│       └── models.py            # Pydantic models
├── frontend/                    # Next.js (build separately)
├── docker-compose.yml
└── README.md
```

---

## HITL (Human-In-The-Loop) Flow

```
Agent finishes CVs/emails
        ↓
Agent raises HITLCheckpoint (saved in Redis)
        ↓
Workflow pauses → status = AWAITING_APPROVAL
        ↓
Frontend shows notification + badge on CVs/Applications tab
        ↓
User reviews, edits (inline editor or chatbot), approves
        ↓
POST /hitl/{checkpoint_id}/resolve?approved=true
        ↓
Workflow resumes from checkpoint
```

---

## Observability

Every agent action is traced:

- **LangSmith**: Full chain tracing, LLM call logs, latency
- **Langfuse**: Production monitoring, user feedback scores, cost tracking
- **Observability Tab** in frontend: visual trace viewer, agent DAG

View traces at:
- LangSmith: https://smith.langchain.com
- Langfuse: https://cloud.langfuse.com

---

## License

MIT
