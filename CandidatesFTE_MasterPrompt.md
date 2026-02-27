# Candidates FTE — Master Build Prompt

> Copy-paste the entire block below as your project prompt into Claude or your LLM-powered dev environment.

---

## 🧠 SYSTEM PROMPT — CANDIDATES FTE (Digital Full-Time Employee)

You are a senior full-stack AI systems architect. Your task is to design and build **Candidates FTE** — a **Digital Full-Time Employee (FTE)** for autonomous job searching, resume tailoring, applying, and interview preparation. This is **NOT a SaaS tool**. It is a delegated digital worker the user manages, not a product the user operates.

---

### 🏗️ ARCHITECTURE OVERVIEW

Build a **multi-agent orchestration system** using:

- **LangChain** (agent framework + tool use)
- **LangGraph** (agent state machines + multi-agent graphs)
- **LangSmith** (tracing, observability, evals)
- **Langfuse** (production monitoring, cost tracking, human feedback loops)
- **FastAPI** (backend API layer)
- **Next.js + Tailwind CSS** (frontend — single-window interface)
- **PostgreSQL + Redis** (persistent memory + short-term context cache)
- **Gmail API** (Google OAuth2 for sending emails)
- **Apify API** (job scraping from LinkedIn, Indeed, Glassdoor, etc.)

---

### 👷 AGENTS TO BUILD

#### 1. **Orchestrator Agent** (Central Brain)
- Reads user's natural language query
- Identifies intent and scope (full pipeline vs. single task)
- Decides which sub-agents to activate and in what sequence
- Manages shared memory across agents
- Can interrupt and resume workflows on user command
- All decisions and reasoning must be traceable via LangSmith/Langfuse

#### 2. **Job Search Agent**
- Uses Apify API to scrape job listings from: LinkedIn, Indeed, Glassdoor, company career pages
- Deduplicates listings (same role posted on multiple platforms → merge into one record)
- Matches jobs against user profile: skills, experience, location, salary, role preferences
- Stores results in DB with metadata: source, posted date, match score, status
- Exposes results in a dedicated **"Jobs" tab** in the UI

#### 3. **Resume & Profile Building Agent**
- Parses user's existing CV (PDF/DOCX)
- Clones and adapts the CV-building logic from: `https://github.com/rurahim/BowJob.git`
  - Study their CV parsing pipeline and JD-CV matching prompts before writing your own
- Generates a tailored, ATS-optimized resume per job
- Optimizes for keywords from the job description
- Optionally enhances LinkedIn profile sections
- **HITL Gate:** Before finalizing, present all generated CVs in an interactive CV Review Panel (see UI spec below)

#### 4. **Apply Agent**
- Fetches HR/recruiter email for each shortlisted company (via Apify, Apollo.io, or Hunter.io)
- Drafts cover letter + email body per application
- **HITL Gate:** Shows all outgoing emails + cover letters in an Email Review Panel before sending
- Sends approved emails via Gmail API (Google OAuth2)
- Logs application status (Sent / Pending / Replied / Rejected) in a **"Applications" tab**

#### 5. **Interview Preparation Agent**
- Generates role-specific technical + behavioral interview questions
- Runs mock interview simulations (chat-based Q&A)
- Provides skill gap analysis + learning resource recommendations
- Results visible in an **"Interview Prep" tab**

---

### 🔁 WORKFLOW LOGIC

```
User Input (natural language)
        ↓
Orchestrator Agent
  → Parses intent
  → Selects agents
  → Builds execution plan
  → Executes step-by-step
        ↓
[Job Search] → [Resume Tailoring] → [HITL: CV Review] → [Apply] → [HITL: Email Review] → [Send]
                                                                              ↓
                                                                    [Track Responses]
```

**Important rules:**
- The flow above is NOT always linear. User may trigger only one or two agents (e.g., "just prep me for interviews" → only Interview Prep Agent activates).
- Orchestrator must handle partial flows gracefully.
- User can interrupt at any point.
- Every agent handoff must be logged and visible.

---

### 🖥️ UI SPECIFICATION (Single-Window — Digital FTE Interface)

#### Main Layout
```
┌──────────────────────────────────────────────────────────┐
│  HEADER: Candidates FTE  |  [Status: Idle/Working/Paused]│
├──────────────────────────────────────────────────────────┤
│  TABS: [Chat] [Jobs] [CVs] [Applications] [Interview Prep]│
│        [Observability]                                   │
├──────────────────────────────────────────────────────────┤
│                                                          │
│   ACTIVE TAB CONTENT AREA                               │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

#### Tab Descriptions

**[Chat Tab]** — Primary interaction window
- Natural language input field at bottom
- Chat history showing: user messages, agent responses, agent reasoning summaries, HITL prompts
- Live activity feed: "Job Search Agent fetching LinkedIn roles…", "Resume Agent tailoring CV for Stripe…"
- Interrupt button: "⏸ Pause" to stop mid-flow

**[Jobs Tab]** — Job listings fetched by the Job Search Agent
- Table/card view: Role, Company, Platform, Location, Match Score, Status
- Filter/search
- User can manually shortlist or exclude roles

**[CVs Tab]** — HITL: CV Review Panel *(Critical Feature)*
- For each shortlisted job, show a card with:
  - Job title + company name
  - Tailored CV rendered inline (scrollable preview)
  - **Inline CV Editor**: Rich text editor (e.g., TipTap or Quill) for direct edits
  - **CV Chatbot**: A side panel where user can type "Rewrite my summary section to sound more senior" and the agent regenerates just that section
  - Status: [Pending Approval] [Editing] [Approved]
  - "✅ Approve" button per CV
- "Approve All" button to bulk-approve
- Only approved CVs proceed to the Apply Agent

**[Applications Tab]** — HITL: Email Review Panel + Tracker
- Before sending: Show each email draft + cover letter side by side
  - Editable email body (inline editor)
  - Email Chatbot: "Make the tone more confident" → agent regenerates
  - "✅ Approve & Send" per email
- After sending: Application tracker table
  - Columns: Company, Role, Sent Date, HR Email, Status (Sent/Replied/Interview/Rejected)

**[Interview Prep Tab]**
- Role selector (pick from applied jobs)
- Q&A simulation chat
- Skill gap report
- Resource links

**[Observability Tab]** — LangSmith + Langfuse integration
- Live trace viewer: all agent actions, tool calls, LLM prompts/responses, latencies
- Agent handoff diagram (visual DAG)
- Cost tracker (tokens used, estimated cost)
- User can click any trace node to inspect inputs/outputs

---

### 🛑 HUMAN-IN-THE-LOOP (HITL) RULES

These are non-negotiable. No irreversible action happens without user approval.

| Action | HITL Required? | Where Shown |
|--------|---------------|-------------|
| Finalizing tailored CV | ✅ Yes | CVs Tab |
| Sending application email | ✅ Yes | Applications Tab |
| Posting to job boards (future) | ✅ Yes | Chat notification |
| Internal data processing | ❌ No | Background |

HITL implementation:
- Agent raises a `HITLCheckpoint` event
- Workflow pauses and sets status to `AWAITING_APPROVAL`
- UI highlights the pending review with a badge/notification
- User approves, edits, or rejects
- On approval, workflow resumes from checkpoint
- On rejection, agent regenerates and re-presents

---

### 🧠 MEMORY ARCHITECTURE

**Short-term (Redis)**
- Current session context
- Agent working state
- Conversation history (last N turns)

**Long-term (PostgreSQL)**
- User profile: skills, experience, preferences, uploaded CV
- Job history: all fetched jobs + match scores
- Application history: all sent emails, statuses, replies
- CV versions: per-job tailored CV snapshots
- Feedback: user edits and corrections (used for agent self-improvement)

---

### 📡 OBSERVABILITY REQUIREMENTS

- Integrate **LangSmith** for:
  - Full chain/agent tracing
  - LLM call logging (prompt + completion)
  - Latency and token tracking

- Integrate **Langfuse** for:
  - Production monitoring
  - Human feedback capture (thumbs up/down on CV quality, email quality)
  - Session-level analytics

- All agent traces must be visible in the **Observability Tab**
- No agent should operate invisibly — every tool call, LLM call, and handoff is logged

---

### 🔌 EXTERNAL INTEGRATIONS

| Integration | Purpose | Notes |
|-------------|---------|-------|
| **Apify API** | Job scraping | Sign up at apify.com, get API key |
| **Gmail API** | Send emails | Google Developer Console, OAuth2 |
| **Google OAuth2** | Auth for Gmail | Redirect URI must be configured |
| **BowJob (GitHub)** | CV parsing + JD matching prompts | Clone `https://github.com/rurahim/BowJob.git`, study and adapt |
| **Hunter.io / Apollo.io** | HR email lookup | Optional fallback to manual input |
| **LangSmith** | Tracing | Set `LANGCHAIN_TRACING_V2=true` |
| **Langfuse** | Monitoring | Set `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` |

---

### 🗂️ PROJECT STRUCTURE (Recommended)

```
candidates-fte/
├── backend/
│   ├── agents/
│   │   ├── orchestrator.py
│   │   ├── job_search_agent.py
│   │   ├── resume_agent.py
│   │   ├── apply_agent.py
│   │   ├── interview_agent.py
│   │   └── hitl_manager.py
│   ├── tools/
│   │   ├── apify_scraper.py
│   │   ├── gmail_sender.py
│   │   ├── cv_parser.py
│   │   ├── email_finder.py
│   │   └── jd_matcher.py
│   ├── memory/
│   │   ├── short_term.py      # Redis
│   │   └── long_term.py       # PostgreSQL
│   ├── api/
│   │   └── main.py            # FastAPI routes
│   └── observability/
│       ├── langsmith_config.py
│       └── langfuse_config.py
├── frontend/
│   ├── app/
│   │   ├── page.tsx           # Main FTE interface
│   │   ├── tabs/
│   │   │   ├── ChatTab.tsx
│   │   │   ├── JobsTab.tsx
│   │   │   ├── CVsTab.tsx
│   │   │   ├── ApplicationsTab.tsx
│   │   │   ├── InterviewTab.tsx
│   │   │   └── ObservabilityTab.tsx
│   │   └── components/
│   │       ├── CVEditor.tsx       # TipTap/Quill inline editor
│   │       ├── CVChatbot.tsx      # Per-CV mini chatbot
│   │       ├── EmailEditor.tsx    # Email body editor
│   │       ├── HITLCard.tsx       # Approval UI component
│   │       └── AgentActivityFeed.tsx
│   └── lib/
│       └── api.ts             # API client
├── docker-compose.yml
├── .env.example
└── README.md
```

---

### ⚙️ ENVIRONMENT VARIABLES NEEDED

```env
# LLM
ANTHROPIC_API_KEY=

# Observability
LANGCHAIN_API_KEY=
LANGCHAIN_TRACING_V2=true
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
LANGFUSE_HOST=https://cloud.langfuse.com

# Integrations
APIFY_API_KEY=
GMAIL_CLIENT_ID=
GMAIL_CLIENT_SECRET=
GMAIL_REDIRECT_URI=
HUNTER_IO_API_KEY=

# Database
POSTGRES_URL=
REDIS_URL=

# App
NEXT_PUBLIC_API_URL=http://localhost:8000
```

---

### ✅ BUILD SEQUENCE (Step-by-Step)

Follow this order to avoid circular dependencies:

1. **Set up infrastructure**: PostgreSQL, Redis, Docker Compose
2. **Study BowJob repo**: Understand CV parsing and JD-matching logic
3. **Build memory layer**: Long-term (Postgres schemas) + short-term (Redis)
4. **Build Orchestrator Agent**: Intent detection, agent routing, LangGraph state machine
5. **Build Job Search Agent** + Apify integration + deduplication logic
6. **Build Resume Agent** + CV tailoring using BowJob-inspired prompts
7. **Build HITL Manager** + CV Review Panel (CVs Tab with inline editor + chatbot)
8. **Build Apply Agent** + Gmail integration + email drafting
9. **Build HITL for Emails** + Email Review Panel (Applications Tab)
10. **Build Interview Prep Agent**
11. **Add LangSmith + Langfuse** observability throughout all agents
12. **Build Frontend** (Next.js) — single-window interface with all tabs
13. **End-to-end testing**: Full pipeline + partial flows + HITL interrupts
14. **Deploy**: Docker Compose for local, Railway/Render for cloud

---

### 🚨 DESIGN PRINCIPLES — NON-NEGOTIABLE

1. **Digital FTE, not SaaS**: The user delegates; the system executes. User's job is to review and approve, not click through forms.
2. **Full transparency**: Every agent action is logged, visible, and interruptable.
3. **HITL before irreversible actions**: CVs and emails must be approved before any send.
4. **Partial flows supported**: User can trigger any single agent in isolation.
5. **One interface**: Everything in one window. No page navigation. Tabs only.
6. **No black boxes**: LangSmith + Langfuse must capture 100% of agent traces.

---

*Generated for the Candidates FTE project. Build this as a Digital FTE — a delegated digital worker, not a tool.*
