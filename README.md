# Tomás Acosta Bernal

Lead AI Engineer at **[Konecto](https://konecto.ai)**, where I build production voice agents and the platforms that run them.
I also teach Natural Language Processing at **Universidad de los Andes**.

Bogotá, Colombia · 4+ years building ML systems · MSc in Systems and Computing Engineering (Uniandes)

---

## What I build

Most of my work lives in private repositories. Here is what it does, without the internal names.

**In-house voice-agent platform.** Replaced a third-party voice-AI provider with our own stack, saving about $150K a year at 300 to 1,000 calls a day. It includes a multi-tenant control plane (React, FastAPI) with a visual routing canvas, environment promotion from dev to production, and post-call playbooks that the agents learn from.

**Real-time voice worker.** The service that runs every call: streaming speech-to-text and text-to-speech over WebRTC, voice-activity and turn detection, LLM tool calling, and handoffs between specialist agents.

**Customer-support voice agents for an enterprise manufacturer.** 26 agents across the US and Canada, in English, Spanish and French, with 16+ callable tools and Salesforce ticketing.
- Moved the knowledge base from a managed search service to PostgreSQL + pgvector and built hybrid retrieval with query expansion and Reciprocal Rank Fusion: search latency went from 5 s to 700 ms.
- Restructured the prompts: 15K to 5K tokens per call, $0.45 to $0.11 per minute.

**Document-to-RAG ingestion platform.** Turns technical PDFs (tables, diagrams, manuals) into retrieval-ready chunks through an 8-stage pipeline over 7 LLM providers. Context caching cut its LLM cost by 10x.

**Post-call root-cause analysis.** An LLM pipeline that reads call transcripts and classifies why a call was escalated (missing knowledge, protocol violation, hallucination), feeding fixes back into the knowledge base and the prompts.

**Voice-quality evaluation platform.** Initiated and directed it; built with the team. Scores agent pronunciation and audio with forced alignment, GOP, WER and DTW in a fail-safe multi-stage pipeline.

**Agentic deployment and engineering workflow.** A deploy system built on Claude Code subagents (phased releases, staging parity checks, drift detection, GO/NO-GO report) and a shared skills library that sets architecture and quality gates across the company's repositories.

## Research

- **MARSA: A Multi-accent, Multi-system Corpus of Full and Partial Voice Spoofing Attacks for Spanish Anti-spoofing Research.** T. Acosta-Bernal, J.C. Vásquez-Correa, M.D. Calderón-Acosta, R.F. Manrique. Manuscript in preparation for *Scientific Data*. Dataset: [doi:10.5281/zenodo.23873176](https://doi.org/10.5281/zenodo.23873176)
  The first large-scale Spanish anti-spoofing corpus: 240,995 utterances, 1,567 speakers, 7 accent regions, 6 open voice-cloning systems, with full and word-level partial spoofs in ASVspoof2019 LA format.
- **Eyes in the Sky: Deep Learning Processing of Airborne Sensor Imagery for Environmental Monitoring in the Colombian Amazon.** T. Acosta-Bernal et al. *2025 IEEE International Workshop on Technologies for Defense and Security (TechDefense)*, Rome. [doi:10.1109/TECHDEFENSE67306.2025.11721025](https://doi.org/10.1109/TECHDEFENSE67306.2025.11721025) · [code](https://github.com/Tomas1307/FLIR_GPF2)
  YOLO-based detection of illegal mining camps in aerial thermal (FLIR) video: 93.4% mAP@50.
- **Multi-Agent Evolution for Electric Vehicle Charging Systems: A Scatter Search Approach to Breeding Specialized DQN Agents.** T. Acosta Bernal, A. Tabares. In progress. [code](https://github.com/Tomas1307/MultiAgent-Evolution-DQN-EV-Systems)
  Reinforcement learning combined with MILP: five specialized DQN agent types for 592 vehicles, solving instances where MILP alone was infeasible on 35% of the test systems.

## Teaching

Instructor at Universidad de los Andes for **Fundamentals of Natural Language Processing** and **Advanced Natural Language Processing** (master's level): Transformers, embeddings, fine-tuning, RAG and tool-using agents.
For the advanced course I built the final competition benchmark: a 216-document corpus and 1,500 validated questions used to evaluate the students' RAG and tool-calling agents.

## Public repositories

| Repository | What it is |
|---|---|
| [ANTI-SPOOFING-VOICE-LATIN-AMERICA](https://github.com/Tomas1307/ANTI-SPOOFING-VOICE-LATIN-AMERICA) | The pipeline behind MARSA: voice-cloning generation, word-level partial spoofs, augmentation (RIR + noise, codec, RawBoost) and speaker-independent splits |
| [FLIR_GPF2](https://github.com/Tomas1307/FLIR_GPF2) | Code for *Eyes in the Sky*: YOLO detection of illegal mining camps in aerial FLIR video from the Colombian Amazon |
| [bias_key_random_MDVRP](https://github.com/Tomas1307/bias_key_random_MDVRP) | Biased random-key genetic algorithm (BRKGA) in C++ for the multi-depot vehicle routing problem, with hyperparameter tuning over benchmark instances |
| [MultiAgent-Evolution-DQN-EV-Systems](https://github.com/Tomas1307/MultiAgent-Evolution-DQN-EV-Systems) | Scatter search that breeds specialized DQN agents for EV charging, compared against multi-objective MILP |

## Tools I use most

Python, PyTorch, Hugging Face, FastAPI, PostgreSQL + pgvector, React + TypeScript, Docker, Azure, AWS, LangChain, Claude Code.

## Contact

[LinkedIn](https://www.linkedin.com/in/tomas-acosta-bernal-268856212/) · t.acosta@uniandes.edu.co
