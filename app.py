# =====================================================================
# STREAMLIT MULTI-AGENT TUTORING SYSTEM (`app.py`)
# =====================================================================
import streamlit as st
import json
from pathlib import Path

from agents.workflow import AdaptiveTutorWorkflow
from agents.orchestrator import load_student_kb, SKB_FILE_PATH
from agents.diagnosis_agent import DiagnosisAgent
from agents.root_cause_agent import (
    DEFAULT_PRESET, METHOD_LABELS, METHODS, PARAM_LABELS, PARAMS, WEIGHT_PRESETS, SIGNS,
)

# =====================================================================
# PAGE CONFIGURATION
# =====================================================================
st.set_page_config(
    page_title="AI Python Tutoring System",
    page_icon="🎓",
    layout="wide",
)

# Initialize workflow and diagnosis agent.
# Streamlit re-runs this whole script on every click, so the workflow is created once and cached:
# a new workflow would have an empty checkpointer and could not resume a paused diagnosis or quiz.
@st.cache_resource
def get_workflow() -> AdaptiveTutorWorkflow:
    return AdaptiveTutorWorkflow(db_filepath=SKB_FILE_PATH)


workflow = get_workflow()
diag_agent = DiagnosisAgent(db_filepath=SKB_FILE_PATH)

# =====================================================================
# SESSION STATE INITIALIZATION
# =====================================================================
if "student_id" not in st.session_state:
    st.session_state.student_id = "S001"
if "thread_id" not in st.session_state:
    st.session_state.thread_id = "S001-session"
if "messages" not in st.session_state:
    st.session_state.messages = []
if "waiting_for_interrupt" not in st.session_state:
    st.session_state.waiting_for_interrupt = None
# Agent steps of a start() call that paused for a diagnosis/quiz; shown with the final answer.
if "pending_trace" not in st.session_state:
    st.session_state.pending_trace = []

# =====================================================================
# AGENT FLOW & ROOT CAUSE RENDERING
# =====================================================================
AGENT_ICONS = {
    "Orchestrator": "🧭",
    "Diagnosis Agent": "🩺",
    "Pedagogy Controller": "🧠",
    "Tutor Agent": "👩‍🏫",
    "Assessment Agent": "🎯",
    "Error Analysis Agent": "🧪",
    "Root Cause Agent": "🔎",
}


def render_trace(trace: list) -> None:
    """Collapsible timeline of the agents that ran, with the input and output of each step."""
    if not trace:
        return
    total_ms = sum(step.get("duration_ms") or 0 for step in trace)
    path = []
    for step in trace:
        if not path or path[-1] != step["agent"]:
            path.append(step["agent"])
    with st.expander(f"🛰️ Agent flow · {len(trace)} steps · {total_ms / 1000:.1f}s"):
        st.caption(" → ".join(f"{AGENT_ICONS.get(a, '⚙️')} {a}" for a in path))
        for index, step in enumerate(trace, 1):
            with st.container(border=True):
                duration = f" · `{step['duration_ms']} ms`" if step.get("duration_ms") is not None else ""
                st.markdown(
                    f"**{index}. {AGENT_ICONS.get(step['agent'], '⚙️')} {step['agent']}** — "
                    f"{step['title']}{duration}"
                )
                tab_in, tab_out = st.tabs(["📥 Input", "📤 Output"])
                with tab_in:
                    st.json(step.get("input") or {}, expanded=1)
                with tab_out:
                    st.json(step.get("output") or {}, expanded=1)


def render_root_cause_details(details: dict) -> None:
    """Hypothesis parameters and what each method / weight set picked, for evaluation."""
    if not details:
        return
    with st.expander("📐 Root-cause evidence & method comparison"):
        hypotheses = details.get("hypotheses") or []
        if hypotheses:
            st.markdown("**Hypotheses (top 3 candidate subtopics)**")
            st.dataframe(
                [
                    {
                        "Hypothesis": f"{h['subtopic_id']} {h['name']}",
                        **{("PM'" if k == "PM" else k): h["params"][k] for k in PARAMS},
                        "Score": h["score"],
                        "P(H)": h["probability"],
                    }
                    for h in hypotheses
                ],
                hide_index=True, width="stretch",
            )
        comparison = details.get("comparison")
        if comparison:
            primary = (details.get("root_cause") or {}).get("subtopic_id")
            rows = []
            for group, label in (("methods", "Method"), ("presets", "Weight preset"), ("ablations", "Ablation")):
                for name, entry in comparison[group].items():
                    root = entry if group == "methods" else entry.get("root_cause")
                    rows.append({
                        "Type": label,
                        "Name": METHOD_LABELS.get(name, name),
                        "Root cause": f"{root['subtopic_id']} {root['name']}" if root else "—",
                        "P": root.get("probability") if root else None,
                        "Agrees with selected": bool(root) and root["subtopic_id"] == primary,
                    })
            st.markdown("**What each method / weight combination picks on this quiz**")
            st.dataframe(rows, hide_index=True, width="stretch")


def render_message(message: dict) -> None:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        render_root_cause_details(message.get("root_cause_details"))
        render_trace(message.get("trace"))

# =====================================================================
# SIDEBAR: STUDENT PROFILE & CONTROLS
# =====================================================================
with st.sidebar:
    st.header("🎓 Student Portal")
    
    input_id = st.text_input("Enter Student ID:", value=st.session_state.student_id)
    
    if st.button("Load / Switch Student"):
        st.session_state.student_id = input_id.strip()
        st.session_state.thread_id = f"{input_id.strip()}-session"
        st.session_state.messages = []
        st.session_state.waiting_for_interrupt = None
        st.session_state.pending_trace = []
        st.rerun()

    st.divider()

    # Root-cause settings: which method decides the stored root cause, and with which weights.
    with st.expander("🔎 Root-cause settings"):
        method = st.selectbox(
            "Method", METHODS, index=METHODS.index("hypothesis"),
            format_func=lambda m: METHOD_LABELS[m],
        )
        preset_names = list(WEIGHT_PRESETS)
        preset = st.selectbox("Weight preset", preset_names, index=preset_names.index(DEFAULT_PRESET))
        custom = st.toggle("Custom weights", value=False)
        weights = dict(WEIGHT_PRESETS[preset])
        if custom:
            for key in PARAMS:
                weights[key] = st.slider(
                    f"{'−' if SIGNS[key] < 0 else '+'} {PARAM_LABELS[key]} ({key})",
                    0.0, 1.0, float(weights[key]), 0.05, key=f"w_{key}",
                )
        st.caption(
            "Score(H) = " + " ".join(
                f"{'−' if SIGNS[k] < 0 else '+'} {weights[k]:.2f}·{'PM′' if k == 'PM' else k}" for k in PARAMS
            ).lstrip("+ ")
        )
        workflow.root_cause_method = method
        workflow.root_cause_preset = preset
        workflow.root_cause_weights = weights if custom else None

    st.divider()

    kb = load_student_kb(st.session_state.student_id, SKB_FILE_PATH)
    if kb:
        st.success("Status: Registered Student")
        st.markdown(f"**Strengths:** {', '.join(kb.get('strengths', [])) or 'None yet'}")
        st.markdown(f"**Weaknesses:** {', '.join(kb.get('weaknesses', [])) or 'None yet'}")

        pedagogy_history = kb.get("pedagogy_history", [])
        if pedagogy_history:
            with st.expander("Adaptive teaching history"):
                for event in reversed(pedagogy_history[-8:]):
                    topic = event.get("topic_id", "Unknown topic")
                    if event.get("event") == "lesson":
                        outcome = event.get("assessment_accuracy")
                        evidence = (
                            f" | next quiz accuracy: {outcome:.0%}"
                            if outcome is not None
                            else " | awaiting assessment"
                        )
                        st.markdown(
                            f"- **{event.get('strategy', 'unknown').replace('_', ' ').title()}** "
                            f"on {topic}{evidence}"
                        )
                    elif event.get("event") == "assessment":
                        accuracy = event.get("accuracy")
                        result = f"{accuracy:.0%}" if accuracy is not None else "not topic-aligned"
                        st.markdown(
                            f"- Quiz on {topic}: **{result}**; "
                            f"{len(event.get('error_signals', []))} error signal(s)"
                        )

        root_causes = kb.get("root_cause_history", [])
        if root_causes:
            with st.expander("Root causes of quiz errors"):
                for entry in reversed(root_causes[-5:]):
                    summary = entry.get("summary") or (
                        f"{entry.get('subtopic_id')} {entry.get('name')} → {entry.get('reason', '')}"
                    )
                    st.markdown(f"- Quiz on {entry.get('quiz_topic_id', '?')}: {summary}")

        with st.expander("View Topic Scores"):
            for tid, score in kb.get("topics", {}).items():
                st.progress(score, text=f"{tid}: {score}")
    else:
        st.warning("Status: New Student / Unregistered")
        st.info("Type a message or click below to trigger the diagnostic onboarding test.")
        if st.button("Start Diagnostic Test", disabled=bool(st.session_state.waiting_for_interrupt)):
            # The orchestrator sends every new student to the diagnosis, whatever the message says.
            result = workflow.start(
                student_id=st.session_state.student_id,
                user_input="I am a new student. Please start my diagnostic test.",
                thread_id=st.session_state.thread_id,
            )
            if result.get("status") == "waiting_for_interrupt":
                st.session_state.waiting_for_interrupt = result["payload"]
                st.session_state.pending_trace = result.get("trace", [])
                st.rerun()

    st.divider()
    if st.button("Reset Chat / Session"):
        st.session_state.messages = []
        st.session_state.waiting_for_interrupt = None
        st.session_state.pending_trace = []
        st.rerun()

# =====================================================================
# MAIN CHAT & INTERRUPT INTERFACE
# =====================================================================
st.title("🤖 AI Python Tutoring Assistant")
st.markdown(f"Chatting as **{st.session_state.student_id}**. Ask questions, request explanations, or ask for a quiz!")

# Display chat history
for message in st.session_state.messages:
    render_message(message)

# Check if graph is interrupted (e.g. waiting for diagnosis test or quiz answers)
interrupt_payload = st.session_state.waiting_for_interrupt

if interrupt_payload:
    int_type = interrupt_payload.get("type")
    
    if int_type == "diagnosis":
        st.warning("📝 **Diagnostic Assessment Required** (New Student Onboarding)")
        questions = interrupt_payload.get("questions", [])
        
        with st.form("interrupt_diagnosis_form"):
            submitted_answers = {}
            for idx, q in enumerate(questions, 1):
                st.subheader(f"Question {idx}: {q['body']}")
                options = q.get("options", q.get("option", []))
                choice = st.radio(
                    f"Select answer for Q{idx}:",
                    options=options,
                    key=f"diag_q_{q.get('problem_id', idx)}"
                )
                submitted_answers[q.get('problem_id')] = choice
                st.divider()

            submit_diag = st.form_submit_button("Submit Diagnostic Answers")
            if submit_diag:
                with st.spinner("Processing diagnostic test and building Knowledge Base..."):
                    res = workflow.resume(st.session_state.thread_id, submitted_answers)
                    st.session_state.waiting_for_interrupt = None
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": "🎉 **Diagnostic completed** — your knowledge base is ready. "
                                   "Ask a question or request a quiz!",
                        "trace": st.session_state.pending_trace + res.get("trace", []),
                    })
                    st.session_state.pending_trace = []
                    st.rerun()

    elif int_type == "quiz":
        st.warning(f"🎯 **Adaptive Quiz Challenge** — Topic: {interrupt_payload.get('topic_focus')}")
        st.markdown(interrupt_payload.get("intro_text", ""))
        
        questions = interrupt_payload.get("questions", [])
        with st.form("interrupt_quiz_form"):
            quiz_answers = {}
            for idx, q in enumerate(questions, 1):
                st.subheader(f"Quiz Question {idx}: {q.get('prob_body')}")
                options = q.get("options", {})
                # options is a dict like {"A": "...", "B": "..."}
                option_labels = list(options.keys())
                option_texts = [f"{k}: {v}" for k, v in options.items()]
                
                choice_text = st.radio(
                    f"Select answer for Q{idx}:",
                    options=option_texts,
                    key=f"quiz_q_{q.get('prob_id')}"
                )
                # Extract chosen key (e.g., 'A' from 'A: The Python interpreter')
                chosen_key = choice_text.split(":")[0].strip() if choice_text else None
                quiz_answers[q.get('prob_id')] = chosen_key
                st.divider()

            submit_quiz = st.form_submit_button("Submit Quiz Answers")
            if submit_quiz:
                with st.spinner("Grading quiz and updating progress..."):
                    res = workflow.resume(st.session_state.thread_id, quiz_answers)
                    st.session_state.waiting_for_interrupt = None
                    
                    score_msg = (
                        f"📊 **Quiz Completed!** Score: {res.get('quiz_score')} "
                        f"(Accuracy: {res.get('accuracy', 0.0):.0%})"
                    )
                    root_cause = res.get("root_cause")
                    if root_cause:
                        # One line: topic -> subtopic -> reason.
                        score_msg += (
                            f"\n\n🔎 **Actual reason for the MCQ errors** "
                            f"_({METHOD_LABELS[res.get('root_cause_method', 'hypothesis')]})_\n\n"
                            f"> **{root_cause['topic_id']} {root_cause.get('topic_name', '')}** → "
                            f"**{root_cause['subtopic_id']} {root_cause['name']}** → {root_cause['reason']}"
                        )
                    elif res.get("accuracy") == 1.0:
                        score_msg += "\n\n✅ No wrong answers — nothing to diagnose."
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": score_msg,
                        "root_cause_details": root_cause and {
                            "root_cause": root_cause,
                            "hypotheses": res.get("hypotheses"),
                            "comparison": res.get("comparison"),
                        },
                        "trace": st.session_state.pending_trace + res.get("trace", []),
                    })
                    st.session_state.pending_trace = []
                    st.rerun()

else:
    # Standard Chat Input
    if user_prompt := st.chat_input("Ask your tutor a question or request a quiz..."):
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        with st.spinner("Processing request..."):
            result = workflow.start(
                student_id=st.session_state.student_id,
                user_input=user_prompt,
                thread_id=st.session_state.thread_id
            )

            # Check if workflow hit an interrupt (diagnosis or quiz)
            if result.get("status") == "waiting_for_interrupt":
                st.session_state.waiting_for_interrupt = result["payload"]
                st.session_state.pending_trace = result.get("trace", [])
                st.rerun()
            else:
                # Normal completion (Tutor response or summary)
                response_text = result.get("response_text", "I am here to help you learn Python!")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response_text,
                    "trace": result.get("trace", []),
                })
                st.rerun()