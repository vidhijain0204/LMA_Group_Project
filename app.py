# =====================================================================
# STREAMLIT MULTI-AGENT TUTORING SYSTEM (`app.py`)
# =====================================================================
import streamlit as st
import json
from pathlib import Path
from langchain_core.messages import HumanMessage, AIMessage

# Import your custom modules
from agents.diagnosis_agent import DiagnosisAgent  # Assuming your DiagnosisAgent class is saved here
from agents.orchestrator import create_orchestrator_node, load_student_kb, SKB_FILE_PATH, TOPICS_FILE_PATH
from agents.tutor_agent import create_tutor_node

# =====================================================================
# PAGE CONFIGURATION
# =====================================================================
st.set_page_config(
    page_title="AI Python Tutoring System",
    page_icon="🎓",
    layout="wide",
)

# Initialize Diagnosis Agent
diag_agent = DiagnosisAgent()

# =====================================================================
# SESSION STATE INITIALIZATION
# =====================================================================
if "student_id" not in st.session_state:
    st.session_state.student_id = "S001"
if "student_kb" not in st.session_state:
    st.session_state.student_kb = None
if "in_diagnosis" not in st.session_state:
    st.session_state.in_diagnosis = False
if "messages" not in st.session_state:
    st.session_state.messages = []
if "diagnosis_answers" not in st.session_state:
    st.session_state.diagnosis_answers = {}

# =====================================================================
# SIDEBAR: STUDENT PROFILE & CONTROLS
# =====================================================================
with st.sidebar:
    st.header("🎓 Student Portal")
    
    # Input Student ID
    input_id = st.text_input("Enter Student ID:", value=st.session_state.student_id)
    
    if st.button("Load / Switch Student"):
        st.session_state.student_id = input_id.strip()
        st.session_state.student_kb = load_student_kb(st.session_state.student_id, SKB_FILE_PATH)
        st.session_state.in_diagnosis = not diag_agent.student_exists(st.session_state.student_id)
        st.session_state.messages = []  # Clear chat history on switch
        st.rerun()

    st.divider()

    # Display Student Status / KB Info
    if diag_agent.student_exists(st.session_state.student_id):
        st.success(f"Status: Registered Student")
        kb = load_student_kb(st.session_state.student_id, SKB_FILE_PATH)
        if kb:
            st.markdown(f"**Strengths:** {', '.join(kb.get('strengths', [])) or 'None yet'}")
            st.markdown(f"**Weaknesses:** {', '.join(kb.get('weaknesses', [])) or 'None yet'}")
            
            with st.expander("View Topic Scores"):
                for tid, score in kb.get("topics", {}).items():
                    st.progress(score, text=f"{tid}: {score}")
    else:
        st.warning(f"Status: New Student (Diagnosis Required)")
        if st.button("Start Diagnostic Test"):
            st.session_state.in_diagnosis = True
            st.rerun()

    st.divider()
    if st.button("Reset Chat / Session"):
        st.session_state.messages = []
        st.rerun()

# =====================================================================
# MAIN VIEW 1: DIAGNOSIS MODE (For New Students)
# =====================================================================
if not diag_agent.student_exists(st.session_state.student_id) or st.session_state.in_diagnosis:
    st.title("📝 Initial Diagnostic Assessment")
    st.markdown(f"Welcome **{st.session_state.student_id}**! Since you are a new student, please complete this brief diagnostic questionnaire so we can build your initial Knowledge Base.")

    questions = diag_agent.get_questions()
    
    with st.form("diagnosis_form"):
        submitted_answers = {}
        for idx, q in enumerate(questions, 1):
            st.subheader(f"Question {idx}: {q['body']}")
            
            # Render options based on type
            options = q.get("option", [])
            # Assuming options is a list of strings or dicts
            choice = st.radio(
                f"Select your answer for Q{idx}:",
                options=options,
                key=f"q_{q['problem_id']}"
            )
            submitted_answers[q["problem_id"]] = choice
            st.divider()

        submit_test = st.form_submit_button("Submit Diagnostic Test")
        
        if submit_test:
            try:
                # Administer test and generate SKB using your DiagnosisAgent
                new_kb = diag_agent.administer_test(st.session_state.student_id, submitted_answers)
                st.session_state.student_kb = new_kb.model_dump(mode="json")
                st.session_state.in_diagnosis = False
                st.success("🎉 Diagnostic test completed successfully! Initializing your tutoring session...")
                st.rerun()
            except Exception as e:
                st.error(f"Error processing test submission: {e}")

# =====================================================================
# MAIN VIEW 2: INTERACTIVE TUTORING CHAT (Orchestrator + Tutor)
# =====================================================================
else:
    st.title("🤖 AI Python Tutoring Assistant")
    st.markdown(f"Chatting as **{st.session_state.student_id}**. Ask questions, request code explanations, or practice quizzes!")

    # Ensure student KB is loaded in session state
    if not st.session_state.student_kb:
        st.session_state.student_kb = load_student_kb(st.session_state.student_id, SKB_FILE_PATH)

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User chat input
    if user_prompt := st.chat_input("Ask your tutor a question (e.g., 'How do if/else statements work?')..."):
        # 1. Append user message
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        # 2. Invoke Orchestrator Node to route intent & topic
        with st.spinner("Orchestrator is analyzing your request..."):
            orchestrator_node = create_orchestrator_node()
            
            state = {
                "student_id": st.session_state.student_id,
                "messages": [HumanMessage(content=m["content"]) if m["role"]=="user" else AIMessage(content=m["content"]) for m in st.session_state.messages],
                "student_kb": st.session_state.student_kb,
                "topic_focus": [],
                "last_active_agent": None,
                "orchestrator_instruction": "",
                "orchestrator_reasoning": "",
            }

            # Run orchestrator
            routing_command = orchestrator_node(state)
            chosen_agent = routing_command.goto
            updated_state_updates = routing_command.update

            # Extract orchestrator metadata
            topic_focus = updated_state_updates.get("topic_focus", ["Python Basics"])
            instruction = updated_state_updates.get("orchestrator_instruction", "Explain concept.")
            reasoning = updated_state_updates.get("orchestrator_reasoning", "Direct routing.")

            st.sidebar.info(f"🧭 **Route:** `{chosen_agent}`\n\n📌 **Topic:** {topic_focus}\n\n💡 **Reason:** {reasoning}")

        # 3. Route to Specialist Agent (Tutor or Assessment)
        with st.spinner(f"Agent ({chosen_agent}) is generating response..."):
            if chosen_agent == "tutor_agent":
                tutor_node = create_tutor_node(enable_kb_context=True)
                agent_output_state = tutor_node({
                    **state,
                    "topic_focus": topic_focus,
                    "orchestrator_instruction": instruction
                })
                assistant_response = agent_output_state["messages"][-1].content

            elif chosen_agent == "assessment_agent":
                # Placeholder for Assessment Agent response
                assistant_response = f"*(Assessment Agent)* Here is a coding quiz challenge for **{', '.join(topic_focus)}** based on your instruction: {instruction}"
            else:
                assistant_response = "I am here to help you learn Python! What would you like to explore next?"

        # 4. Append and display assistant response
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})
        with st.chat_message("assistant"):
            st.markdown(assistant_response)