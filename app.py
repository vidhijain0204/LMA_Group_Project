import streamlit as st
from agents.diagnosis_agent import DiagnosisAgent  # Adjust import path as needed
from agents.course_retriever import CourseRetriever
from agents.tutor_agent import TutorAgent

# Filepaths configuration
TOPICS_PATH = "data/common/topics.json"
MCQS_PATH = "data/diagnosis/static_diagnosis_mcq_list.json"
DB_PATH = "data/skb_database.json"
CORPUS_PATH = "data/common/corpus.jsonl"

st.set_page_config(page_title="Python Learning Platform", layout="wide")

@st.cache_resource(show_spinner="Loading diagnosis agent and questions...")
def get_diagnosis_agent():
    return DiagnosisAgent(
        topics_filepath=TOPICS_PATH,
        mcqs_filepath=MCQS_PATH,
        db_filepath=DB_PATH
    )


@st.cache_resource(show_spinner="Loading course material and tutor...")
def get_tutor_agent():
    retriever = CourseRetriever(corpus_filepath=CORPUS_PATH)
    return TutorAgent(db_filepath=DB_PATH, retriever=retriever)


agent = get_diagnosis_agent()

# Load topic name lookup dictionary for clean UI rendering
topic_names = {t["topic_id"]: t["name"] for t in agent.topics}
topic_name_to_id = {t["name"]: t["topic_id"] for t in agent.topics}

# --- SIDEBAR: Student Management ---
with st.sidebar:
    st.header("Student Portal")
    
    # Default or input student ID
    suggested_id = agent.new_student_id()
    student_id = st.text_input("Student ID", value=suggested_id).strip()
    
    has_taken_test = agent.student_exists(student_id)
    
    if has_taken_test:
        st.success(f"Status: KB found for **{student_id}**")
    else:
        st.info(f"Status: New student session (**{student_id}**)")

    st.divider()
    st.markdown("### Platform Info")
    st.markdown(f"* **Total Questions:** {len(agent.mcqs)}")
    st.markdown(f"* **Covered Topics:** {len(agent.topics)}")
    st.markdown("* **Passing Threshold:** 70%")

# Retrieve existing student profile if saved in DB
existing_skb = None
if has_taken_test:
    for entry in agent._load_db():
        if entry.get("student_id") == student_id:
            existing_skb = entry
            break

# --- MAIN CONTENT AREA ---
if has_taken_test and existing_skb:
    # --- TUTOR AGENT PAGE (POST-DIAGNOSIS) ---
    st.title(f"🎓 Tutor Agent — Student: {student_id}")
    st.caption("Personalized learning session based on your diagnostic profile.")

    # Quick profile summary bar
    col1, col2, col3 = st.columns(3)
    with col1:
        strengths = existing_skb.get("strengths", [])
        st.metric("Strengths Identified", len(strengths))
    with col2:
        developing = existing_skb.get("developing", [])
        st.metric("Developing Topics", len(developing))
    with col3:
        weaknesses = existing_skb.get("weaknesses", [])
        st.metric("Weaknesses Targeted", len(weaknesses))

    st.divider()

    if st.session_state.get("tutor_student_id") != student_id:
        st.session_state.tutor_student_id = student_id
        st.session_state.tutor_messages = []

    for message in st.session_state.tutor_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Ask a Python question")
    if question:
        history = list(st.session_state.tutor_messages)
        st.session_state.tutor_messages.append(
            {"role": "user", "content": question}
        )
        try:
            with st.spinner("Finding relevant course material..."):
                answer = get_tutor_agent().explain(
                    student_id=student_id,
                    user_input=question,
                    chat_history=history,
                )
            st.session_state.tutor_messages.append(
                {"role": "assistant", "content": answer}
            )
            st.rerun()
        except Exception as error:
            st.error(f"The tutor could not answer this question: {error}")

else:
    # --- DIAGNOSTIC ASSESSMENT FORM ---
    st.title("📝 Python Diagnostic Assessment")
    st.caption("Complete the assessment below to evaluate your foundational Python knowledge and build your personalized profile.")
    
    questions = agent.get_questions()
    
    if not questions:
        st.error("No diagnostic questions found. Please check your `static_diagnosis_mcq_list.json` path.")
    else:
        with st.form("diagnosis_form"):
            st.info("Answer all questions below. Unanswered or blank questions will be marked incorrect.")
            
            student_answers = {}
            
            for idx, q in enumerate(questions, start=1):
                prob_id = q["problem_id"]
                st.markdown(f"### Q{idx}: {q['body']}")
                
                # Render options as radio buttons
                options = q["option"]
                choice = st.radio(
                    label=f"Select option for {prob_id}",
                    options=options,
                    index=None,
                    key=f"q_{prob_id}",
                    label_visibility="collapsed"
                )
                
                if choice is not None:
                    student_answers[prob_id] = choice
                
                st.divider()

            submit_test = st.form_submit_button("🚀 Submit Assessment & Launch Tutor Agent")
            
            if submit_test:
                if not student_id:
                    st.warning("Please enter a valid Student ID before submitting.")
                elif agent.student_exists(student_id):
                    st.warning(f"Student ID {student_id} already exists in the database.")
                else:
                    with st.spinner("Evaluating responses and building Knowledge Base..."):
                        skb = agent.administer_test(student_id, student_answers)
                    st.success(f"Assessment successfully submitted and saved for **{student_id}**!")
                    # Rerun immediately to transition into the Tutor Agent view using this student_id
                    st.rerun()