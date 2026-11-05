
"""
HR Policy Assistant
Streamlit entry point for the secure HR RAG application.

Run:
    streamlit run app.py
"""

import uuid
import logging

import streamlit as st

from hr_assistant import config
from hr_assistant.logging_config import configure_logging
from hr_assistant.pipeline import (
    INPUT_BLOCKED_MESSAGE,
    OUTPUT_BLOCKED_MESSAGE,
    ask,
    build_hr_assistant,
)
from hr_assistant.guardrails import GuardrailProviderError
from hr_assistant.tracing import check_langsmith_tracing


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="HR Policy Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==================================================
# LOGGING AND TRACING
# ==================================================

configure_logging()
check_langsmith_tracing()

logger = logging.getLogger(__name__)


# ==================================================
# APPLICATION HEADER
# ==================================================

st.title("🤖 HR Policy Assistant")

st.markdown(
    """
    Your intelligent assistant for company HR policies.

    Ask questions about leave, holidays, work-from-home policies,
    reimbursements, probation, notice periods, and more.
    """
)

st.divider()


# ==================================================
# AUTHENTICATION
# ==================================================

def auth_configured() -> bool:
    """Check whether Streamlit Google OAuth is configured."""

    try:
        return bool(st.secrets.get("auth"))
    except Exception:
        return False


AUTH_ENABLED = auth_configured()


if AUTH_ENABLED:

    if not st.user.is_logged_in:

        st.info("Please log in using your company Google account.")

        st.button(
            "Log in with Google",
            on_click=st.login,
            args=("google",),
            type="primary",
        )

        st.stop()

    if st.user.email not in config.ALLOWED_EMPLOYEE_EMAILS:

        st.error(
            f"Access denied. The account {st.user.email} "
            "is not on the approved employee list."
        )

        st.button("Log out", on_click=st.logout)

        st.stop()


# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:

    st.title("HR Assistant")

    st.caption("Secure RAG-powered policy assistant")

    st.divider()

    st.subheader("Application status")

    st.success("RAG pipeline configured")

    st.info(
        f"Safety provider: {config.GUARDRAIL_PROVIDER}"
    )

    if AUTH_ENABLED:

        st.success("Google authentication enabled")

        st.caption(f"Logged in as: {st.user.email}")

        st.button("Log out", on_click=st.logout)

    else:

        st.warning("Local development mode")

        st.caption(
            "Google authentication is not configured."
        )

    st.divider()

    st.subheader("Conversation")

    if st.button(
        "＋ New conversation",
        use_container_width=True,
    ):

        st.session_state.thread_id = str(uuid.uuid4())

        st.session_state.messages = []

        st.rerun()

    st.caption(
        "Starting a new conversation creates a separate "
        "conversation thread."
    )

    st.divider()

    st.caption("Powered by")

    st.markdown(
        """
        - LangGraph
        - Gemini
        - Qdrant
        - Jina AI
        - LangSmith
        """
    )


# ==================================================
# ASSISTANT INITIALIZATION
# ==================================================

@st.cache_resource(
    show_spinner="Initializing HR assistant..."
)
def get_agent():

    return build_hr_assistant()


try:

    agent, cache = get_agent()

except Exception:

    logger.exception(
        "HR assistant initialization failed."
    )

    st.error(
        "The HR assistant could not be initialized. "
        "Please check your API configuration and service connections."
    )

    st.stop()


# ==================================================
# SESSION STATE
# ==================================================

if "thread_id" not in st.session_state:

    st.session_state.thread_id = str(uuid.uuid4())


if "messages" not in st.session_state:

    st.session_state.messages = []


# ==================================================
# CHAT INTERFACE
# ==================================================

st.subheader("Ask an HR policy question")

st.caption(
    "Answers are generated using the available HR policy documents. "
    "Check the cited policy source when reviewing important information."
)


# Display existing conversation history.

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        if message.get("type") == "error":

            st.error(message["content"])

        elif message.get("type") == "warning":

            st.warning(message["content"])

        else:

            st.markdown(message["content"])


# ==================================================
# PROCESS USER QUESTION
# ==================================================

question = st.chat_input(
    "Ask something about HR policies..."
)


if question:

    with st.chat_message("user"):

        st.markdown(question)

    with st.chat_message("assistant"):

        try:

            with st.spinner(
                "Searching HR policies and generating an answer..."
            ):

                answer = ask(
                    agent,
                    cache,
                    question,
                    thread_id=st.session_state.thread_id,
                )

            if answer == INPUT_BLOCKED_MESSAGE:

                st.warning(answer)

                # Input-blocked questions are not stored
                # in the agent's conversation memory.
                # Keep them out of the visible transcript too.

            elif answer == OUTPUT_BLOCKED_MESSAGE:

                st.warning(answer)

                # The pipeline records the blocked interaction
                # in its conversation memory.
                st.session_state.messages.append(
                    {
                        "role": "user",
                        "content": question,
                    }
                )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "type": "warning",
                    }
                )

            else:

                st.markdown(answer)

                st.session_state.messages.append(
                    {
                        "role": "user",
                        "content": question,
                    }
                )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                    }
                )

        except GuardrailProviderError:

            logger.exception(
                "Safety verification provider unavailable."
            )

            error_message = (
                "The safety verification service is temporarily "
                "unavailable. Your request could not be safely "
                "processed. Please try again later."
            )

            st.error(error_message)

            # Do not add this interaction to the visible transcript:
            # the pipeline did not complete successfully.

        except Exception:

            logger.exception(
                "Unexpected error while processing an HR question."
            )

            error_message = (
                "Something went wrong while processing your question. "
                "Please try again. If the problem continues, "
                "check the application logs."
            )

            st.error(error_message)


# ==================================================
# FOOTER
# ==================================================

st.divider()

st.caption(
    "HR Policy Assistant | RAG-powered question answering | "
    "Responses are grounded in available policy documents."
)
