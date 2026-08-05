import streamlit as st
from datetime import date, timedelta
import backend

APP_TITLE = "AI Travel Planner"
APP_SUBTITLE = "Get a personalized itinerary for your next US city trip"
CITIES = ["Boston", "New York City", "Los Angeles", "Washington DC", "Orlando"]
INTEREST_OPTIONS = [
    "Food & Dining", "Museums & History", "Nightlife", "Outdoors & Nature",
    "Shopping", "Family-Friendly", "Art & Culture", "Sports",
]
MAX_INPUT_CHARS = 2000

st.set_page_config(page_title=APP_TITLE, layout="centered")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "trip_submitted" not in st.session_state:
    st.session_state.trip_submitted = False

if "error_count" not in st.session_state:
    st.session_state.error_count = 0


def reset_trip():
    st.session_state.messages = []
    st.session_state.trip_submitted = False
    st.session_state.error_count = 0

with st.sidebar:
    st.header(APP_TITLE)
    st.caption(APP_SUBTITLE)
    st.divider()
    st.write(
        "Fill out your trip details to get a starting itinerary, then keep "
        "chatting to refine it — swap out a day, add restaurants, adjust "
        "for budget, etc."
    )
    st.divider()
    if st.button("Start a new trip", use_container_width=True):
        reset_trip()
        st.rerun()
    st.divider()
    st.caption(f"Messages this session: {len(st.session_state.messages)}")


st.title(APP_TITLE)
st.caption(APP_SUBTITLE)

def send_message(prompt_text: str, display_text: str | None = None):
    """
    Sends prompt_text to the backend, appends to history, and renders
    the exchange. display_text lets us show a friendlier version of the
    message in the chat bubble than what's actually sent to the model
    (used for the trip-form submission).
    """
    st.session_state.messages.append(
        {"role": "user", "content": prompt_text, "sources": None}
    )
    with st.chat_message("user"):
        st.markdown(display_text if display_text else prompt_text)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("Planning...")
        try:
            result = backend.generate_response(
                user_input=prompt_text,
                history=st.session_state.messages[:-1],
            )
            answer = (result.get("answer") or "").strip()
            sources = result.get("sources")

            if not answer:
                raise ValueError("Empty response from backend")

            placeholder.markdown(answer)
            if sources:
                with st.expander("Sources"):
                    for src in sources:
                        st.markdown(f"- {src}")

            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": sources}
            )
            st.session_state.error_count = 0

        except Exception as e:
            st.session_state.error_count += 1
            friendly_msg = "Sorry, something went wrong generating a response. Please try again."
            if st.session_state.error_count >= 3:
                friendly_msg += " If this keeps happening, please refresh the page."
            placeholder.markdown(f"! {friendly_msg}")
            st.session_state.messages.append(
                {"role": "assistant", "content": f"! {friendly_msg}", "sources": None}
            )
            if st.session_state.get("debug_mode"):
                st.exception(e)

def validate_trip_form(city, arrival, departure, budget, interests) -> str | None:
    if not city:
        return "Please select a city."
    if arrival is None or departure is None:
        return "Please select both arrival and departure dates."
    if departure <= arrival:
        return "Departure date must be after the arrival date."
    if budget is None or budget <= 0:
        return "Please enter a budget greater than $0."
    if not interests:
        return "Please select at least one interest."
    return None


def validate_chat_input(text: str) -> str | None:
    if not text or not text.strip():
        return "Please enter a message before sending."
    if len(text) > MAX_INPUT_CHARS:
        return f"Message is too long ({len(text)} chars). Please keep it under {MAX_INPUT_CHARS}."
    return None

if not st.session_state.trip_submitted:
    with st.form("trip_form"):
        st.subheader("Plan your trip")

        city = st.selectbox("Destination city", CITIES)

        col1, col2 = st.columns(2)
        with col1:
            arrival = st.date_input("Arrival date", value=date.today() + timedelta(days=14))
        with col2:
            departure = st.date_input("Departure date", value=date.today() + timedelta(days=18))

        budget = st.number_input("Total budget (USD)", min_value=0, step=50, value=1000)

        interests = st.multiselect("Interests", INTEREST_OPTIONS)

        submitted = st.form_submit_button("Generate", use_container_width=True)

    if submitted:
        error = validate_trip_form(city, arrival, departure, budget, interests)
        if error:
            st.warning(error)
        else:
            num_days = (departure - arrival).days
            prompt = (
                f"Plan a {num_days}-day trip to {city} from {arrival} to {departure}. "
                f"Total budget: ${budget}. Interests: {', '.join(interests)}. "
                f"Please create a day-by-day itinerary."
            )
            display = (
                f"**Trip request:** {city}, {arrival} → {departure} "
                f"({num_days} days) · Budget ${budget} · {', '.join(interests)}"
            )
            st.session_state.trip_submitted = True
            send_message(prompt, display_text=display)
            st.rerun()

else:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("Sources"):
                    for src in msg["sources"]:
                        st.markdown(f"- {src}")

    follow_up = st.chat_input("Ask for changes — e.g. \"make day 2 more relaxed\"")
    if follow_up is not None:
        error = validate_chat_input(follow_up)
        if error:
            st.warning(error)
        else:
            send_message(follow_up)