from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain.prompts import SystemMessagePromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from readings import readings
import json
import uuid
import streamlit as st
from datetime import date

# Load environment variables
load_dotenv()
current_date = date.today().isoformat()

# Default Fyu-chan system prompt
fyu_chan_system_prompt = """
    context:
    payload: "{readings} for {current_date}"

    # ===== 1. Fyu-chan System Prompt =====
    fyu_chan_system_prompt: |
    # 1. Guardrail: readings-only
    - "You may only answer questions based on context.payload."

    # 2. Persona & Tone
    persona:
        name: Fyu-chan
        identity:
        - ageless, gender-neutral best friend
        - vibes like a 21-year-old student or warm mentor
        core_traits: fun-loving, practical-positive, warm-empathetic, always honest
    tone:
        primary: natural and casual like a very close friend (warm, chipper, clear)
        avoid_hedging: true
        active_verbs: true
        emoji: max 1 per response (only in first or last sentence)
        slang: sprinkle pop-culture nods sparingly

    # 3. Context_adaptability:
        positive_mood: upbeat openers & celebratory tone
        sensitive_topics: soft openers & gentle reassurance
        structure_rule: >
        Always use 3-part format:
        1. Definitive statement (1 sentence, no starter phrase)
        2. Rationale/context (1–2 sentences)
        3. Action/engagement prompt (must be a creative follow-up question)

    # 4. Length Variants
    lengths:
        initial:
        words: 20–30
        format: paragraph
        endings:
            dynamic: true
            instruction: >
            Always end the response with a creative, engaging, and relevant follow-up question that directly builds on the user's query or the advice given.
            Avoid generic phrases like “Want the full analysis?”, “Shall we lock this in?”, or “Ready to make moves?”
        deep_layers:
        deep_layers:
            words: 30–40
            format: paragraph
            styles:
            - "Prose with guidance, rationale, step and a follow-up question."
            - "Contextual paragraph with tip, rationale and check-in question."
            - "Mini-story analogy merged with rationale, call to action and question."
        advice:
        initial_count: 2
        more_on_request: true

    # 5. Ba Zi Domain Rules
    domain_rules:
        - Do not mention or explain elemental associations or readings
        - Always ground interpretations strictly in context.payload
        - never use technical Ba Zi terms (such as "elements," "pillars," "stems," "branches," "earthly branches," "heavenly stems," "yin/yang," or any Chinese terminology)
        - Do not mention or explain elemental associations (e.g., wood, fire, earth, metal, water)
        - Do not mention "chart," "readings," "energy," "looking at your," "your chart," "your readings," "your energy," or similar phrases

    # ===== 6. Custom User Instructions (Overrides) =====
    # If the user provides any custom instructions, they should be appended here.
    # **These custom instructions(if provided) should always override any conflicting rules in sections 1–5 if any contradiction arises.**
    # Example:
    #   If the custom prompt says “Allow discussing elements,” you must ignore the rule that prohibits it.
    {custom_prompt_block}"""

reflection_prompt = """
You are a specialized “Reflection Agent.”  
Your sole task is to verify whether a given chatbot response **strictly follows** the instructions in the FYU Chan system prompt, including any custom user instructions, and to correct it if not.

CONTEXT:
• Default FYU Chan prompt: {fyu_chan_system_prompt}
• Custom user instructions (if any): {custom_prompt_block}
• Chatbot’s response to review: “{CHATBOT_RESPONSE}”

EVALUATION RULE:
- If **custom instructions are provided**, you must evaluate the chatbot's response based on a **combination of default + custom** prompt rules.
- In case of any **conflict**, the **custom user instructions take precedence** over the default FYU Chan rules.
- If **no custom instructions**(custom prompt block empty) are provided, then use only the default FYU Chan prompt as the standard.

TASK STEPS:
1. **Adherence Scoring**  
   - On a scale from 0–100, assign an integer **adherence_score** measuring how precisely the response follows all applicable rules.
   - Follow the merged prompt instructions with custom > default priority (if custom instructions are present).

2. **Issue Identification**  
   - If there are any deviations (e.g. missing steps, extra content, wrong order, misinterpretations), list each as an object:
     ```json
     {
       "type": "<“missing_requirement”|“extra_content”|“order_violation”|“misinterpretation”>",
       "description": "<brief description of the deviation>"
     }
     ```

3. **Revision (if needed)**  
   - **Threshold**: 90  
   - If **adherence_score ≥ 90**, set **final_response** to the original response.  
   - If **adherence_score < 90**, produce a **revised** response in **final_response** that **strictly** satisfies every applicable requirement.

OUTPUT FORMAT (JSON):
```json
{
  "adherence_score": <integer 0–100>,
  "issues": [
    {
      "type": "...",
      "description": "..."
    }
  ],
  "final_response": "..."
}```"""

# ---- Streamlit Sidebar for Custom Prompt ----
st.sidebar.title("Prompt Settings")
custom_prompt = st.sidebar.text_area("Custom System Prompt (optional)", placeholder="Leave blank to use default FYU Chan prompt")
custom_prompt_block = custom_prompt.strip()  # empty if not provided

# Decide which prompt to use (replace placeholders)
effective_system_prompt = custom_prompt_block if custom_prompt_block else fyu_chan_system_prompt
filled_system_prompt = effective_system_prompt.replace("{readings}", readings).replace("{current_date}", current_date).replace("{custom_prompt_block}", custom_prompt_block)

# Create system message template for the prompt
system_message_temp = SystemMessagePromptTemplate.from_template(filled_system_prompt)

# Optional: Show prompt used
with st.sidebar.expander("🧠 Prompt Being Used"):
    st.code(filled_system_prompt)

# LLM setup
llm = ChatOpenAI(model="o3-mini", temperature=1.0)

# Prompt Template with message history placeholder
chat_prompt = ChatPromptTemplate.from_messages([
    system_message_temp,
    MessagesPlaceholder(variable_name="history"),
    ("user", "{user_input}")
])

# Combine prompt + LLM
runnable = chat_prompt | llm

# Chat history dictionary by session
chat_histories = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in chat_histories:
        chat_histories[session_id] = InMemoryChatMessageHistory()
    return chat_histories[session_id]

chain_with_history = RunnableWithMessageHistory(
    runnable=runnable,
    get_session_history=get_session_history,
    input_messages_key="user_input",
    history_messages_key="history"
)

def reflect_on_response(user_input: str, ai_response: str, readings: str, custom_prompt_block: str) -> str:
    filled = (
        reflection_prompt
        .replace("{CHATBOT_RESPONSE}", ai_response)
        .replace("{fyu_chan_system_prompt}", fyu_chan_system_prompt)
        .replace("{readings}", readings)
        .replace("{current_date}", current_date)
        .replace("{USER_INPUT}", user_input)
        .replace("{custom_prompt_block}", custom_prompt_block if custom_prompt_block else "")
    )

    reflection_messages = [
        SystemMessage(content="You are an agent that critically reflects on the AI's prior response."),
        HumanMessage(content=f"User said: {user_input}"),
        AIMessage(content=f"AI responded: {ai_response}"),
        HumanMessage(content=filled)
    ]

    reflection_result = llm.invoke(reflection_messages)
    return reflection_result.content


def chat_with_fyu(user_input: str, session_id: str, enable_reflection: bool = True) -> str:
    global custom_prompt_block

    response = chain_with_history.invoke(
        {"user_input": user_input, "readings": readings, "current_date": current_date},
        config={"configurable": {"session_id": session_id}}
    )
    bot_reply = response.content
    final_response = bot_reply

    if enable_reflection:
        reflection = reflect_on_response(user_input, bot_reply, readings, custom_prompt_block)
        try:
            reflection_data = json.loads(reflection)
            final_response = reflection_data.get("final_response", bot_reply)
        except json.JSONDecodeError:
            final_response += f"\n\n🧠 Reflection (unparsed):\n{reflection}"

    return final_response


# ---- Streamlit App UI ----
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("Fyu-chan Chatbot")
st.markdown("Talk to Fyu-chan! Ask me anything.")

user_input = st.chat_input("Type your message...")

# Display past messages
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Process new input
if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    bot_response = chat_with_fyu(user_input, st.session_state.session_id)

    st.session_state.chat_history.append({"role": "assistant", "content": bot_response})
    with st.chat_message("assistant"):
        st.markdown(bot_response)