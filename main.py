from html import escape
import math

import streamlit as st
from openai import OpenAI, Stream

st.set_page_config(page_title="GPT Confidence")

st.sidebar.markdown(
    """
This interface is an experiment in showing probabilities ('confidence') in the model's response.

It uses the <a href="https://cookbook.openai.com/examples/using_logprobs" target="blank">logprobs</a> feature of the OpenAI API and underlines each token in the response. Brighter red means less certainty.

Hover over a token to see the exact confidence and the top 10 candidates.

Note that there's no support for LaTeX or Markdown, since this shows the actual tokens returned by the model.

It uses the `gpt-4o-mini` model.
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def get_client():
    return OpenAI(api_key=st.secrets.OPENAI_API_KEY)


client = get_client()

if "messages" not in st.session_state:
    st.session_state.messages = []


def stream_to_html(stream: Stream):
    html = ""
    with st.empty():
        for chunk in stream:
            choice = chunk.choices[0]
            text = chunk.choices[0].delta.content

            if not text:
                continue

            text = escape(text)
            content = choice.logprobs.content[0]
            if "\n" in content.token:
                # Tokens can be :\n\n, etc, ideally these would be split into returns and content
                html += content.token.replace("\n", "<br>")
            else:
                prob = math.exp(content.logprob)
                underline = f"3px solid rgba(255, 0, 0, {1-prob ** 1.6:.0%})"
                options = [
                    f"{escape(x.token)} ({math.exp(x.logprob):.2%})"
                    for x in content.top_logprobs
                ]
                tooltip = "\n".join(options)
                html += f"<span title='{tooltip}' style='border-bottom: {underline}'>{text}</span>"

            st.html(html)

    return html


avatars = dict(assistant="open_ai_logo.svg", user=""":material/person:""")

for message in st.session_state.messages:
    role = message["role"]
    text = message["content"]
    with st.chat_message(role, avatar=avatars.get(role)):
        if role == "user":
            st.markdown(text, unsafe_allow_html=True)
        else:
            st.html(text)

last_message = st.empty()

if prompt := st.chat_input("E.g. Pick a number between 1 and 6"):
    st.session_state.messages.append(dict(role="user", content=prompt))

    with st.chat_message("user", avatar=avatars.get("user")):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar=avatars.get("assistant")):
        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=st.session_state.messages,
            logprobs=True,
            top_logprobs=10,
            stream=True,
        )
        text = stream_to_html(stream)

        st.session_state.messages.append(dict(role="assistant", content=text))

if not st.session_state.messages:
    st.write("Type something in the input down the bottom 👇")
