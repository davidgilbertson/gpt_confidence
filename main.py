from html import escape
import math

import streamlit as st
from openai import OpenAI, Stream
from openai.types.chat import ChatCompletionChunk

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


def stream_to_html(stream: Stream[ChatCompletionChunk]):
    html = ""
    raw_text = ""

    with st.empty():
        for chunk in stream:
            text = chunk.choices[0].delta.content

            if not text:
                continue

            raw_text += text
            content = chunk.choices[0].logprobs.content[0]
            if "\n" in content.token:
                html += content.token.replace("\n", "<br>")
            else:
                prob = math.exp(content.logprob)
                underline = f"3px solid rgba(255, 0, 0, {1-prob ** 1.6:.0%})"
                options = [
                    f"{x.token} ({math.exp(x.logprob):.2%})"
                    for x in content.top_logprobs
                ]
                tooltip = escape("\n".join(options))
                text = escape(text)
                html += f"<span title='\u200B{tooltip}' style='border-bottom: {underline}'>{text}</span>"

            st.html(html)

    return html, raw_text


avatars = dict(assistant="open_ai_logo.svg", user=""":material/person:""")

for message in st.session_state.messages:
    role = message["role"]
    with st.chat_message(role, avatar=avatars.get(role)):
        if role == "user":
            st.markdown(message["content"])
        else:
            st.html(message["html"])

last_message = st.empty()

if prompt := st.chat_input("E.g. Pick a number between 1 and 6", max_chars=500):
    st.session_state.messages.append(dict(role="user", content=prompt))

    with st.chat_message("user", avatar=avatars.get("user")):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar=avatars.get("assistant")):
        # Strip out the HTML before sending to API, and just the last few messages
        messages: list[dict[str, str]] = [
            dict(role=msg["role"], content=msg["content"])
            for msg in st.session_state.messages[-8:]
        ]

        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            logprobs=True,
            top_logprobs=10,
            max_tokens=1000,
            stream=True,
        )
        response_html, response_text = stream_to_html(stream)

        st.session_state.messages.append(
            dict(role="assistant", content=response_text, html=response_html)
        )

if not st.session_state.messages:
    st.write("Type something in the input down the bottom 👇")
