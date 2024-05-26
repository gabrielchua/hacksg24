"""
app.py
"""
import base64


import streamlit as st
from openai import OpenAI
from openai.types.beta.assistant_stream_event import (
    ThreadRunStepCreated,
    ThreadRunStepDelta,
    ThreadRunStepCompleted,
    ThreadMessageCreated,
    ThreadMessageDelta
    )
from openai.types.beta.threads.text_delta_block import TextDeltaBlock 
from openai.types.beta.threads.runs.tool_calls_step_details import ToolCallsStepDetails
from openai.types.beta.threads.runs.code_interpreter_tool_call import (
    CodeInterpreterOutputImage,
    CodeInterpreterOutputLogs
    )
from utils import (
    initialise_session_state,
    render_custom_css,
    moderation_endpoint
)


# Set page config
st.set_page_config(page_title="ASSURE",
                   layout='centered')

# Get secrets
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
ASSISTANT_ID = st.secrets["OPENAI_ASSISTANT_ID"]

# Initialise the OpenAI client, and retrieve the assistant
client = OpenAI(api_key=OPENAI_API_KEY)
assistant = client.beta.assistants.retrieve(ASSISTANT_ID)

# Apply custom CSS
render_custom_css()

# Initialise session state
initialise_session_state()

# UI
st.subheader("ASSURE")

# Create a new thread
if "thread_id" not in st.session_state:
    thread = client.beta.threads.create()
    st.session_state.thread_id = thread.id
    print(st.session_state.thread_id)

# Update the thread to attach the file
client.beta.threads.update(
        thread_id=st.session_state.thread_id,
        )

# Local history
if "messages" not in st.session_state:
    st.session_state.messages = []

# UI
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        for item in message["items"]:
            item_type = item["type"]
            if item_type == "text":
                st.markdown(item["content"])
            elif item_type == "image":
                for image in item["content"]:
                    st.html(image)
            elif item_type == "code_input":
                with st.status("Code", state="complete"):
                    st.code(item["content"])
            elif item_type == "code_output":
                with st.status("Results", state="complete"):
                    st.code(item["content"])

if prompt := st.chat_input("Ask me a question about your dataset"):
    if moderation_endpoint(prompt):
        st.toast("Your message was flagged. Please try again.", icon="⚠️")
        st.stop

    st.session_state.messages.append({"role": "user",
                                      "items": [
                                          {"type": "text",
                                           "content": prompt
                                           }]})
    
    client.beta.threads.messages.create(
        thread_id=st.session_state.thread_id,
        role="user",
        content=prompt
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        stream = client.beta.threads.runs.create(
            thread_id=st.session_state.thread_id,
            assistant_id=ASSISTANT_ID,
            stream=True
        )

        assistant_output = []

        for event in stream:
            if isinstance(event, ThreadRunStepCreated):
                if event.data.step_details.type == "tool_calls":
                    assistant_output.append({"type": "code_input",
                                            "content": ""})

                    code_input_expander= st.status("Writing code ⏳ ...", expanded=True)
                    code_input_block = code_input_expander.empty()

            if isinstance(event, ThreadRunStepDelta):
                if event.data.delta.step_details.tool_calls[0].code_interpreter is not None:
                    code_interpretor = event.data.delta.step_details.tool_calls[0].code_interpreter
                    code_input_delta = code_interpretor.input
                    if (code_input_delta is not None) and (code_input_delta != ""):
                        assistant_output[-1]["content"] += code_input_delta
                        code_input_block.empty()
                        code_input_block.code(assistant_output[-1]["content"])

            elif isinstance(event, ThreadRunStepCompleted):
                if isinstance(event.data.step_details, ToolCallsStepDetails):
                    code_interpretor = event.data.step_details.tool_calls[0].code_interpreter
                    if code_interpretor.outputs is not None:
                        code_interpretor_outputs = code_interpretor.outputs[0]
                        code_input_expander.update(label="Code", state="complete", expanded=False)
                        # Image
                        if isinstance(code_interpretor_outputs, CodeInterpreterOutputImage):
                            image_html_list = []
                            for output in code_interpretor.outputs:
                                image_file_id = output.image.file_id
                                image_data = client.files.content(image_file_id)
                                
                                # Save file
                                image_data_bytes = image_data.read()
                                with open(f"images/{image_file_id}.png", "wb") as file:
                                    file.write(image_data_bytes)

                                # Open file and encode as data
                                file_ = open(f"images/{image_file_id}.png", "rb")
                                contents = file_.read()
                                data_url = base64.b64encode(contents).decode("utf-8")
                                file_.close()

                                # Display image
                                image_html = f'<p align="center"><img src="data:image/png;base64,{data_url}" width=600></p>'
                                st.html(image_html)

                                image_html_list.append(image_html)

                            assistant_output.append({"type": "image",
                                                    "content": image_html_list})
                        # Console log
                        elif isinstance(code_interpretor_outputs, CodeInterpreterOutputLogs):
                            assistant_output.append({"type": "code_output",
                                                        "content": ""})
                            code_output = code_interpretor.outputs[0].logs
                            with st.status("Results", state="complete"):
                                st.code(code_output)    
                                assistant_output[-1]["content"] = code_output   

            elif isinstance(event, ThreadMessageCreated):
                assistant_output.append({"type": "text",
                                        "content": ""})
                assistant_text_box = st.empty()

            elif isinstance(event, ThreadMessageDelta):
                if isinstance(event.data.delta.content[0], TextDeltaBlock):
                    assistant_text_box.empty()
                    assistant_output[-1]["content"] += event.data.delta.content[0].text.value
                    assistant_text_box.markdown(assistant_output[-1]["content"])
            
        st.session_state.messages.append({"role": "assistant",
                                        "items": assistant_output})

with st.sidebar:
    if st.button("Generate Report"):

        # Extract text content from session messages
        text_contents = [f"{msg['role']}: {item['content']}" for msg in st.session_state.messages for item in msg['items'] if item['type'] == 'text']

        # Join all text contents into a single string
        full_text = " ".join(text_contents)

        # Call the LLM API to summarize the conversation
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system",
                "content": "You are a world class financial advisor. Your goal is to summarize a conversation between a financial agent and human. Your task is to create a clear and concise report to help the user with their financial planning."},
                {"role": "user",
                "content": f"Here is the conversation: {full_text}"}
                ],
            temperature=0.2
        )

        summary = response.choices[0].message.content

        # Generate HTML content for PDF
        html_content = f"""
        <html>
        <head>
            <title>Conversation Summary</title>
        </head>
        <body>
            <h1>Summary of Conversation</h1>
            <p>{summary}</p>
        </body>
        </html>
        """

        # Convert HTML to PDF using ReportLab
        from io import BytesIO
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import SimpleDocTemplate, Paragraph

        pdf_path = "conversation_summary.pdf"
        pdf_buffer = BytesIO()
        pdf = SimpleDocTemplate(pdf_buffer, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []

        # Title
        title_style = styles['Title']
        title = Paragraph("Summary of Conversation", title_style)
        story.append(title)

        # Summary text with markdown support
        summary_style = styles['BodyText']
        summary_paragraph = Paragraph(summary, summary_style)
        story.append(summary_paragraph)

        pdf.build(story)

        pdf_buffer.seek(0)

        # Create a download link using Streamlit
        btn = st.download_button(
            label="Download Summary Report",
            data=pdf_buffer,
            file_name="conversation_summary.pdf",
            mime="application/pdf"
        )
