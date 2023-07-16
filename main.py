# Required Libraries
import streamlit as st
import requests
import pdfplumber
# from io import BytesIO
import plotly.graph_objects as go
from typing import Optional
import json

from langchain.chains.openai_functions import (
    create_openai_fn_chain,
    create_structured_output_chain,
)
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from typing import Sequence
from langchain.chat_models import ChatAnthropic

load_dotenv()

class ScoreForEachCriteria(BaseModel):
    """Score and Analysis of each Criteria"""
    name: str = Field(..., description="The criteria of analysis")
    score: int = Field(..., description="The score")
    analysis: str = Field(..., description="The analysis")

class Score(BaseModel):
    "Score of the whitepaper"
    score: Sequence[ScoreForEachCriteria] = Field(..., description="The analysis criteria for scoring")


running_score = {
  "scores": [
    {
      "name": "Offeror/Issuer Information",
      "score": 0,
      "analysis": ""
    },
    {
      "name": "Trading Platform Operator Information",
      "score": 0,
      "analysis": ""
    },
    {
      "name": "Crypto-Asset Project Information",
      "score": 0,
      "analysis": ""
    },
    {
      "name": "Offer to the Public",
      "score": 0,
      "analysis": ""
    },
    {
      "name": "Crypto-Asset Details",
      "score": 0,
      "analysis": ""
    },
    {
      "name": "Non-Approval Statement",
      "score": 0,
      "analysis": ""
    },
    {
      "name": "Future Value Statement",
      "score": 0,
      "analysis": ""
    },
    {
      "name": "Management Body Statement",
      "score": 0,
      "analysis": ""
    },
    {
      "name": "Summary",
      "score": 0,
      "analysis": ""
    },
    {
      "name": "Risk Factors",
      "score": 0,
      "analysis": ""
    },
    {
      "name": "Climate Impact",
      "score": 0,
      "analysis": ""
    },
    {
      "name": "Language and Format",
      "score": 0,
      "analysis": ""
    }
  ]
}

if "score" not in st.session_state:
        st.session_state["score"] = running_score

# def download_pdf(url):
#    # Download the PDF
#     response = requests.get(url, allow_redirects=True)
#     pdf_file = BytesIO(response.content)

#     # Parse the PDF with pdfplumber
#     with pdfplumber.open(pdf_file) as pdf:
#         text = ''
#         for page in pdf.pages:
#             text += page.extract_text()

llm = ChatOpenAI(model="gpt-3.5-turbo-16k-0613", temperature=0,  max_tokens=4096, verbose=False)
# llm = ChatOpenAI(model="gpt-4-0613", temperature=0,  max_tokens=4096, verbose=True)
# chain = ChatAnthropic(model='claude-1.3-100k', temperature=0, max_tokens_to_sample=100000)

def summon_llm(text_chunk, llm, prompt, chain):
    scores = chain.run({'input': text_chunk, 'running_score' : st.session_state.score})
    st.session_state.score = scores.dict()
    return scores.dict()

# parsing and analysis
def analyze(text):
    # Placeholder function for text analysis
    # You would replace this with your own text analysis logic



    criteria = """
    Given the text of whitepaper, perform thorough analysis based on following criteria:

    1. **Offeror/Issuer Information**: Describe the information about the offeror or the person seeking admission to trading, and the issuer (if different). Is this information clear, accurate, and comprehensive? Score from 1 (non-compliant) to 5 (fully compliant).
    
    2. **Trading Platform Operator Information**: If applicable, describe the information about the operator of the trading platform. Is this information clear, accurate, and comprehensive? Score from 1 (non-compliant) to 5 (fully compliant).
    
    3. **Crypto-Asset Project Information**: Describe the information about the crypto-asset project. Consider its objectives, timeline, technical details, use of funds, and risk factors. Is this information clear, accurate, and comprehensive? Score from 1 (non-compliant) to 5 (fully compliant).
    
    4. **Offer to the Public**: Describe the terms of the crypto-asset offer to the public. Are these terms clear, accurate, and comprehensive? Score from 1 (non-compliant) to 5 (fully compliant).
    
    5. **Crypto-Asset Details**: Are the properties and functionalities of the crypto-asset clearly explained? This includes information about the rights and obligations attached to the crypto-asset and the underlying technology. Score from 1 (non-compliant) to 5 (fully compliant).
    
    6. **Non-Approval Statement**: Is there a clear statement indicating that the whitepaper has not been approved by any competent authority? Score from 1 (non-compliant) to 5 (fully compliant).
    
    7. **Future Value Statement**: Does the whitepaper avoid making any guarantees or promises about the future value of the crypto-asset? Score from 1 (non-compliant) to 5 (fully compliant).
    
    8. **Management Body Statement**: Is there a statement from the management body asserting that the whitepaper is in compliance with the guidelines and that all information is fair, clear, and not misleading? Score from 1 (non-compliant) to 5 (fully compliant).
    
    9. **Summary**: Is there a concise summary at the beginning of the whitepaper that provides key information about the crypto-asset and the public offer or intended trading? Score from 1 (non-compliant) to 5 (fully compliant).
    
    10. **Risk Factors**: Does the whitepaper provide clear information on the potential risks and limitations of the crypto-asset? Score from 1 (non-compliant) to 5 (fully compliant).
    
    11. **Climate Impact**: Is there information on the principal adverse impacts on the climate and other environment-related adverse impacts of the consensus mechanism used to issue the crypto-asset? Score from 1 (non-compliant) to 5 (fully compliant).
    
    12. **Language and Format**: Is the whitepaper written in an official language of the home Member State or in a language customary in the sphere of international finance, and is it available in a machine-readable format? Score from 1 (non-compliant) to 5 (fully compliant).

    After reviewing each of these criteria elements, update the Score of the whitepaper.
    """
    prompt_msgs = [
        SystemMessage(
            content="You are a world class algorithm for performing critical analysis of whitepaper."
        ),
        HumanMessage(content=f"{criteria}"),
        HumanMessage(
            content="Here is the text of whitepaper followed by running score."
        ),
        HumanMessagePromptTemplate.from_template("{input}\n{running_score}"),
    ]
    prompt = ChatPromptTemplate(messages=prompt_msgs)

    chain = create_openai_fn_chain([Score], llm, prompt, verbose=True)

    text_chunks = [text[i:i + 13000] for i in range(0, len(text), 13000)]
    for text_chunk in text_chunks:
        score = summon_llm(text_chunk, llm, prompt, chain)
    return score['score']

    # scores = chain.run({'input': text, 'running_score' : st.session_state.score})
    # st.session_state.score = scores.dict()

    # print(json.dumps(scores.dict()))
    # print(st.session_state.score)
    # return scores.dict()['score']

# Streamlit App
st.title('🔍 Whitepaper Lens')

# User input
# pdf_url = st.text_input('Enter the URL of a PDF to analyze')

text = None
uploaded_file = st.file_uploader("Upload a whitepaper", type=['pdf', 'txt', 'docx'])

if uploaded_file is not None:
    if uploaded_file.type == "application/pdf":
        with pdfplumber.open(uploaded_file) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text()
            # st.write(pages)

    elif uploaded_file.type == "text/plain":
        # Decode the uploaded file and convert to string
        text = uploaded_file.getvalue().decode()
        # st.write(text)

    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        # Convert the uploaded docx file to text
        text = docx2txt.process(uploaded_file)
        # st.write(text)

if text:
    # Analyze the PDF
    st.session_state["score"] = running_score
    # st.write('Analyzing Whitepaper...')
    with st.spinner('🧠 Analyzing Whitepaper... '):
        scores = analyze(text)

        # Display the scores
        # st.write('Analysis Results:')
        # st.json(scores)
        with st.expander('Whitepaper Analysis Report'):
            for score in scores:
                st.markdown(f'### {score["name"]}')
                st.markdown(f'**Score:** {score["score"]}')
                st.markdown(f'**Analysis:** {score["analysis"]}')

        # Display a visualization
        # st.write('Visualization:')
        # fig, ax = plt.subplots()
        # ax.barh(list(scores.keys()), list(scores.values()), color='skyblue')
        # ax.set_xlabel('Score')
        # ax.set_title('Compliance Scores for Each Section')
        # st.pyplot(fig)

        # Display a visualization
        st.write('Visualization:')
        # labels = list(scores.keys())
        # values = list(scores.values()) + list(scores.values())[:1]
        labels = [item['name'] for item in scores]
        values = [item['score'] for item in scores] + [scores[0]['score']]


        fig = go.Figure(data=go.Scatterpolar(
        r=values,
        theta=labels,
        fill='toself',
        fillcolor='white'
        ))

        fig.update_layout(
        polar=dict(
            bgcolor='lightgray',
            radialaxis=dict(
            visible=True,
            range=[0, 5],
            color='gray'
            )),
        showlegend=False
        )

        color_palette = ['#e74c3c', '#f39c12', '#f1c40f', '#2ecc71', '#27ae60']

        average_score = round(sum(values) / len(values), 2)

        fig.add_annotation(
            xref="paper", yref="paper",
            x=0.5, y=0.65, 
            text=f'Score',
            showarrow=False, 
            font=dict(size=22, color='grey')
        )
        fig.add_annotation(
            xref="paper", yref="paper",
            x=0.5, y=0.5, 
            text=f'{average_score}',
            showarrow=False, 
            # font=dict(size=70, color=color_palette[round(average_score)])
            font=dict(size=70, color='#27ae60')
        )

        st.plotly_chart(fig)


        # Create a colormap
        color_scale = [[0, 'red'], [1., 'green']]

        # Map the scores to colors in the palette
        colors = [color_palette[val-1] for val in values]

        fig = go.Figure(go.Bar(
            x=values,
            y=labels,
            orientation='h',
            marker_color=colors, # set color to an array/list of desired values
            marker=dict(
                color=colors, # set color to an array/list of desired values
                colorscale=color_scale, # choose a colorscale
                cmin=0,
                cmax=5,
                colorbar=dict(
                    title="Range",
                    tickvals=[0,1, 2, 3, 4, 5],
                    ticktext=["0","1", "2", "3", "4", "5"],
                    len=0.75
                )
            )
        ))

        fig.update_layout(title_text='', xaxis_title='Score')

        st.plotly_chart(fig)

