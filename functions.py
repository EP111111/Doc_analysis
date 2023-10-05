import textract
import numpy as np
import pandas as pd
import re


def empty_Vertex(C,model):
  parameters = {
      "max_output_tokens": 1024,
      "temperature": 0,
      "top_p": 0.8,
      "top_k": 40
  }
  response = model.predict(
      C,
      **parameters
  )
  return response.text

def Query_Vertex(Question,Content,model):

  parameters = {
      "max_output_tokens": 1024,
      "temperature": 0.1,
      "top_p": 0.8,
      "top_k": 40
  }

  response = model.predict(
      f"""{Content}


  input: {Question}
  output:
  """,
      **parameters
  )
  return response.text

def answer_exist(content,query,model):
  parameters = {
      "max_output_tokens": 256,
      "temperature": 0.2,
      "top_p": 0.8,
      "top_k": 40
  }
  response = model.predict(
      f"""{content}
  input: Does the upper document answer \'Who is the richest man\',answer no or yes
  output: no

  input: Does the upper document answer \'What does palm2 do\',answer no or yes
  output: no

  input: Does the upper document answer \'If I eat myself, will I get twice as big or disappear completely?\',answer no or yes
  output: no

  input: Does the upper document answer \'what is the document about\',answer no or yes
  output: yes

    input: Does the upper document answer \'Who win the 2200 world cup\',answer no or yes
  output: no

  input: Does the upper document answer \'what's the purpose of the document\',answer no or yes
  output: yes

  input: Does the upper document answer \'How does peter go to school\',answer no or yes
  output: no

  input: Does the upper document answer \'how many paragraph in the document\',answer no or yes
  output: yes




  input: Does the upper document answer \{query},answer no or yes
  output: 'yes' or 'no'
  """,
      **parameters
  )
  return response.text

def what_chlg(Content,model):
  #find all the questions
  Query='extract out all the questions the author mentioned in the following article'
  response=Query_Vertex(Query,Content,model)
  #remove text before ':'
  response=re.sub('.*:','',response)
  response=response.split('\n')
  for i,c in enumerate(response):
    c=re.sub('^[0-9]\. *','',c)
    c=re.sub('^\* ','',c)
    c=re.sub('^\- ','',c)

    response[i]=c
  try:
    response.remove('')
  except:
    pass
  return response


def why_ask(question,content,model):
  #why the author asked this qustion
  Query=f'Use the sentence in the document only to answer why the author asked {question}'
  response=Query_Vertex(Query,content,model)
  return response

def satify_Q(question,content,model):
  #How to satisfy the requirement
  Query=f'Use the sentence in the document only to answer. To get the precise answer of {question}, what I need to know. '
  response=Query_Vertex(Query,content,model)
  return response



def Q_W_N(Document,model):
  '''{1:{question:Q1,Why:W1, Need:N}}'''
  Json_question={}
  questions=what_chlg(Document,model)
  for num, Q in enumerate(questions):
    if Q!='':
      why=why_ask(Q,Document).split('\n')
      Need_know=satify_Q(Q,Document).split('\n')
      Json_question[num]={'question':Q,'Why':why, 'Need':Need_know}
  return Json_question

def list_req(Content,model):
  #find all the questions
  Query='extract out all the requirments the author mentioned in the article'
  response=Query_Vertex(Query,Content,model)
  #remove text before ':'
  response=re.sub('.*:','',response)
  response=response.split('\n')
  for i,c in enumerate(response):

    c=re.sub('^[0-9]\. *','',c)
    c=re.sub('^\* ','',c)
    c=re.sub('^\- ','',c)
    response[i]=c
  try:
    response.remove('')
  except:
    pass
  return response

def list_context(question,content,model):
  #why the author has this requirement
  Query=f'Use the sentence in the document only to answer why the document has the requirment of {question}'
  response=Query_Vertex(Query,content,model)
  return response

def list_satisfy(question,content,model):
  #How to satisfy the requirement
  Query=f'To satisfy the requirment of {question}, what I need to know. Use the sentence in the document only to answer'
  response=Query_Vertex(Query,content,model)
  return response


def to_q(question,content,model):
  #why the author asked this qustion
  Query=f'Change the REQURIEMENT to question format: {question}.'
  response=Query_Vertex(Query,content,model)
  return response




########

def LLM_split(doc,model):
    a=Query_Vertex('split the document into 2 sections. The first section is more than 4000 words. output is the first sentence in the second section' ,doc,model)
    return a

def LLM_SPLIT_FUNC(Doc, model, split_size=10000):
  remain_doc=Doc[:]
  doc=remain_doc[:split_size]
  ind=0
  LLM_c={}
  while len(doc)>split_size-1:
    sentence=LLM_split(doc,model)
    sentence=sentence.split('\n')[0]
    sections=re.split(sentence,remain_doc)
    remain_doc=sentence+sections[1]
    LLM_c[ind]=sections[0]
    doc=remain_doc[:split_size]
    ind+=1
  else:
    LLM_c[ind]=remain_doc
  return LLM_c


    