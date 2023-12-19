from langchain.schema import (
    SystemMessage,
)
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)

SYSTEM_PROMPT = """Answer the following questions as best you can. You have access to the following tools:

{tools}

The way you use the tools is by specifying a json blob.
Specifically, this json should have a `action` key (with the name of the tool to use) and a `action_input` key (with the input to the tool going here).

The only values that should be in the "action" field are: {tool_names}

The $JSON_BLOB should only contain a SINGLE action and MUST be formatted as markdown, do NOT return a list of multiple actions. Here is an example of a valid $JSON_BLOB:

```
{{
  "action": $TOOL_NAME,
  "action_input": $INPUT
}}
```

ALWAYS use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action:
```
$JSON_BLOB
```
Observation: the result of the action
... (this Thought/Action/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin! Reminder to always use the exact characters `Final Answer` when responding. If using a tool you the $JSON_BLOB must be formatted as markdown and only use a SINGLE action at a time."""

HUMAN_PROMPT = """{input}

{agent_scratchpad}"""


EVALUATION_PROMPT = """###Task Description:
An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets a score of 5, and a score rubric representing a evaluation criteria are given.
1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
3. The output format should look as follows: \"Feedback: {{write a feedback for criteria}} [RESULT] {{an integer number between 1 and 5}}\"
4. Please do not generate any other opening, closing, and explanations. Be sure to include [RESULT] in your output.

###The instruction to evaluate:
{instruction}

###Response to evaluate:
{response}

###Reference Answer (Score 5):
{reference_answer}

###Score Rubrics:
[{criteria_description}]
Score 1: {score1_description}
Score 2: {score2_description}
Score 3: {score3_description}
Score 4: {score4_description}
Score 5: {score5_description}

###Feedback:"""

CORRECTNESS_CRITERIA = {
    "criteria_description": "Is the response correct, accurate, and factual based on the reference answer?",
    "score1_description": "The response is completely incorrect, inaccurate, and/or not factual.",
    "score2_description": "The response is mostly incorrect, inaccurate, and/or not factual.",
    "score3_description": "The response is somewhat correct, accurate, and/or factual.",
    "score4_description": "The response is mostly correct, accurate, and factual.",
    "score5_description": "The response is completely correct, accurate, and factual.",
}


def build_eval_prompt():
    prometheus_prompt_template = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content="You are a fair evaluator language model."),
            HumanMessagePromptTemplate.from_template(EVALUATION_PROMPT),
        ]
    )

    return prometheus_prompt_template.partial(
        criteria_description=CORRECTNESS_CRITERIA["criteria_description"],
        score1_description=CORRECTNESS_CRITERIA["score1_description"],
        score2_description=CORRECTNESS_CRITERIA["score2_description"],
        score3_description=CORRECTNESS_CRITERIA["score3_description"],
        score4_description=CORRECTNESS_CRITERIA["score4_description"],
        score5_description=CORRECTNESS_CRITERIA["score5_description"],
    )
