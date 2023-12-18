import os
import json
import asyncio
from datetime import datetime
from typing import Any, Dict, List

import pandas as pd
from datasets import Dataset
from chat_wrapper import HuggingFaceChatWrapper
from langchain.agents import AgentExecutor, load_tools
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import (
    ReActJsonSingleInputOutputParser,
)
from langchain.llms import (
    HuggingFaceEndpoint,
)
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.schema import (
    SystemMessage,
)
from langchain.tools.render import render_text_description
from prompts import HUMAN_PROMPT, EVALUATION_PROMPT, SYSTEM_PROMPT, CORRECTNESS_CRITERIA
from tqdm import tqdm


def build_agent(hf_endpoint_url: str):
    """
    Build a zero-shot ReAct chat agent from HF endpoint.

    Args:
        hf_endpoint_url (str): The endpoint URL for the Hugging Face model.

    Returns:
        AgentExecutor: An agent executor object that can be used to run the agent.

    """
    # instantiate LLM and chat model
    llm = HuggingFaceEndpoint(
        endpoint_url=hf_endpoint_url,
        task="text-generation",
        model_kwargs={
            "max_new_tokens": 512,
            "do_sample": False,
        },
    )

    chat_model = HuggingFaceChatWrapper(llm=llm)

    # setup tools
    tools = load_tools(["serpapi", "llm-math"], llm=llm)

    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT),
            HumanMessagePromptTemplate.from_template(HUMAN_PROMPT),
        ]
    )
    prompt = prompt.partial(
        tools=render_text_description(tools),
        tool_names=", ".join([t.name for t in tools]),
    )

    # define the agent
    chat_model_with_stop = chat_model.bind(stop=["\nObservation"])
    agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_log_to_str(x["intermediate_steps"]),
        }
        | prompt
        | chat_model_with_stop
        | ReActJsonSingleInputOutputParser()
    )

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        return_intermediate_steps=True,
        handle_parsing_errors=True,
        max_iterations=5,
    )


def build_evaluator(hf_endpoint_url: str) -> tuple:
    """
    Build an evaluator language model using the given Hugging Face endpoint URL.

    Args:
        hf_endpoint_url (str): The URL of the Hugging Face endpoint.

    Returns:
        Tuple: A tuple containing the evaluator chat model and the correctness prompt template.
    """
    eval_llm = HuggingFaceEndpoint(
        endpoint_url=hf_endpoint_url,
        task="text-generation",
        model_kwargs={
            "max_new_tokens": 512,
            "do_sample": False,
        },
    )

    eval_chat_model = HuggingFaceChatWrapper(llm=eval_llm)

    prometheus_prompt_template = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content="You are a fair evaluator language model."),
            HumanMessagePromptTemplate.from_template(EVALUATION_PROMPT),
        ]
    )

    correctness_prompt_template = prometheus_prompt_template.partial(
        criteria_description=CORRECTNESS_CRITERIA["criteria_description"],
        score1_description=CORRECTNESS_CRITERIA["score1_description"],
        score2_description=CORRECTNESS_CRITERIA["score2_description"],
        score3_description=CORRECTNESS_CRITERIA["score3_description"],
        score4_description=CORRECTNESS_CRITERIA["score4_description"],
        score5_description=CORRECTNESS_CRITERIA["score5_description"],
    )

    return eval_chat_model, correctness_prompt_template


async def run_agent_eval(
    question: str,
    ground_truth_answer: str,
    agent_name: str,
    agent_executor: AgentExecutor,
    evaluator: HuggingFaceChatWrapper,
    eval_prompt_template: ChatPromptTemplate,
) -> dict:
    """
    Runs the agent and evaluation process for a given question and ground truth answer.

    Args:
        question (str): The input question to be evaluated.
        ground_truth_answer (str): The ground truth answer for the question.
        agent_executor (AgentExecutor): The agent executor object used to run the agent.
        evaluator (HuggingFaceChatWrapper): The evaluator object used to evaluate the agent's response.
        eval_prompt_template (ChatPromptTemplate): The template for the evaluation prompt.

    Returns:
        dict: A dictionary containing the evaluation results, including the agent model ID, evaluator model ID,
        question, ground truth answer, prediction, intermediate steps, evaluation score, evaluation feedback,
        tool call parsing error flag, iteration limit exceeded flag, and agent error (if any).
    """
    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        # run agent
        out = await agent_executor.ainvoke({"input": question})

        # run evaluator
        eval_prompt = eval_prompt_template.format_messages(
            instruction=question,
            response=out["output"],
            reference_answer=ground_truth_answer,
        )
        eval_result = await evaluator.ainvoke(eval_prompt)
        feedback, score = [
            item.strip() for item in eval_result.content.split("[RESULT]")
        ]

        # check for parsing errors which indicate the LLM failed to follow the ReACT format
        # this could be due to an issue with the tool calling format or ReACT formatting (i.e. Thought, Action, Observation, etc.)
        parsing_error = (
            True
            if any(
                [
                    "Could not parse LLM output" in step[0].log
                    for step in out["intermediate_steps"]
                ]
            )
            else False
        )

        # check if iteration limit exceeded
        iteration_limit_exceeded = (
            True
            if "Agent stopped due to iteration limit or time limit." in out["output"]
            else False
        )
        raised_exception = False

    except Exception as e:
        out = {"output": None, "intermediate_steps": None}
        score = None
        feedback = None
        parsing_error = False
        iteration_limit_exceeded = False
        exception = e
        raised_exception = True

    end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if out["intermediate_steps"] is not None:
        intermediate_steps = [
            {
                "tool": response[0].tool,
                "tool_input": response[0].tool_input,
                "tool_output": response[1],
            }
            for response in out["intermediate_steps"]
        ]
    else:
        intermediate_steps = None

    tools_used = (
        [step["tool"] for step in intermediate_steps]
        if intermediate_steps is not None
        else None
    )
    # collect results
    return {
        "agent_name": agent_name,
        "agent_model_id": agent_executor.dict()["agent"]["runnable"]["middle"][-1][
            "bound"
        ]["_type"],
        "evaluator_model_id": evaluator.model_id,
        "question": question,
        "gt_answer": ground_truth_answer,
        "prediction": out["output"],
        "intermediate_steps": intermediate_steps,
        "eval_score": score,
        "eval_feedback": feedback,
        "parsing_error": parsing_error,
        "iteration_limit_exceeded": iteration_limit_exceeded,
        "agent_error": repr(exception) if raised_exception else None,
        "start_time": start_time,
        "end_time": end_time,
        "tools_used": tools_used,
        "number_distinct_tools_used": len(list(set(tools_used)))
        if tools_used is not None
        else None,
        "number_of_steps": len(intermediate_steps)
        if intermediate_steps is not None
        else None,
    }


async def evaluate_on_dataset(
    dataset: Dataset,
    agent_name: str,
    agent_executor: AgentExecutor,
    evaluator: HuggingFaceChatWrapper,
    eval_prompt_template: ChatPromptTemplate,
) -> List[Dict[str, Any]]:
    """
    Evaluates the agent on a given dataset.

    Args:
        dataset (Dataset): The dataset to evaluate the agent on.
        agent_executor (AgentExecutor): The agent executor object used to run the agent.
        evaluator (HuggingFaceChatWrapper): The evaluator object used to evaluate the agent's response.
        eval_prompt_template (ChatPromptTemplate): The template for the evaluation prompt.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing the evaluation results for each example in the dataset.
        Each dictionary includes the agent model ID, evaluator model ID, question, ground truth answer, prediction,
        intermediate steps, evaluation score, evaluation feedback, tool call parsing error flag, iteration limit
        exceeded flag, agent error (if any), and example metadata (id, type, level).
    """

    # load results if they exist
    file_name = f"output/{agent_name}.json"
    try:
        with open(file_name, "r") as f:
            results = json.load(f)
    except FileNotFoundError:
        results = []

    results_df = pd.DataFrame(results)

    for i, example in tqdm(enumerate(dataset), total=len(dataset)):
        # skip if already evaluated
        if len(results_df) > 0:
            if example["question"] in results_df["question"].unique():
                continue

        # run agent and evaluator
        result = await run_agent_eval(
            question=example["question"],
            ground_truth_answer=example["answer"],
            agent_name=agent_name,
            agent_executor=agent_executor,
            evaluator=evaluator,
            eval_prompt_template=eval_prompt_template,
        )

        # add in example metadata
        result.update(
            {
                "task": example["task"],
            }
        )
        results.append(result)

        # save results
        if not os.path.exists(file_name):
            os.makedirs(os.path.dirname(file_name))
        with open(file_name, "w") as f:
            json.dump(results, f)

    return results


async def run_full_eval(
    dataset: Dataset,
    agent_model_endpoints: Dict[str, str],
    evaluator: HuggingFaceChatWrapper,
    eval_prompt_template: ChatPromptTemplate,
) -> pd.DataFrame:
    """
    Run a full evaluation on the given dataset using multiple agent models.

    Args:
        dataset (Dataset): The dataset to evaluate on.
        agent_model_endpoints (List[str]): List of endpoints for the agent models.
        evaluator (HuggingFaceChatWrapper): The evaluator object for evaluating the models.
        eval_prompt_template (ChatPromptTemplate): The template for generating evaluation prompts.

    Returns:
        pd.DataFrame: The evaluation results as a pandas DataFrame.
    """
    results = []

    # agent_executors = [build_agent(endpoint) for endpoint in agent_model_endpoints]
    agents = {
        name: build_agent(endpoint) for name, endpoint in agent_model_endpoints.items()
    }

    tasks = [
        evaluate_on_dataset(
            dataset=dataset,
            agent_name=agent_name,
            agent_executor=agent_executor,
            evaluator=evaluator,
            eval_prompt_template=eval_prompt_template,
        )
        for agent_name, agent_executor in agents.items()
    ]

    results = await asyncio.gather(*tasks)

    return pd.DataFrame([element for sublist in results for element in sublist])
