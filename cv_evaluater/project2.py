import os
import uuid
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain.llms import Ollama
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Dict

# Initialize LLM
llm = Ollama(model="llama3.2:latest", base_url="http://localhost:11434")

# File paths
cv_folder = "data/cv/"
job_path = "data/job_description.txt"

# Define State class properly
class State(TypedDict):
    messages: Dict[str, str]
    best_cv: str
    best_score: int

# Utility function to read a file
def read_file(file_path):
    with open(file_path, "r") as f:
        return f.read()

# Step 1: Read Job Description
def get_job_description(state: State):
    print("***** get_job_description *****")
    return {
        "messages": {
            **state["messages"],
            "job_content": read_file(job_path),
        },
        "best_cv": "",
        "best_score": 0,
    }

# Step 2: Read CVs
def get_cv_list(state: State):
    print("***** get_cv_list *****")
    cv_files = [f for f in os.listdir(cv_folder) if f.endswith(".txt")]
    cv_files = cv_files[:2]
    return {
        **state,
        "messages": {
            **state["messages"],
            "cv_list": cv_files,
        }
    }

def evaluate_cv(state: State):
    print("***** evaluate_cv *****")
    job_content = state["messages"]["job_content"]
    cv_list = state["messages"]["cv_list"][:2]  # Limit to 3 CVs for faster debugging
    print("Processing CVs:", cv_list)

    best_cv = ""
    best_score = 0
    results = {}

    for cv_file in cv_list:
        cv_content = read_file(os.path.join(cv_folder, cv_file))

        # Create LLM prompt
        prompt = f"""
        Compare the following CVs with the job description and rate the match on a scale of 0-100. 
        if there is a cv with score before recent cv and previous cv with each other according to explanations.

        CV:
        {cv_content}

        Job Description:
        {job_content}

        Response Format:
        Score: <numeric value>
        Explanation: <write full name .brief reasoning. max 10 centences>
        """

        response = llm.invoke([SystemMessage(content="Evaluate CV-job match"), HumanMessage(content=prompt)]).strip()

        print(f"\nReceived LLM Response for {cv_file}:\n{response}\n")

        # Extract score
        try:
            score_line = next(line for line in response.split("\n") if "Score:" in line)
            score = int(score_line.split(":")[1].strip())
        except (ValueError, StopIteration):
            score = 0  # Default if extraction fails

        results[cv_file] = {"score": score, "explanation": response}

        # Update best CV
        if score > best_score:
            best_cv = cv_file
            best_score = score

    return {
        "messages": {
            **state["messages"],
            "results": results,
        },
        "best_cv": best_cv,
        "best_score": best_score,
    }

# Step 4: Generate Email
def generate_email(state: State):
    print("***** generate_email *****")
    best_cv = state["best_cv"]
    best_score = state["best_score"]
    job_content = state["messages"]["job_content"]

    if not best_cv:
        email_content = "Unfortunately, we could not find a suitable match for this position."
    else:
        # Read selected CV content
        cv_content = read_file(os.path.join(cv_folder, best_cv))

        # Create LLM prompt for email generation
        prompt = f"""
        Write a professional email to {best_cv.replace(".txt", "")}, informing them about their selection for the job. 
        Include a congratulatory message, key reasons for selection max 3 cen
        , and next steps.

        CV:
        {cv_content}

        Job Description:
        {job_content}

        Response Format:
        Subject: <email subject>
        Body: <email body>
        """

        response = llm.invoke([SystemMessage(content="Generate candidate email"), HumanMessage(content=prompt)]).strip()
        email_content = response

    return {
        "messages": {
            **state["messages"],
            "email": email_content,
        }
    }


# Create the LangGraph Workflow
workflow = StateGraph(State)

# Define Nodes
workflow.add_node("get_job_description", get_job_description)
workflow.add_node("get_cv_list", get_cv_list)
workflow.add_node("evaluate_cv", evaluate_cv)
workflow.add_node("generate_email", generate_email)

# Define Flow
workflow.add_edge(START, "get_job_description")
workflow.add_edge("get_job_description", "get_cv_list")
workflow.add_edge("get_cv_list", "evaluate_cv")
workflow.add_edge("evaluate_cv", "generate_email")
workflow.add_edge("generate_email", END)

# Compile Graph
graph = workflow.compile()

# Run the workflow **correctly**
final_state = graph.invoke({
    "messages": {
        "cv_list": [],
        "job_content": None,
        "results": None,
        "email": None,
    },
    "best_cv": "",
    "best_score": 0,
})

# Print Results
print("\n***** Process Completed Successfully *****")
print(f"\nBest CV: {final_state['best_cv']} with a score of {final_state['best_score']}.")

# Print Email
email_content = final_state["messages"].get("email", "No email generated.")
print("\nGenerated Email:\n", email_content)