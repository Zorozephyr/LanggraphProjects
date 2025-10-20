# Install required libraries (run in Kaggle/Jupyter)
# !pip install -q transformers torch bitsandbytes accelerate
# !pip install -q peft
# !pip install -q langchain langchain-community langchainhub
# !pip install -q langgraph

# ==================== IMPORTS ====================

from dotenv import load_dotenv
import os

# 1) Load environment FIRST
load_dotenv(override=True)

# 2) Get credentials
AUTOX_API_KEY  = os.getenv("AUTOX_API_KEY")
NTNET_USERNAME = (os.getenv("NTNET_USERNAME") or "").strip()

# 3) Set proxy bypass BEFORE creating HTTP clients
os.environ["NO_PROXY"] = ",".join(filter(None, [
    os.getenv("NO_PROXY",""),
    ".autox.corp.amdocs.azr",
    "chat.autox.corp.amdocs.azr",
    "localhost","127.0.0.1"
]))
os.environ["no_proxy"] = os.environ["NO_PROXY"]

import torch
import json
import re
from typing import Annotated, TypedDict, Any, Optional, List
from operator import add

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, get_peft_model, LoraConfig, TaskType

from langchain.llms.base import LLM
#from langchain.callbacks.manager import CallbackManagerLLMRun
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.outputs import GenerationChunk
from langchain_core.language_models import LLM as CoreLLM

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import Command

# # ==================== CUSTOM LLM WRAPPER ====================
# class QuantizedLlamaLLM(LLM):
#     """Custom LangChain LLM wrapper for quantized Llama model"""
    
#     model: Any = None
#     tokenizer: Any = None
#     max_length: int = 500
#     temperature: float = 0.7
#     top_p: float = 0.9
    
#     @property
#     def _llm_type(self) -> str:
#         return "quantized_llama"
    
#     def _call(
#         self,
#         prompt: str,
#         stop: Optional[List[str]] = None,
#         run_manager: Optional[CallbackManagerLLMRun] = None,
#         **kwargs: Any,
#     ) -> str:
#         """Generate text from prompt"""
#         inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
#         outputs = self.model.generate(
#             **inputs,
#             max_length=self.max_length,
#             temperature=self.temperature,
#             top_p=self.top_p,
#             do_sample=True,
#             pad_token_id=self.tokenizer.eos_token_id
#         )
        
#         response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
#         return response

# # ==================== MODEL SETUP ====================
# print("Loading model and tokenizer...")
# MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"  # or "meta-llama/Meta-Llama-3.1-8B"

# # Quantization configuration (4-bit)
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.float16
# )

# # Load tokenizer
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# tokenizer.pad_token = tokenizer.eos_token

# # Load model with 4-bit quantization
# model = AutoModelForCausalLM.from_pretrained(
#     MODEL_NAME,
#     quantization_config=bnb_config,
#     device_map="auto",
#     trust_remote_code=True
# )

# # Create LangChain LLM instance
# llm = QuantizedLlamaLLM(
#     model=model,
#     tokenizer=tokenizer,
#     max_length=300,
#     temperature=0.7
# )

# print("Model loaded successfully!")


# 4) NOW create HTTP clients with correct proxy settings
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import httpx

http_client = httpx.Client(
    verify=r"C:\amdcerts.pem",  # Use corporate certs
    timeout=30.0
)

# Create async client too (if needed)
async_http_client = httpx.AsyncClient(
    verify=r"C:\amdcerts.pem",
    timeout=30.0
)

# 5) Create LLM with custom HTTP client
llm = AzureChatOpenAI(
    azure_endpoint="https://chat.autox.corp.amdocs.azr/api/v1/proxy",
    api_key=AUTOX_API_KEY,
    azure_deployment="gpt-4o-128k",
    model="gpt-4o-128k",
    temperature=0.1,
    openai_api_version="2024-08-01-preview",
    default_headers={"username": NTNET_USERNAME, "application": "testing-proxyapi"},
    http_client=http_client,
    http_async_client=async_http_client
)

# ==================== STATE DEFINITION ====================
class DatasetGeneratorState(TypedDict):
    """State for the dataset generator graph"""
    messages: Annotated[list[BaseMessage], add_messages]
    use_case: str
    example_data: str
    generated_dataset: list[dict]
    dataset_count: int
    editing_mode: bool
    feedback: str
    success_criteria_met: bool
    validation_report: str
    retry_count: int

# ==================== NODE FUNCTIONS ====================

def input_collection_node(state: DatasetGeneratorState) -> Command:
    """Collect use case and example data from user"""
    print("\n=== Synthetic Dataset Generator ===")
    print("Please provide the following information:\n")
    
    use_case = input("1. Describe your use case (e.g., 'Customer data for e-commerce'): ")
    example_data = input("2. Provide example data format/structure: ")
    
    new_messages = [
        HumanMessage(content=f"Use case: {use_case}"),
        HumanMessage(content=f"Example data: {example_data}")
    ]
    
    return Command(
        update={
            "messages": new_messages,
            "use_case": use_case,
            "example_data": example_data,
            "dataset_count": 0,
            "editing_mode": False,
            "feedback": "",
            "success_criteria_met": False,
            "validation_report": "",
            "retry_count": 0
        },
        goto="generate_dataset"
    )

def generate_dataset_node(state: DatasetGeneratorState) -> Command:
    """Generate synthetic dataset based on use case and example data"""
    print("\n--- Generating Synthetic Dataset ---")
    
    retry_count = state.get("retry_count", 0)
    if retry_count > 0:
        print(f"Regenerating dataset (Attempt {retry_count + 1})...")
    
    base_message = """You are a synthetic data generator. Generate realistic synthetic data based on the use case and example data.
Requirements:
- Generate exactly 50 records
- Follow the structure and format of the example data
- Make data realistic and varied
- Return ONLY valid JSON array, no other text
- Each record should be a JSON object
- Ensure consistency across all records
- Data should be logically coherent with the use case"""
    
    user_message = """Use case: {use_case}

Example data structure:
{example_data}

Generate 50 synthetic records in valid JSON format as an array.
Ensure the data is realistic and follows the use case context. Don't write anything around it. Give only the json output in the response."""
    
    # Add validation report if criteria not met
    if not state["success_criteria_met"] and state.get("validation_report"):
        user_message = user_message + """

Previous validation issues to address:
{validation_report}

Please regenerate the dataset addressing all the issues mentioned above."""
    
    # Add feedback if provided
    if state.get("feedback"):
        user_message = user_message + """

User feedback to incorporate:
{feedback}

Please regenerate the dataset incorporating this feedback."""
    
    # Create prompt for dataset generation
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", base_message),
        ("user", user_message)
    ])
    
    chain = prompt_template | llm | StrOutputParser()
    
    try:
        # Prepare invoke parameters
        invoke_params = {
            "use_case": state["use_case"],
            "example_data": state["example_data"]
        }
        
        # Add optional parameters if they exist
        if state.get("validation_report"):
            invoke_params["validation_report"] = state["validation_report"]
        
        if state.get("feedback"):
            invoke_params["feedback"] = state["feedback"]
        
        response = chain.invoke(invoke_params)
        
        # Extract JSON from response
        json_match = re.search(r'\[.*\]', response, re.DOTALL)
        if json_match:
            dataset = json.loads(json_match.group())
        else:
            dataset = json.loads(response)
        
        # Ensure max 50 records
        dataset = dataset[:50]
        
        new_messages = [
            AIMessage(content=f"Generated {len(dataset)} synthetic records successfully.")
        ]
        
        return Command(
            update={
                "messages": new_messages,
                "generated_dataset": dataset,
                "dataset_count": len(dataset),
                "editing_mode": False,
                "success_criteria_met": False,
                "retry_count": retry_count + 1
            },
            goto="evaluate_dataset"
        )
    
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        error_message = [HumanMessage(content="Failed to generate valid dataset. Please try again.")]
        return Command(
            update={"messages": error_message},
            goto="input_collection"
        )

def evaluate_dataset_node(state: DatasetGeneratorState) -> Command:
    """Evaluate generated dataset for quality and consistency"""
    print("\n--- Evaluating Dataset Quality ---")
    
    dataset = state["generated_dataset"]
    use_case = state["use_case"]
    example_data_str = state["example_data"]
    
    # Create evaluation prompt - ESCAPE JSON structure with double braces
    eval_prompt_template = ChatPromptTemplate.from_messages([
        ("system", """You are a data quality evaluator. Analyze the generated synthetic dataset and provide a detailed evaluation.

Evaluation criteria:
1. Structure Consistency: Do all records follow the example data structure?
2. Data Type Compliance: Are values of correct types (strings, numbers, dates)?
3. Realistic Values: Are the values realistic and plausible for the use case?
4. Logical Coherence: Is there logical consistency across related fields?
5. Data Variety: Is there sufficient variety in the data (not repetitive)?
6. Use Case Alignment: Does the data align with the described use case?

Provide your evaluation in the following JSON format (use actual values, not placeholders):
{{
  "overall_score": 85,
  "structure_compliant": true,
  "data_types_correct": true,
  "realistic": true,
  "logically_coherent": true,
  "sufficient_variety": true,
  "use_case_aligned": true,
  "issues": ["list of specific issues found"],
  "suggestions": ["list of improvement suggestions"],
  "recommendation": "PASS"
}}"""),
        ("user", """Use case: {use_case}

Example data structure:
{example_data}

Generated dataset sample (first 10 records):
{dataset_sample}

Please evaluate this dataset based on the criteria above.""")
    ])
    
    eval_chain = eval_prompt_template | llm | StrOutputParser()
    
    # Get sample of dataset for evaluation
    dataset_sample = json.dumps(dataset[:10], indent=2)
    
    try:
        response = eval_chain.invoke({
            "use_case": use_case,
            "example_data": example_data_str,
            "dataset_sample": dataset_sample
        })
        
        # Extract JSON from response
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            evaluation = json.loads(json_match.group())
        else:
            evaluation = json.loads(response)
        
        # Determine if criteria met
        criteria_met = (
            evaluation.get("overall_score", 0) >= 70 and
            evaluation.get("recommendation", "FAIL").upper() == "PASS"
        )
        
        # Format evaluation report
        validation_report = f"""
=== DATASET VALIDATION REPORT ===
Overall Score: {evaluation.get('overall_score', 0)}/100
Recommendation: {evaluation.get('recommendation', 'UNKNOWN')}

Compliance Checks:
- Structure Compliant: {evaluation.get('structure_compliant', False)}
- Data Types Correct: {evaluation.get('data_types_correct', False)}
- Realistic Data: {evaluation.get('realistic', False)}
- Logically Coherent: {evaluation.get('logically_coherent', False)}
- Sufficient Variety: {evaluation.get('sufficient_variety', False)}
- Use Case Aligned: {evaluation.get('use_case_aligned', False)}

Issues Found:
{chr(10).join([f'  - {issue}' for issue in evaluation.get('issues', [])])}

Suggestions:
{chr(10).join([f'  - {suggestion}' for suggestion in evaluation.get('suggestions', [])])}
"""
        
        print(validation_report)
        
        eval_message = [AIMessage(content=validation_report)]
        
        if criteria_met:
            print("\n✓ Dataset passed validation! Ready for review.")
            return Command(
                update={
                    "messages": eval_message,
                    "success_criteria_met": True,
                    "validation_report": validation_report
                },
                goto="display_and_edit"
            )
        else:
            print("\n✗ Dataset did not meet quality criteria.")
            return Command(
                update={
                    "messages": eval_message,
                    "success_criteria_met": False,
                    "validation_report": validation_report
                },
                goto="handle_evaluation_feedback"
            )
    
    except json.JSONDecodeError as e:
        print(f"Error parsing evaluation: {e}")
        error_message = [HumanMessage(content="Error during evaluation. Proceeding to manual review.")]
        return Command(
            update={"messages": error_message},
            goto="display_and_edit"
        )

def handle_evaluation_feedback_node(state: DatasetGeneratorState) -> Command:
    """Handle feedback when evaluation fails"""
    print("\n--- Dataset Quality Review ---")
    print(state["validation_report"])
    
    print("\nOptions:")
    print("1. Regenerate dataset (AI will improve based on issues)")
    print("2. Proceed anyway (accept dataset as is)")
    print("3. Return to input collection")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == "1":
        issues = "Identified issues that need to be fixed. " + state["validation_report"]
        feedback_message = [HumanMessage(content=f"Please regenerate the dataset addressing these issues: {issues}")]
        
        return Command(
            update={
                "messages": feedback_message,
                "feedback": "regenerate_from_evaluation",
                "generated_dataset": [],
                "dataset_count": 0
            },
            goto="generate_dataset"
        )
    
    elif choice == "2":
        accept_message = [HumanMessage(content="User accepted dataset despite quality concerns")]
        return Command(
            update={
                "messages": accept_message,
                "success_criteria_met": True,
                "editing_mode": True
            },
            goto="display_and_edit"
        )
    
    elif choice == "3":
        return_message = [HumanMessage(content="User chose to restart with new parameters")]
        return Command(
            update={"messages": return_message},
            goto="input_collection"
        )
    
    else:
        error_message = [HumanMessage(content="Invalid choice")]
        return Command(
            update={"messages": error_message},
            goto="handle_evaluation_feedback"
        )

def display_and_edit_node(state: DatasetGeneratorState) -> Command:
    """Display generated dataset and ask for edits"""
    print(f"\n--- Generated Dataset ({state['dataset_count']} records) ---")
    print(f"Quality Status: {'PASSED' if state['success_criteria_met'] else 'REVIEW MODE'}")
    
    # Display first 3 records as preview
    print("\nPreview of generated data (first 3 records):")
    for i, record in enumerate(state["generated_dataset"][:3]):
        print(f"Record {i+1}: {json.dumps(record, indent=2)}")
    
    print(f"\n... and {max(0, state['dataset_count'] - 3)} more records")
    
    print("\nOptions:")
    print("1. Accept dataset and export")
    print("2. Regenerate dataset")
    print("3. Edit specific records")
    print("4. Provide feedback for improvement")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == "1":
        accept_message = [AIMessage(content="Dataset accepted and ready for use.")]
        return Command(
            update={
                "messages": accept_message,
                "editing_mode": False,
                "success_criteria_met": True
            },
            goto="export_dataset"
        )
    
    elif choice == "2":
        feedback_message = [HumanMessage(content="User requested dataset regeneration")]
        return Command(
            update={
                "messages": feedback_message,
                "feedback": "regenerate",
                "generated_dataset": [],
                "dataset_count": 0
            },
            goto="generate_dataset"
        )
    
    elif choice == "3":
        record_num = input("Enter record number to edit (1-50): ").strip()
        new_values = input("Enter new values in JSON format: ").strip()
        
        try:
            record_idx = int(record_num) - 1
            updated_record = json.loads(new_values)
            
            if 0 <= record_idx < len(state["generated_dataset"]):
                state["generated_dataset"][record_idx].update(updated_record)
                edit_message = [AIMessage(content=f"Record {record_num} updated successfully.")]
                return Command(
                    update={"messages": edit_message},
                    goto="display_and_edit"
                )
            else:
                error_message = [HumanMessage(content="Invalid record number")]
                return Command(
                    update={"messages": error_message},
                    goto="display_and_edit"
                )
        except (ValueError, json.JSONDecodeError):
            error_message = [HumanMessage(content="Invalid input format")]
            return Command(
                update={"messages": error_message},
                goto="display_and_edit"
            )
    
    elif choice == "4":
        feedback = input("Provide feedback for improvement: ").strip()
        feedback_message = [HumanMessage(content=f"User feedback: {feedback}")]
        return Command(
            update={
                "messages": feedback_message,
                "feedback": feedback,
                "generated_dataset": [],
                "dataset_count": 0
            },
            goto="generate_dataset"
        )
    
    else:
        error_message = [HumanMessage(content="Invalid choice")]
        return Command(
            update={"messages": error_message},
            goto="display_and_edit"
        )

def export_dataset_node(state: DatasetGeneratorState) -> Command:
    """Export the final dataset"""
    print("\n--- Final Dataset Ready ---")
    print(f"Total records: {state['dataset_count']}")
    
    # Return dataset
    dataset_json = json.dumps(state["generated_dataset"], indent=2)
    print("\nFinal Dataset:")
    print(dataset_json)
    
    # Optional: Save to file
    with open("synthetic_dataset.json", "w") as f:
        f.write(dataset_json)
    print("\nDataset saved to 'synthetic_dataset.json'")
    
    export_message = [AIMessage(content="Dataset is ready for export")]
    
    return Command(
        update={"messages": export_message},
        goto=END
    )

# ==================== BUILD GRAPH ====================
workflow = StateGraph(DatasetGeneratorState)

# Add nodes
workflow.add_node("input_collection", input_collection_node)
workflow.add_node("generate_dataset", generate_dataset_node)
workflow.add_node("evaluate_dataset", evaluate_dataset_node)
workflow.add_node("handle_evaluation_feedback", handle_evaluation_feedback_node)
workflow.add_node("display_and_edit", display_and_edit_node)
workflow.add_node("export_dataset", export_dataset_node)

# Add edges
workflow.add_edge(START, "input_collection")

# Compile graph
graph = workflow.compile()

# ==================== RUN APPLICATION ====================
if __name__ == "__main__":
    initial_state = {
        "messages": [],
        "use_case": "",
        "example_data": "",
        "generated_dataset": [],
        "dataset_count": 0,
        "editing_mode": False,
        "feedback": "",
        "success_criteria_met": False,
        "validation_report": "",
        "retry_count": 0
    }
    
    result = graph.invoke(initial_state)
    
    print("\n=== Session Complete ===")
    print(f"Total messages: {len(result['messages'])}")
    print(f"Final dataset records: {result['dataset_count']}")
    print(f"Quality criteria met: {result['success_criteria_met']}")