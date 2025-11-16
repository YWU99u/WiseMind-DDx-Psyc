from collections import Counter
import re
from llm_agents import LLM_Agent
import pandas as pd
from utils import WiseMind_Tree, parser_llm, parser_llm_WiseMind
import os
import time
from tqdm import tqdm
from bigtree import tree_to_dataframe
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from fastapi import WebSocket
from model_args import config_llm
import json
import asyncio
#STORE_DIR = f"../result/baseline_results_{disorder_type}"


class WiseMind_agent:
    """
    WiseMind_agent is a class that initializes an LLM_Agent and invokes it with given arguments.

    Attributes:
        input_arg (dict): A dictionary containing the initialization parameters for the LLM_Agent.
            - 'llm_type' (str): The type of the language model.
            - 'model_name' (str): The name of the model to be used.
            - 'temperature' (float): The temperature parameter for the model.
            - 'system_prompt' (str): The system prompt to be used.
            - 'user_prompt' (str): The user prompt to be used.

    Methods:
        __init__(input_arg: dict):
            Initializes the WiseMind_agent with the given input arguments.

        invoke(trun_arg: dict) -> str:
            Generates a prompt using the LLM_Agent and invokes it with the generated prompt.
            Args:
                trun_arg (dict): A dictionary containing the arguments for generating the prompt.
            Returns:
                str: The response from the LLM_Agent after invoking it with the generated prompt.
    """
    def __init__(self, input_arg: dict):
        self.input_arg = input_arg
        self.agent = LLM_Agent(llm_type= input_arg['llm_type'],model_name=input_arg['model_name'], temperature=input_arg['temperature'])
    def invoke(self,trun_arg: dict) -> str:
        turn_prompt = self.agent.generate_prompt(trun_arg, self.input_arg['system_prompt'], self.input_arg['user_prompt'])
        return self.agent.invoke(turn_prompt)

async def KFP_doctor(patient_df: pd.DataFrame, WiseMind_tree, websocket: WebSocket, hci: bool = False, llm_type = 'openai') -> dict:
    """
    Asynchronous Void Memory Doctor using FastAPI WebSocket.
    
    This function simulates a diagnostic conversation between a doctor and a patient using the WiseMind framework.
    The doctor agent, referred to as the "void memory doctor," does not retain memory of past interactions, making each turn independent.
    The conversation continues until a final diagnostic decision is reached.

    Args:
        patient_df (pd.DataFrame): DataFrame containing patient information.
        WiseMind_tree: An instance of the WiseMind_Tree class used to generate the diagnostic tree and labels.
        websocket (WebSocket): FastAPI WebSocket instance to communicate with the client.
        hci (bool, optional): Flag indicating Human-Computer Interaction mode. Defaults to False.

    Returns:
        dict: A dictionary containing the results of the experiment, including:
            - 'final_path' (str): The final diagnostic path taken.
            - 'conversations' (list): A list of conversation turns between the doctor and patient.
            - 'complete_dict' (dict): A dictionary detailing the nodes, criteria met, and reasons during the diagnostic process.
            - 'result' (str): The final diagnostic decision.
            - 'num_of_rounds' (int): The number of conversation rounds conducted.
            - 'inference_time' (float): The total time taken for the inference process, in seconds.
    """
    # Generate the diagnostic tree and labels
    tree, diagnostic_labels = WiseMind_tree.gen_tree()

    # Initialize the doctor and patient agents
    KFP_DOCTOR_ARG,_,_,_,PATIENT_ARG = config_llm(llm_type)
    KFP_doctor_arg = KFP_DOCTOR_ARG
    patient_arg = PATIENT_ARG

    KFP_doctor = WiseMind_agent(KFP_doctor_arg)
    patient = WiseMind_agent(patient_arg)

    # Initialize conversation memory and other variables
    st_memo = ['Doctor: Hi there. I am Dr. WiseMind. I am a AI doctor copilot to help the doctors collect some infomation about your situations. You can feel free to tell me about your real thoughts, ask me any question or clarifications :) What can I assist you for today?']
    final_path = ''
    complete_dict = {'Node': [], 'Met_Criteria': [], 'Reasons': []}
    now = time.time()
    final_decision = 'None'
    upper_bound = 100
    await websocket.send_text(f"{st_memo[-1]}")
    # Loop until a final decision is made
    while final_decision == 'None':
        if not hci:
            while True:
                try:
                    # Prepare the patient turn arguments
                    patient_turn_arg = {
                        'profile': patient_df.to_dict(orient='records'),
                        'question': st_memo[-1],
                        'st_memo': st_memo[:-1]
                    }
                    # Invoke the patient agent and get the response
                    patient_res = parser_llm(patient.invoke(patient_turn_arg))['Response']
                    break
                except Exception as e:
                    print(f'Error in patient response: {e}')
                    continue
        else:
            # await websocket.send_text("Please provide the patient's response:")
            patient_res = await websocket.receive_text()

        st_memo.append('Patient: ' + patient_res)
        # await websocket.send_text(f"Patient: {patient_res}")
        print(f"Patient: {patient_res}")

        while True:
            try:
                # Prepare the doctor turn arguments
                doc_input_arg = {
                    'diagnostic_labels': diagnostic_labels,
                    'patient_response': st_memo[-1],
                    'st_memo': st_memo[:-1]
                }
                # Invoke the doctor agent and get the response
                res = parser_llm(KFP_doctor.invoke(doc_input_arg))
                doc_res = res['Response']
                final_decision = str(res['Final_Decision'])
                break
            except Exception as e:
                print(f'Error in doctor response: {e}')
                continue

        st_memo.append('Doctor: ' + doc_res)
        await websocket.send_text(f"Doctor: {doc_res}")
        print(f"Doctor: {doc_res}")
        
        if len(st_memo) // 2 > upper_bound:
            final_decision = 'I cannot diagnose'

    return {
        'final_path': final_path,
        'conversations': st_memo,
        'complete_dict': complete_dict,
        'result': final_decision,
        'num_of_rounds': len(st_memo) // 2,
        'inference_time': round(time.time() - now, 2)
    }

async def TKEP_ICL_doctor(patient_df: pd.DataFrame, WiseMind_tree, websocket: WebSocket, hci: bool = False, llm_type = 'openai') -> dict:
    """
    Asynchronous Unstructured In-Context Doctor using FastAPI WebSocket.
    
    This function simulates a diagnostic conversation asynchronously using FastAPI WebSocket.
    The doctor agent uses in-context learning to make diagnostic decisions, continuing until a final decision is reached.

    Args:
        patient_df (pd.DataFrame): DataFrame containing patient information.
        WiseMind_tree: An instance of the WiseMind_Tree class used to generate the diagnostic tree and labels.
        websocket (WebSocket): FastAPI WebSocket instance to communicate with the client.
        hci (bool, optional): Flag indicating Human-Computer Interaction mode. Defaults to False.

    Returns:
        dict: A dictionary containing the results of the experiment, including:
            - 'final_path' (str): The final diagnostic path taken.
            - 'conversations' (list): A list of conversation turns between the doctor and patient.
            - 'complete_dict' (dict): A dictionary detailing the nodes, criteria met, and reasons during the diagnostic process.
            - 'result' (str): The final diagnostic decision.
            - 'num_of_rounds' (int): The number of conversation rounds conducted.
            - 'inference_time' (float): The total time taken for the inference process, in seconds.
    """
    # Initialize the doctor and patient agents
    _,TKEP_ICL_DOCTOR_ARG,_,_,PATIENT_ARG = config_llm(llm_type)
    TKEP_ICL_doctor_arg = TKEP_ICL_DOCTOR_ARG
    TKEP_ICL_doctor = WiseMind_agent(TKEP_ICL_doctor_arg)
    if not hci:
        patient_arg = PATIENT_ARG
        patient = WiseMind_agent(patient_arg)

    # Generate the diagnostic tree and labels
    tree, diagnostic_labels = WiseMind_tree.gen_tree()
    tree_df = tree_to_dataframe(tree, attr_dict={'Description': 'Description'})
    criteria_dict = dict(zip(tree_df['name'], tree_df['Description']))

    # Initialize conversation memory and other variables
    st_memo = ['Doctor: Hi there. I am Dr. WiseMind. What can I do for you today?']
    now = time.time()
    final_path_li = []
    complete_dict = {'Node': [], 'Met_Criteria': [], 'Reasons': []}
    final_decision = 'None'
    upper_bound = 30
    
    # Loop until a final decision is made
    while final_decision == 'None':
        await websocket.send_text(f"{st_memo[-1]}")
        if not hci:
            while True:
                try:
                    # Prepare the patient turn arguments
                    patient_turn_arg = {
                        'profile': patient_df.to_dict(orient='records'),
                        'question': st_memo[-1],
                        'st_memo': st_memo[:-1]
                    }
                    # Invoke the patient agent and get the response
                    patient_res = parser_llm(patient.invoke(patient_turn_arg))['Response']
                    break
                except Exception as e:
                    print(f'Error in patient response: {e}')
                    continue
        else:
            # await websocket.send_text("Please provide the patient's response:")
            patient_res = await websocket.receive_text()

        st_memo.append('Patient: ' + patient_res)
        # await websocket.send_text(f"Patient: {patient_res}")
        print(f"Patient: {patient_res}")

        while True:
            try:
                # Prepare the doctor turn arguments
                doc_input_arg = {
                    'diagnostic_labels': diagnostic_labels,
                    'patient_responce': st_memo[-1],
                    'st_memo': st_memo[:-1],
                    'criteria': criteria_dict
                }
                # Invoke the doctor agent and get the response
                res = parser_llm(TKEP_ICL_doctor.invoke(doc_input_arg))
                doc_res = res['Response']
                final_decision = str(res['Final_Decision'])
                knowledge_used = res['Knowledge_Used']
                node = knowledge_used.split(',')[0][1:]
                
                binary = knowledge_used.split(',')[-1][1:-1]
                print(node,binary)
                reasons = res['Reason']
                complete_dict['Node'].append(node)
                complete_dict['Met_Criteria'].append(binary)
                complete_dict['Reasons'].append(reasons)
                final_path_li.append(node)
                break
            except Exception as e:
                print(f'Error in doc response: {e}')
                continue

        st_memo.append('Doctor: ' + doc_res)
        
        print(f"Doctor: {doc_res}")

        if len(st_memo) // 2 > upper_bound:
            final_decision = 'I cannot diagnose'

    # Construct the final diagnostic path
    final_path = '->'.join(list(set(final_path_li)))
    final_path += f'->{final_decision}'

    return {
        'final_path': final_path,
        'conversations': st_memo,
        'complete_dict': complete_dict,
        'result': final_decision,
        'num_of_rounds': len(st_memo) // 2,
        'inference_time': round(time.time() - now, 2)
    }

async def TKEP_RAG_doctor(patient_df: pd.DataFrame, WiseMind_tree, websocket: WebSocket, hci: bool = False, llm_type = 'openai') -> dict:
    """
    Asynchronous Unstructured RAG Doctor using FastAPI WebSocket.

    This function simulates a diagnostic conversation asynchronously using FastAPI WebSocket.
    The doctor agent uses retrieved documents to make diagnostic decisions, continuing until a final decision is reached.

    Args:
        patient_df (pd.DataFrame): DataFrame containing patient information.
        WiseMind_tree: An instance of the WiseMind_Tree class used to generate the diagnostic tree and labels.
        websocket (WebSocket): FastAPI WebSocket instance to communicate with the client.
        hci (bool, optional): Flag indicating Human-Computer Interaction mode. Defaults to False.

    Returns:
        dict: A dictionary containing the results of the experiment.
    """
    # Initialize the doctor and patient agents
    _,_,TKEP_RAG_DOCTOR_ARG,_,PATIENT_ARG = config_llm(llm_type)
    TKEP_RAG_doctor_arg = TKEP_RAG_DOCTOR_ARG
    TKEP_RAG_doctor = WiseMind_agent(TKEP_RAG_doctor_arg)
    if not hci:
        patient_arg = PATIENT_ARG
        patient = WiseMind_agent(patient_arg)

    # Generate the diagnostic tree and labels
    tree, diagnostic_labels = WiseMind_tree.gen_tree()

    # Load the tree data and initialize the retriever
    loader = CSVLoader(file_path=WiseMind_tree.tree_path)
    data = loader.load()
    vectorstore = Chroma.from_documents(documents=data, embedding=OpenAIEmbeddings())
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    # Initialize conversation memory and other variables
    st_memo = ['Doctor: Hi there. I am Dr. WiseMind. What can I do for you today?']
    tree_df = tree_to_dataframe(tree, attr_dict={'Description': 'Description'})
    criteria_dict = dict(zip(tree_df['name'], tree_df['Description']))
    now = time.time()
    final_path_li = []
    complete_dict = {'Node': [], 'Met_Criteria': [], 'Reasons': []}
    final_decision = 'None'
    upper_bound = 30
    
    # Loop until a final decision is made
    while final_decision == 'None':
        await websocket.send_text(f"{st_memo[-1]}")
        if not hci:
            while True:
                try:
                    # Prepare the patient turn arguments
                    patient_input_arg = {
                        'profile': patient_df.to_dict(orient='records'),
                        'question': st_memo[-1],
                        'st_memo': st_memo[:-1]
                    }
                    # Invoke the patient agent and get the response
                    patient_res = parser_llm(patient.invoke(patient_input_arg))['Response']
                    break
                except Exception as e:
                    print(f'Error in patient response: {e}')
                    continue
        else:
            await websocket.send_text("Please provide the patient's response:")
            patient_res = await websocket.receive_text()

        st_memo.append('Patient: ' + patient_res)
        # await websocket.send_text(f"Patient: {patient_res}")
        print(f"Patient: {patient_res}")

        while True:
            try:
                # Retrieve relevant documents based on the patient's response
                docs = retriever.get_relevant_documents(patient_res)
                doc_txt = docs[1].page_content.split('\n')

                # Prepare the doctor turn arguments
                doc_input_arg = {
                    'diagnostic_labels': diagnostic_labels,
                    'patient_response': st_memo[-1],
                    'st_memo': st_memo[:-1],
                    'context': doc_txt
                }
                # Invoke the doctor agent and get the response
                res = parser_llm(TKEP_RAG_doctor.invoke(doc_input_arg))
                doc_res = res['Response']
                final_decision = str(res['Final_Decision'])
                knowledge_used = res['Knowledge_Used']
                node = knowledge_used.split(',')[0][1:]
                binary = knowledge_used.split(',')[-1][1:-1]
                print(node,binary)
                reasons = res['Reason']
                complete_dict['Node'].append(node)
                complete_dict['Met_Criteria'].append(binary)
                complete_dict['Reasons'].append(reasons)
                final_path_li.append(node)
                break
            except Exception as e:
                print(f'Error in doctor response: {e}')
                continue

        st_memo.append('Doctor: ' + doc_res)
        
        print(f"Doctor: {doc_res}")

        if len(st_memo) // 2 > upper_bound:
            final_decision = 'I cannot diagnose'

    # Construct the final diagnostic path
    final_path = '->'.join(list(set(final_path_li)))
    final_path += f'->{final_decision}'

    return {
        'final_path': final_path,
        'conversations': st_memo,
        'complete_dict': complete_dict,
        'result': final_decision,
        'num_of_rounds': len(st_memo) // 2,
        'inference_time': round(time.time() - now, 2)
    }

async def WiseMind_doctor(patient_df: pd.DataFrame, WiseMind_tree, websocket: WebSocket, hci: bool = False, llm_type: str = "openai") -> dict:
    """
    Asynchronous diagnostic conversation function using FastAPI WebSocket.
    """
    def initialize_agents(llm_type: str = 'openai') -> dict:
        _,_,_,doc_arg,patient_arg = config_llm(llm_type)
        WiseMind_arg = doc_arg
        patient_arg = patient_arg
        return {
            'decision_maker': WiseMind_agent(WiseMind_arg['WiseMind_decision_maker']),
            'question_generator': WiseMind_agent(WiseMind_arg['WiseMind_question_generator']),
            'transitioner': WiseMind_agent(WiseMind_arg['WiseMind_transitioner']),
            'summarizer': WiseMind_agent(WiseMind_arg['WiseMind_summarizer']),
            'sanity_checker': WiseMind_agent(WiseMind_arg['WiseMind_sanity_checker']),
            # 'contradiction_auditor': WiseMind_agent(WiseMind_arg['WiseMind_contradiction_auditor'])
        }

    def sanity_check(num_of_sc: int, input_arg: dict) -> str:
        """
        Perform a sanity check by invoking the sanity checker agent multiple times and taking a majority vote.
        """
        # Invoke the `sanity_checker` asynchronously
        mv_list = [WiseMind_agents['sanity_checker'].invoke(input_arg) for _ in range(num_of_sc)]
        vote_counts = Counter(mv_list)
        return vote_counts.most_common(1)[0][0]

    async def get_patient_response(node):
        question = f"{st_memo[-1]}"
        await websocket.send_text(question)

        try:
            # Await the response from the client
            response = await websocket.receive_text()
            # await websocket.send_text(f"Patient: {response}")
            return response
        except Exception as e:
            print(f"Unexpected error: {e}")
            raise

    async def check_node_condition(node,close_ended = False) -> tuple:
        nonlocal st_memo, complete_conversations, summary_bank, transiting
        recheck_node = False

        patient_res = await get_patient_response(node)
        # print(f"Patient: {patient_res}")
        st_memo.append('Patient: ' + patient_res)
        complete_conversations.append('Patient: ' + patient_res)

        decision_maker_input_arg = {
            'st_memo': f'Previous Summarization: {summary_bank} \n Current dialogue: {st_memo[:-1]}',
            'node': node,
            'patient_res': patient_res
        }
        
        ## Contradiction detection

        # contradiction_auditor_input_arg = {
        #     'summarization': f'Previous Summarization: {summary_bank}',
        #     'patient_res': patient_res
        # }
        print(f'summarization: {summary_bank}')

        if transiting and (not close_ended):
            transitioner_input_arg = {
                'st_memo': f'Previous Summarization: {summary_bank} \n Current dialogue: {st_memo[:-1]}',
                'node': node,
                'patient_res': patient_res
            }
            while True: 
                try:
                    decisions = parser_llm_WiseMind(WiseMind_agents['decision_maker'].invoke(decision_maker_input_arg))
                    trans_question = parser_llm(WiseMind_agents['transitioner'].invoke(transitioner_input_arg), num_of_ouputs=3)
                    if decisions['Action'] in ['more_detail','detect_contradiction']:
                        action = decisions['Action']
                    else:
                        action = 'ask_more_detail'
                    response = {
                        'Reason_for_Action': decisions['Reason_for_Action'],
                        'Action': action,
                        'Response': trans_question['Question'],
                        'Reason_for_Response': f'Transitioning: {trans_question["Reason_Trust"]}'
                    }
                    transiting = trans_question['Trust'] == 'False'
                    break
                except Exception as e:
                    print(f'Error in doctor response: {e}')
                    continue
        else:
            transiting = False
            while True:
                try:
                    decisions = parser_llm_WiseMind(WiseMind_agents['decision_maker'].invoke(decision_maker_input_arg))
                    if close_ended:
                        question_generator_input_arg = {
                            'st_memo': f'Previous Summarization: {summary_bank} \n Current dialogue: {st_memo[:-1]}',
                            'node': node,
                            'patient_res': patient_res + '**YOU MUST START ASKING CLOSE ENDED QUESTIONS**',
                            'action': decisions['Action']
                        }
                    else:
                        question_generator_input_arg = {
                            'st_memo': f'Previous Summarization: {summary_bank} \n Current dialogue: {st_memo[:-1]}',
                            'node': node,
                            'patient_res': patient_res,
                            'action': decisions['Action']
                        }
                    responses = parser_llm_WiseMind(WiseMind_agents['question_generator'].invoke(question_generator_input_arg))
                    response = {
                        'Reason_for_Action': decisions['Reason_for_Action'],
                        'Action': decisions['Action'],
                        'Response': responses['Response'],
                        'Reason_for_Response': responses['Reason_for_Response']
                    }
                    break
                except Exception as e:
                    print(f'Error in doctor response: {e}')
                    continue
    
        await websocket.send_text(f"Diagnosis Procedure Status: {response['Action']},{node.name}")
        if response['Action'] in ['met_criteria', 'not_met_criteria']:
            # print(st_memo)
            summarization = WiseMind_agents['summarizer'].invoke({'st_memo': st_memo,'node': node})
            summary_bank[node.name] = {
                'summarization':summarization,
                'conversations': st_memo,
                'decision':response['Action']
            }
            
            
            st_memo = []
            # sanity_check_input_arg = {
            #     'summarization': summarization,
            #     'meet': 'MEETS' if response['Action'] == 'met_criteria' else 'DOES NOT MEET',
            #     'node_description': node,
            #     'decisions': response['Reason_for_Action']
            # }
 
            # # Use await for `sanity_check` as it is now asynchronous
            # if sanity_check(1, sanity_check_input_arg) == 'True':
            #     complete_conversations.append(st_memo)
            #     summary_bank[node.name] = summarization
            #     st_memo = []
            # else:
            #     print('*****************Failing Sanity Check************************')
            #     response['Action'] = 'ask_more_detail'
            #     response['Reason_for_Action'] = 'I need to recheck the patient\'s response'
            #     response['Response'] = 'Sorry, it seems that I have missed something, can you describe your condition again?'
            #     response['Reason_for_Response'] = 'I need to recheck the patient\'s response'
            transiting = True
        elif response['Action'] == 'detect_contradiction':
            previous_checked_node = summary_bank.keys()
            pattern = r"|".join(map(re.escape, previous_checked_node))
            regex = re.compile(pattern)
            if regex.search(response['Reason_for_Action']):
                recheck_node = regex.search(response['Reason_for_Action']).group(0)
                print(patient_res)
                print(response['Response'])
                summary_bank[node.name] = ''
            else:
                response['Action'] = 'ask_more_detail'
        print(f"Node: {node.name}\nDecision: {response['Action']}")

        st_memo.append('Doctor: ' + response['Response'])
        complete_conversations.append('Doctor: ' + response['Response'])
        # print(f'Doctor: {response["Response"]}')
        print(response)
        return response['Action'] == 'met_criteria', response['Action'] == 'ask_more_detail', {
            'Action': response['Action'],
            'Reason_for_Action': response['Reason_for_Action'],
            'Reason_for_Response': response['Reason_for_Response']
        }, recheck_node

    async def tree_search(node: object) -> str:
        nonlocal final_path, skipped_node
        complete_dict['Node'].append(node.name)

        if node.is_leaf:
            final_path += node.name
            complete_dict['Met_Criteria'].append(True)
            complete_dict['Reasons'].append(['At the diagnostic point, no need to reason'])
            return node.name

        final_path += node.name + ' -> '
        more_detail = True
        reasons = []
        check_count = 0

        while more_detail and check_count < 10:
            met_criteria, more_detail, reason, recheck_node = await check_node_condition(node)
            if recheck_node:
                print('*****************Rechecking node: ', recheck_node, ' due to contradiction************************')
                break
            reasons.append(reason)
            check_count += 1
        while more_detail and check_count < 15:
            print('STARTING CLOSE ENDED QUESTIONS')
            met_criteria, more_detail, reason, recheck_node = await check_node_condition(node,close_ended=True)
            if recheck_node:
                print('*****************Rechecking node: ', recheck_node, ' due to contradiction************************')
                break
            reasons.append(reason)
            check_count += 1

        if check_count == 15 and more_detail:
            met_criteria = False
            more_detail = False
            reasons.append("Unable to decide after asking 15 questions")

        complete_dict['Met_Criteria'].append(met_criteria)
        complete_dict['Reasons'].append(reasons)

        if recheck_node:
            return await tree_search(WiseMind_tree.get_tree_info(recheck_node))

        if met_criteria:
            result = await tree_search(node.left)
        else:
            result = await tree_search(node.right)

        return result

    # Initialize conversation state variables
    WiseMind_agents = initialize_agents(llm_type=llm_type)
    tree, diagnostic_labels = WiseMind_tree.gen_tree()
    st_memo = ['Doctor: Hi there. I am Dr. WiseMind. I am a AI doctor copilot to help the doctors collect some infomation about your situations. You can feel free to tell me about your real thoughts, ask me any question or clarifications :) What can I assist you for today?']
    final_path = ''
    skipped_node = None
    complete_dict = {'Node': [], 'Met_Criteria': [], 'Reasons': []}
    complete_conversations = []
    transiting = True
    summary_bank = {}
    now = time.time()

    # Perform depth-first search on the diagnostic tree
    result = await tree_search(tree.left)
    num_of_round = sum(len(item) for item in complete_conversations)
    return {
        'final_path': final_path,
        'summarizations': summary_bank,
        'conversations': complete_conversations,
        'complete_dict': complete_dict,
        'result': result,
        'num_of_rounds': num_of_round // 2,
        'inference_time': round(time.time() - now, 2)
    }





async def process_single_disorder(
    sample_dir: str,
    Disorder_Name: str,
    all_disorders: list,
    tree_type: str,
    doctor_type: str,
    hci: bool = False,
    llm_type: str = 'openai',
    websocket=None  # WebSocket instance from FastAPI
) -> dict:
    """
    Process a single disorder example.

    This function processes a single example of a disorder by simulating a diagnostic conversation between a doctor and a patient using the WiseMind framework.
    In HCI mode, it uses WebSocket to communicate with the client.
    """

    async def non_hci(fp_sample: str, websocket: WebSocket):
        try:
            # Read and parse the JSON file
            with open(fp_sample, 'r') as f:
                data = json.load(f)

            # Flatten the conversation data
            flat_li = []
            for item in data['details']['conversations']:
                if isinstance(item, list):
                    flat_li.extend(item)
                else:
                    flat_li.append(item)

            # Separate doctor and patient responses
            flat_li_doc = flat_li[::2]
            flat_li_pat = flat_li[1::2]

            # Build actions list from nodes and reasons
            actions = []
            for node, reasons in zip(data['details']['complete_dict']["Node"], data['details']['complete_dict']["Reasons"]):
                for reason_item in reasons:
                    if isinstance(reason_item, dict) and 'Action' in reason_item:
                        actions.append((node, reason_item['Action']))

            # Ensure lists are all the same length
            min_length = min(len(flat_li_doc), len(flat_li_pat), len(actions))
            flat_li_doc = flat_li_doc[:min_length]
            flat_li_pat = flat_li_pat[:min_length]
            actions = actions[:min_length]

            # Send messages via WebSocket with a delay
            continue_task = False
            for doc, pat, action in zip(flat_li_doc, flat_li_pat, actions):
                # Send doctor message
                if not continue_task:
                    mode = await websocket.receive_text()
                else:
                    await websocket.send_text(f"{doc}")
                    await asyncio.sleep(1)  # 1-second delay
                    # Send patient message
                    await websocket.send_text(f"{pat}")
                    await asyncio.sleep(1)  # 1-second delay
                    # Send Diagnosis Procedure status
                    await websocket.send_text(f"Diagnosis Procedure Status: {action[1]},{action[0]}")
                    await asyncio.sleep(0.2)  # 1-second delay
                if mode == 'NEXT':
                    await websocket.send_text(f"{doc}")
                    await asyncio.sleep(0.5)  # 1-second delay
                    # Send patient message
                    await websocket.send_text(f"{pat}")
                    await asyncio.sleep(0.5)  # 1-second delay
                    # Send Diagnosis Procedure status
                    await websocket.send_text(f"Diagnosis Procedure Status: {action[1]},{action[0]}")
                elif mode == 'CONTINUE':
                    continue_task = True
                    mode = websocket.receive_text()
                

            return data  # Return the data after the loop completes

        except Exception as e:
            # Handle any errors that occur
            await websocket.send_text(f"An error occurred: {str(e)}")
            return None
    # Initialize the WiseMind tree
    WiseMind_tree = WiseMind_Tree(tree_type)
    tree, _ = WiseMind_tree.gen_tree()
    df = pd.read_csv(WiseMind_tree.tree_path)

    # Determine the doctor function to use
    if doctor_type == 'KFP_doctor':
        doctor_trail = KFP_doctor
    elif doctor_type == 'TKEP_ICL_doctor':
        doctor_trail = TKEP_ICL_doctor
    elif doctor_type == 'TKEP_RAG_doctor':
        doctor_trail = TKEP_RAG_doctor
    elif doctor_type == 'WiseMind_doctor':
        doctor_trail = WiseMind_doctor
    else:
        raise ValueError(f"Invalid doctor type: {doctor_type}")

    

    # HCI mode: Run the doctor trail asynchronously using WebSocket
    if hci:
        result = await doctor_trail(patient_df=None, WiseMind_tree=WiseMind_tree, hci=hci, websocket=websocket,llm_type=llm_type)

        # Return the results
        return {
            'gt_disorder': 'hci',
            'pred_disorder': result['result'],
            'is_final_correct': False,
            'intermediate_accuracy': 0,
            'patient_file_path': None,
            'details': result,
            'result_description': WiseMind_tree.get_tree_info(result['result']).Description
        }
    else:
        # Non-HCI mode: Load the patient data from the sample directory
        non_hci_result = await non_hci('../result/demo_example/mdd.json', websocket=websocket)
        return non_hci_result
if __name__ == "__main__":
    # Define the LLM configuration
    llm_type = 'anthropic'
    tree_type = 'anx'
    TEST_DIR = f'../data/Medical/mock_patient_{tree_type}/'
    all_disorders = list(set([f"{case.split('.')[0].split('_')[0]}_0" for case in os.listdir(TEST_DIR)]))
    final_result = []

    for Disorder_Name in tqdm(all_disorders, desc="Processing disorders"):
        result = process_single_disorder(sample_dir=TEST_DIR,Disorder_Name=Disorder_Name, all_disorders=all_disorders,tree_type=tree_type,doctor_type='WiseMind_doctor',hci=False)
        if result:
            print(result['gt_disorder'], result['pred_disorder'], result['is_final_correct'])
            final_result.append(result)
