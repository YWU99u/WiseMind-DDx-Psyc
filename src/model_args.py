import json

def config_llm(llm_type: str = "openai"):
    if llm_type == "openai":
        # Define llm_type and model_name
        LLM = llm_type
        MODEL_FULL = 'gpt-4o'
        MODEL_LIGHT = 'gpt-4o-mini'
        TEMPERATURE = 0.2
    elif llm_type == "anthropic":
        LLM = llm_type
        MODEL_FULL = 'claude-3-5-sonnet-latest'
        MODEL_LIGHT = 'claude-3-5-haiku-latest'
        TEMPERATURE = 0.2
    elif llm_type == "google":
        LLM = llm_type
        MODEL_FULL = 'gemini-2.0-flash-exp'
        MODEL_LIGHT = 'gemini-1.5-flash-002'
        TEMPERATURE = 0.2
    elif llm_type == "mistral":
        LLM = llm_type
        MODEL_FULL = 'mistral-large-latest'
        MODEL_LIGHT = 'ministral-8b-latest'
        TEMPERATURE = 0.2
    elif llm_type == "qwen":
        LLM = llm_type
        MODEL_FULL = 'qwen2.5:72b'
        MODEL_LIGHT = 'qwen2.5:7b'
        TEMPERATURE = 0.2
    elif llm_type == "deepseek":
        LLM = llm_type
        MODEL_FULL = 'deepseek-r1:70b'
        MODEL_LIGHT = 'llama3.1:latest'
        TEMPERATURE = 0.2
    else:
        raise ValueError(f"Unknown LLM type: {llm_type}")
    
    with open('WiseMind_prompts.json') as f:
        DOC_PROPMTS = json.load(f)
    with open('baseline_prompts.json') as f:
        BASELINE_PROMPTS = json.load(f)
    with open('mock_patient_prompts.json') as f:
        MOCK_PATIENT_PROMPTS = json.load(f)

    # Void Memory Doctor
    KFP_DOCTOR_ARG = {
        "llm_type": LLM,
        "model_name": MODEL_FULL,
        "temperature": TEMPERATURE,
        "system_prompt": BASELINE_PROMPTS['KFP_doctor']['system_prompt'],
        "user_prompt": BASELINE_PROMPTS['KFP_doctor']['user_prompt'],
    }

    # Unstructured Memory Doctor
    TKEP_ICL_DOCTOR_ARG = {
        "llm_type": LLM,
        "model_name": MODEL_FULL,
        "temperature": TEMPERATURE,
        "system_prompt": BASELINE_PROMPTS['TKEP_ICL_doctor']['system_prompt'],
        "user_prompt": BASELINE_PROMPTS['TKEP_ICL_doctor']['user_prompt'],
    }

    TKEP_RAG_DOCTOR_ARG = {
        "llm_type": LLM,
        "model_name": MODEL_FULL,
        "temperature": TEMPERATURE,
        "system_prompt": BASELINE_PROMPTS['TKEP_RAG_doctor']['system_prompt'],
        "user_prompt": BASELINE_PROMPTS['TKEP_RAG_doctor']['user_prompt'],
    }


    # Structured Memory Doctor
    WiseMind_DOCTOR_ARG = {
                "WiseMind_decision_maker": {
                    "llm_type": LLM,
                    "model_name": MODEL_FULL, # claude-3-5-sonnet-20241022
                    "temperature": 0,
                    "system_prompt": DOC_PROPMTS['WiseMind_decision_maker']['system_prompt'],
                    "user_prompt": DOC_PROPMTS['WiseMind_decision_maker']['user_prompt'],
                },
                "WiseMind_question_generator": {
                    "llm_type": LLM,
                    "model_name": MODEL_FULL,
                    "temperature": 0.8,
                    "system_prompt": DOC_PROPMTS['WiseMind_question_generator']['system_prompt'],
                    "user_prompt": DOC_PROPMTS['WiseMind_question_generator']['user_prompt'],
                },
                "WiseMind_transitioner": {
                    "llm_type": LLM,
                    "model_name": MODEL_FULL,
                    "temperature": 0.8,
                    "system_prompt": DOC_PROPMTS['WiseMind_transitioner']['system_prompt'],
                    "user_prompt": DOC_PROPMTS['WiseMind_transitioner']['user_prompt'],
                },
                "WiseMind_summarizer": {
                    "llm_type": LLM,
                    "model_name": MODEL_LIGHT,
                    "temperature": TEMPERATURE,
                    "system_prompt": DOC_PROPMTS['WiseMind_summarizer']['system_prompt'],
                    "user_prompt": DOC_PROPMTS['WiseMind_summarizer']['user_prompt'],
                },
                "WiseMind_sanity_checker": {
                    "llm_type": LLM,
                    "model_name": MODEL_LIGHT,
                    "temperature": TEMPERATURE,
                    "system_prompt": DOC_PROPMTS['WiseMind_sanity_checker']['system_prompt'],
                    "user_prompt": DOC_PROPMTS['WiseMind_sanity_checker']['user_prompt'],
                },
                # "WiseMind_contradiction_auditor":{
                #     "llm_type": LLM,
                #     "model_name": MODEL_LIGHT,
                #     "temperature": TEMPERATURE,
                #     "system_prompt": DOC_PROPMTS['WiseMind_contradiction_auditor']['system_prompt'],
                #     "user_prompt": DOC_PROPMTS['WiseMind_contradiction_auditor']['user_prompt'],
                # }
            }

    # Patient
    PATIENT_ARG = {
        "llm_type": LLM,
        "model_name": MODEL_FULL,
        "temperature": TEMPERATURE,
        "system_prompt": MOCK_PATIENT_PROMPTS['patient']['system_prompt'],
        "user_prompt": MOCK_PATIENT_PROMPTS['patient']['user_prompt'],
    }
    return KFP_DOCTOR_ARG, TKEP_ICL_DOCTOR_ARG, TKEP_RAG_DOCTOR_ARG, WiseMind_DOCTOR_ARG, PATIENT_ARG