import argparse
import os
import json
import random
from collections import defaultdict

import pandas as pd
from openai import OpenAI

client = OpenAI()

# ---------- Tree structure ----------

class DiagnosisTree:
    def __init__(self, csv_path: str):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Tree file not found: {csv_path}")

        self.df = pd.read_csv(csv_path)
        self.df['path'] = self.df['path'].str.rstrip('/')

        # path -> node info
        self.path_to_node = {
            row['path']: {
                'name': row['name'],
                'description': row.get('Description', '')
            }
            for _, row in self.df.iterrows()
        }

        # parent -> children
        self.children = defaultdict(list)
        self.parents = {}
        for p in self.path_to_node:
            parts = p.strip('/').split('/')
            parent = None if len(parts) == 1 else '/' + '/'.join(parts[:-1])
            self.parents[p] = parent
            if parent:
                self.children[parent].append(p)

        # leaf nodes = terminal diagnoses / endpoints
        self.leaf_paths = [p for p in self.path_to_node if p not in self.children]

        # fixed ordering of nodes for consistent vectors
        self.all_node_paths = sorted(self.path_to_node.keys())

    def path_from_root(self, leaf_path: str):
        parts = leaf_path.strip('/').split('/')
        return ['/' + '/'.join(parts[:i]) for i in range(1, len(parts) + 1)]

    def random_leaf(self) -> str:
        return random.choice(self.leaf_paths)

    def leafs_for_diagnosis(self, diagnosis_name: str):
        return [
            p for p in self.leaf_paths
            if self.path_to_node[p]['name'] == diagnosis_name
        ]

    def binary_labels_for_path(self, path_list):
        path_set = set(path_list)
        return {
            p: int(p in path_set) for p in self.all_node_paths
        }


# ---------- LLM prompt + call ----------

SYSTEM_PROMPT = """
You are a clinical script writer creating realistic first-person narratives 
for virtual standardized patients in psychiatric evaluation. 
You write short, coherent answers that all belong to the same fictional patient. 
Be consistent about their background, relationships, work/study, and history. 
Do not mention diagnoses, DSM, or criteria explicitly.
"""

def build_user_prompt(disorder_name: str, vsp_id: str, nodes_for_prompt):
    """
    nodes_for_prompt: list of dicts with keys:
      - Node
      - Met_Criteria (bool)
      - Description (str)
    """
    nodes_json_str = json.dumps(nodes_for_prompt, indent=2)
    return f"""
We are constructing a Virtual Standard Patient (VSP) for psychiatric diagnostic simulation.

Disorder: {disorder_name}
VSP ID: {vsp_id}

You are given a list of clinical nodes. Each node has:
- "Node": a machine-readable ID for the node,
- "Met_Criteria": a Boolean indicating whether this criterion is present for this patient,
- "Description": an explanation of what the clinician is probing.

Your task:

1. First, internally imagine a single coherent life story for this patient:
   - Choose a plausible age, gender, occupation or study situation, and living arrangement.
   - Decide on a consistent background (family, relationships, major life events, stressors, culture).
   - Decide on a realistic timeline of symptom development and key turning points.

2. Then, for each node, write 2–5 sentences in first person that the patient might say when talking about that topic
   as part of the same life story:
   - All nodes must reflect the *same* persona, background, and chronology.
   - If "Met_Criteria" is true, the answer should clearly support the presence of that symptom in a natural, narrative way
     (e.g., through everyday examples, feelings, behaviors).
   - If "Met_Criteria" is false, the answer should clearly NOT support the symptom and may gently deny it, while still
     being consistent with the rest of the story (e.g., “I usually sleep fine...”).
   - Avoid copy–paste patterns. The language, examples, and details should feel human and varied.

3. Maintain strict internal consistency:
   - Do NOT contradict yourself about relationships, work/study, timing, or symptom severity across nodes.
   - Reuse the same people, places, and events (e.g., same partner, same job, same family situation) across different nodes.

4. Style constraints:
   - Write in natural, conversational first-person language, as if the patient is speaking to a clinician.
   - Do NOT mention DSM, diagnosis labels, node IDs, or any internal criteria explicitly.
   - Do NOT output any explanations about what you are doing—only the patient’s narrative content inside JSON.

Return ONLY valid JSON in the following format:

{{
  "nodes": [
    {{
      "Node": "<node_id>",
      "Met_Criteria": true,
      "Patient_Story": "<2–5 sentences in first-person, consistent with the shared life story>"
    }},
    ...
  ]
}}

Here are the nodes for this VSP:
{nodes_json_str}
""".strip()



def call_llm(system_prompt: str, user_prompt: str, model_name: str = "gpt-4.1"):
    """
    Call OpenAI Chat Completions API and return JSON-string output.
    Uses response_format to enforce JSON when supported, and logs raw output if empty.
    """
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=0.7,
        # If your installed SDK / model version supports this, it will force JSON.
        # If not, the param will be ignored but will not break.
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content

    # Debug logging
    print("==== RAW MODEL OUTPUT (repr) ====")
    print(repr(raw))
    print("=================================")

    if raw is None or not str(raw).strip():
        raise RuntimeError("Model returned empty or whitespace-only response from chat.completions")

    # Ensure it's a plain string
    raw = str(raw)

    # In case the model still insists on ```json fences, strip them
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        # remove possible leading "json" language tag
        if cleaned.lstrip().lower().startswith("json"):
            cleaned = cleaned.lstrip()[4:].lstrip()

    return cleaned
    



# ---------- VSP generation from tree ----------

def generate_vsp_from_tree(
    tree_csv: str,
    disorder: str,
    vsp_id: str,
    leaf_path: str = None,
    diagnosis_name: str = None,
    out_dir: str = "../data/Medical",
    model_name: str = "gpt-4.1",
):

    tree = DiagnosisTree(tree_csv)

    # 1) choose which leaf (diagnosis) to use
    if leaf_path is not None:
        if leaf_path not in tree.leaf_paths:
            raise ValueError(f"Provided leaf_path not found: {leaf_path}")
        chosen_leaf = leaf_path
    elif diagnosis_name is not None:
        candidates = tree.leafs_for_diagnosis(diagnosis_name)
        if not candidates:
            raise ValueError(f"No leaf found with diagnosis name '{diagnosis_name}'.")
        chosen_leaf = random.choice(candidates)
    else:
        chosen_leaf = tree.random_leaf()

    path_nodes = tree.path_from_root(chosen_leaf)
    binary_labels = tree.binary_labels_for_path(path_nodes)

    # 2) build node list for LLM prompt
    nodes_for_prompt = []
    for p in tree.all_node_paths:
        node_info = tree.path_to_node[p]
        nodes_for_prompt.append({
            "Node": p,
            "Met_Criteria": bool(binary_labels[p]),
            "Description": node_info["description"],
        })

    # 3) build prompt & call LLM
    user_prompt = build_user_prompt(disorder, vsp_id, nodes_for_prompt)
    raw = call_llm(SYSTEM_PROMPT, user_prompt, model_name=model_name)

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        print("Failed to parse JSON from model output.")
        print("Raw output was:\n", raw)
        raise e

    story_map = {n["Node"]: n["Patient_Story"] for n in parsed["nodes"]}

    # 4) assemble final VSP CSV 
    rows = []
    for p in tree.all_node_paths:
        node_info = tree.path_to_node[p]
        rows.append({
            "Node": p,
            "Met_Criteria": binary_labels[p],
            "Description": node_info["description"],
            "Patient_Story": story_map.get(p, ""),
        })

    df = pd.DataFrame(rows)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{vsp_id}.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved VSP script to: {out_path}")

    # optional: you can also return parsed["global_background"] if you want it
    return out_path, parsed.get("global_background", "")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate a VSP script CSV from a DSM tree.")
    parser.add_argument("--disorder", required=True, choices=["mdd", "anx", "bp"],
                        help="Which disorder tree to use (mdd/anx/bp).")
    parser.add_argument("--vsp-id", required=True,
                        help="Identifier for this VSP, e.g. MDD_0, BPI_3.")
    parser.add_argument("--diagnosis-name", type=str, default=None,
                        help="Optional diagnosis name to target (e.g. 'MDD'); if omitted, choose random leaf.")
    parser.add_argument("--leaf-path", type=str, default=None,
                        help="Optional exact leaf path to target; overrides --diagnosis-name if provided.")
    parser.add_argument("--data-dir", type=str, default="../data/Medical",
                        help="Directory containing *_tree_dfs_detailed.csv.")
    parser.add_argument("--out-dir", type=str, default="../data/Medical/vsp_scripts",
                        help="Where to store generated VSP CSVs.")
    parser.add_argument("--model", type=str, default="gpt-4o",
                        help="Model name for LLM generation.")

    args = parser.parse_args()

    tree_csv = os.path.join(args.data_dir, f"{args.disorder}_tree_dfs_detailed.csv")

    generate_vsp_from_tree(
        tree_csv=tree_csv,
        disorder=args.disorder,
        vsp_id=args.vsp_id,
        leaf_path=args.leaf_path,
        diagnosis_name=args.diagnosis_name,
        out_dir=args.out_dir,
        model_name=args.model,
    )
