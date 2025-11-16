import re
import pandas as pd
from bigtree import tree_to_dataframe, dataframe_to_tree, Node,find_name, dataframe_to_tree_by_relation,BinaryNode
class WiseMind_Tree:
    def __init__(self, disorder_type: str):
        self.tree_path = f"../data/Medical/{disorder_type}_tree_dfs_detailed.csv"
        self.tree, self.diagnostic_labels = self.gen_tree()

    def gen_tree(self) -> tuple:
        '''
        Generate the tree for the given disorder type.
        Args: disorder_type (str): The type of disorder for which the tree is to be generated. Options: 'anx', 'mdd', 'bp'.
        output: tree (BinaryNode): The tree for the given disorder type.
                diagnostic_labels (dict): The diagnostic labels for the given disorder type.
        '''

        df = pd.read_csv(self.tree_path)
        tree = dataframe_to_tree(df, node_type=BinaryNode, attribute_cols=['Description'])
        diagnostic_labels = dict(zip([leaf.name for leaf in tree.leaves], [leaf.Description for leaf in tree.leaves]))
        return tree, diagnostic_labels

    def get_tree_info(self, pointer: str) -> dict:
        """
        Get the tree information for the given pointer.
        Args: pointer (str): The pointer for which the tree information is to be retrieved.
        output: dict: The tree information for the given pointer.
        """

        try:
            pointer = pointer.split("/")[-1]
        except:
            pointer = pointer

        node = find_name(self.tree, pointer)
        return node

def parser_llm_WiseMind(doc_response: str,num_of_ouputs: int =2) -> dict:
    '''
    Parse the WiseMind response from the LLM. 
    Args: doc_response (str): The response from the LLM.
            num_of_ouputs (int): The number of outputs expected from the LLM.
    output: dict: The parsed response from the LLM.
    '''

    final_response = {}
    pattern = r'<([^>]+)>(\n?.*?\n?)<\/\1>'
    pattern2 = r'<([^>]+)>(.*?)<\/\1>'

    # Find all matches in the text
    matches = re.findall(pattern, doc_response)
    if len(matches) != num_of_ouputs:
        matches = re.findall(pattern2, doc_response.replace('\n',''))

    # Print results
    for match in matches:
        tag, content = match
        raw_content = content.replace('\n','')
        final_response[tag] = raw_content
    if len(final_response) != num_of_ouputs:
        for key in ['Reason_for_Action','Action','Response','Reason_for_Response']:
            if key not in final_response:
                if key == 'Response':
                    final_response[key] = 'Can you elaborate more?'
                
                else:
                    final_response[key] = 'error in parsing'

        final_response['Action'] = 'ask_more_detail'
    return final_response

def parser_llm(doc_response: str,num_of_ouputs: int =2) -> dict:
    '''
    Parse the response from the LLM. 
    Args: doc_response (str): The response from the LLM.
            num_of_ouputs (int): The number of outputs expected from the LLM.
    output: dict: The parsed response from the LLM.
    '''

    final_response = {}
    pattern = r'<([^>]+)>(\n?.*?\n?)<\/\1>'

    # Find all matches in the text
    matches = re.findall(pattern, doc_response)

    # Print results
    for match in matches:
        tag, content = match
        raw_content = content.replace('\n','')
        final_response[tag] = raw_content

    return final_response
