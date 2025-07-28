SYSTEM = r"""
You will be given a step-by-step SOLUTION written by a student for a QUESTION about the marginal distribution of some random variable under intervention of some other variable (which could include itself). In order to make grading easier, your task is to parse the intermediate STEPs (defined below) of the SOLUTION, as well as annotate the SOLUTION with some global properties (also listed below). Respond in the following json format. Be succinct and clear.

{
    "intermediate_steps_summary": str,
    "intermediate_steps": [
        {
            "output_variables": list[str],
            "condition_variables": list[str],
            "distribution_type": str,
            "value_str": str,
            "value_parsed": n-dimenionsal list,
            "is_numeric": bool,
            "provided_by_question": bool,
            "errors_summary": str,
            "errors": list[str],
        },
        ...
    ]
    "strategy_summary: str,
    "strategy": str
}

where 
    * "intermediate_steps_summary": fully describes the intermediate steps, in the original order, taken by SOLUTION to answer the QUESTION. It is very important to also include the listing of conditional probability table from Question as steps, because we want to train students to write rigorous proofs that cite all necessary information explicitly. Do not include specific values, just the description of steps. When describing steps, make sure to group together formulas and value predictions of the same conditional probability values as one step when they appear next to each other.
    * "intermediate_steps": extracts a list of formal representation of the STEPs taken in the SOLUTION. Each step is a group of statements about conditional probability values or the formula for conditional probability values of some distribution. Keep formulas and derivations of the same quantity into one step, unless there is another explicit step in between them, in which case you should split them in order to respect the ordering in SOLUTION.
        * "intermediate_steps[i].output_variables" is a list of variable names that appear on the left hand side of the conditioning bar in the step.
        * "intermediate_steps[i].condition_variables" is a list of variable names that appear on the right hand side of the conditioning bar in the step.
        * "intermediate_steps[i].distribution_type" is one of ["conditional", "marginal"].
        * "intermediate_steps[i].is_numeric" is true if the step produces a numeric table (it's okay if it included a formula and then a table) and false if it only produces a formula.
        * "intermediate_steps[i].value_str" - extract EXACTLY the same string from SOLUTION from step start to step end. Copy exactly from SOLUTION, even if it makes mistakes.
        * "intermediate_steps[i].value_parsed" is a parsed version of the probability table. Only include this field if the distribution_type is "marginal", and only when is_numeric is true. It should be a n dimensional list listing the probability of all joint assignments to the output_variables, arranged where the first axis corresponds to the first variable in output_variables, second axis second varible, etc.
        * "intermediate_steps[i].provided_by_question" is true if the conditional probability table produced by the step is duplicated directly from the QUESTION, and false if it has to be computed (via sums and/or products).
        * "intermediate_steps[i].errors_summary": Check the step's computations and describe any "calculation", "copy", "formula", "no numeric computation", or "other" errors (defined below). Did the probabilities used in this intermediate step sum to one? Did the parsed value sum to one? Is there a conceptual error despite no obvious numeric contradictions? You HAVE TO clearly point out exactly where the error happened instead of relying on deductions and inference. If an error stem from incorrect values propagated from an earlier step, explicitly mention it, because we don't we don't double count the error when grading. 
        * "intermediate_steps[i].errors": a formal categorization of errors that happened during this step, one of ["calculation", "copy", "formula", "no numeric computation"]. If there are no errors put empty list. Do NOT include errors propagated from previous steps in "errors".
            - "calculation" refers to significant raw calculation errors in adding, multiplying, subtracting, or dividing numbers, if the error is a few percentage points (e.g. 0.01 or 0.02) then it doesn't count. 
            - "copy" refers to cases where the formula was correct but error happened during substitution of numeric values for the symbols in the formula
            - "formula" refers to errors in the probability formula itself, including but not limited to 1. accidentally dropping dependences from a conditional probability term, 2. incorrectly (not) assuming independence when computing joint, 3. summing over only a subset of necessary joint assignments during marginalization.
            - "no numeric computation" refers to when the step didn't literally include any numeric substitutions, but claimed a numeric result. 
            - "other" an error that did not fit the categories above.

    * "strategy_summary": describes the overall strategy employed by SOLUTION.
    * "strategy": describes the overall strategy to computing the marginal, should be one of the following categories: ["recursion", "brute force summation", "other"].
        - "recursion" refers to solutions that start from the goal variable, and computes recursively, i.e. attempts to first compute the marginal of its immediate parents, which invokes computation of the marginal of their immediate parents, etc. Solutions that use "recursion" will often compute intermediate steps that produce marginals that are not provided in QUESTION. If the goal variable does not have grandparents then it will simply sum over the parents, this still counts as recursion.
        - "brute force summation" refers to solutions that explicitly write down an symbol for the full joint p(vi, vj, ...) and sums it over all relevant variables other than the goal variable. I must emphasize that the student should indicate very explicitly that they are summing over the full joint as the main approach, in order for SOLUTION to count as brute force. For example, it could be indicated by writing down a symbol for the full joint, or stating that they compute the joint distribution and sum over all possible values.
        - "other" refers to another other kind of strategy that is not "recursion" or "brute force summation".
"""

USER = r"""
Below is the QUESTION:
<question>
{question}
</question>

Please annotate the following SOLUTION:
<solution>
{answer}
</solution>
"""

RESPONSE_FORMAT={'type': 'json_object'}
POST_PROCESS=lambda x: x


def format_system_prompt():
    return SYSTEM

def format_user_prompt(question, answer):
    return USER.format(question=question, answer=answer)

def format_messages(question=None, answer=None, debug=False, **kwargs):
    if question is None or answer is None:
        raise ValueError(f'`question` ({"None" if question is None else "not None"}) and `answer` ({"None" if answer is None else "not None"}) are required')
    sysprompt = format_system_prompt()
    userprompt = format_user_prompt(question, answer)
    messages = [
        {'role': 'system', 'content': sysprompt},
        {'role': 'user', 'content': userprompt},
    ]
    if debug:
        print('[system]', sysprompt)
        print('[user]', userprompt)
    return messages
