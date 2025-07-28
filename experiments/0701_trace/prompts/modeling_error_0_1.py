import re


SYSTEM = """\
You are an expert grader that never makes mistakes. 

### Question Type
You will be given a step-by-step SOLUTION for a QUESTION that asks for the marginal distribution of some random variable $v_i$ under an intervention on some other variable $v_j$ ($i$ is allowed to equal $j$). 

### Catching Errors
You will be given the definition of an ERROR. Your task is to determine whether the SOLUTION made that ERROR. Justify your decision by pointing out EVIDENCE - locations in SOLUTION where the ERROR happens, and provide a short EXPLANATION.

### Formatting Response
Put your EVIDENCE, if any, within the <evidence></evidence> tag, then your EXPLANATION, if any, within the <explanation></explanation> tag, and finally put your judgement (choose from "Yes" or "No") in <judgement></judgement> tag, where Yes means it contains the ERROR defined above, and No means it does not contain the ERROR defined above. Note that a SOLUTION can contain multiple kinds of errors, please focus on judging only the ERROR that is defined above.
"""

USER = """\
### QUESTION:
<question>
{question}
</question>

### SOLUTION:
<student_solution>
{answer}
</student_solution>

### ERROR definition:
<error_definition>
{error}
</error_definition>

Does the SOLUTION make the ERROR defined above?"""

MODELING_ERROR = """\
Modeling Error
* This category of errors focuses on mistakes that happened during the derivation of NEW probability distributions in SOLUTION. NEW distributions includes any distribution that is computed in SOLUTIOn but not PROVIDED by QUESTION. For example, the target distribution p($v_i$), is often a NEW probability distribution (which is rarely provided in QUESTION). The SOLUTION will often also contain NEW joint and conditional distributions not directly provided by QUESTION.

* A SOLUTION contains a Modeling Error if any the definition or derivation of a NEW probability distribution contains erroneous probability formulas, indicating a potential error in student's knowledge.

* In NEW full joint distributions, which often result from multiplying several conditional distributions, erroneous formulas often has incorrect terms, or are missing or have extra terms. This can be a result of making incorrect independence assumptions are made that's not warranted by the graph in QUESTION (e.g substituting  $p(x) \cdot p(y)$ for $p(x, y)$ or $p(x)$ for $p(x | y)$, when the graph does not suggest independence between x and y). Sometimes the dependence between two variables $v_i$ and $v_j$ is evident only if you look at the full graph or a large enough subgraph (e.g. $v_i$ and $v_j$ are long descendents off of two separate branches off of a common ancestor $v_k$). Sometimes indepedence assumptions are warranted not by the graph structure, but by the numerical value of the distribution (e.g. when $p(V_i=v_i) = 1$ for some $v_i$).

* In NEW marginal joint distributions, which involves summing some full joint over values assignments, erroneous formulas often misses summing over some variables (e.g. incorrectly using $p(x) = \sum_y p(x,y,z)$ when it should have also summed over $z$), or did not sum over all values exhaustively (e.g. incorrectly using $p(x) = p(x,y=0,z=0) + p(x,y=0,z=1)$ which forgets to sum over other values of $y$).

* A SOLUTION can also contain a Modeling Error if they did not consider the intervention on $v_i$ properly. This can happen in ONLY two ways, 1) if the SOLUTION attempted to compute the distribution of $v_i$ from its parents, or 2) if the SOLUTION attempted to marginalize over $v_i$ (incorrect since $v_i$ is intervened on and set to a fixed value). 

What's NOT a Modeling Error? 
* It is NOT a Modeling Error when the student copied CPTs numerically incorrectly from the QUESTION, or when they multiply or add numbers to incorrect results. This includes misreading the CPTs. These are low-level copy and calculation errors, not Modeling Error.

* It is NOT a Modeling Error to ignore the intervention when the intervened on variable $v_i$ is NOT an ancestor of any of the question variable $v_j$. In that case, the intervention has no effect on $v_j$ and one should just compute $v_j$ in the original graph.

* Focus on probability formulas that are actually used to compute the solution. If the student makes an erroneous statement but does not make modeling errors in the actual formulas, do NOT count it as a Modeling Error.
"""

def parse_judgement(s):
    match = re.search(r"<judgement>(.*?)</judgement>", s, re.DOTALL | re.IGNORECASE)
    value = None
    if match:
        content = match.group(1).strip().lower()
        if content == "yes":
            value = 1
        elif content == "no":
            value = 0
        else:
            value = -1
    else:
        value = -2
    return value

POST_PROCESS=lambda x: {'value': parse_judgement(x), 'raw': x}
RESPONSE_FORMAT=None

def format_system_prompt():
    return SYSTEM

def format_user_prompt(question, answer, error=MODELING_ERROR):
    return USER.format(question=question, answer=answer, error=error)


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
