"""
Useful functions from original code generation script from APPS.
"""

import io
import json
import random

import os

from code_ai_tree_search.eval.reindent import run as run_reindent

# okay with parallelization
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def reindent_code(codestr):
    """
    Given code string, reindent it in the same way that the
    Github dataset was indented
    """
    codestr = io.StringIO(codestr)
    ret = io.StringIO()

    run_reindent(
        codestr,
        ret,
        config = {
            "dry-run": False,
            "help": False,
            "to": 10,
            "from": -1,
            "tabs": True,
            "encoding": "utf-8",
            "is-tabs": False,
            "tabsize": 10,
            "all-tabs": False
        }
    )

    return ret.getvalue()

def generate_apps_prompt(args, test_case_path, prompt_path, solutions_path, tokenizer, starter_path=None):
    _input = "\nQUESTION:\n"
    with open(prompt_path, "r") as f:
        data = f.readlines()
        data = "".join(data)
    _input += data
    if starter_path != None:
        with open(starter_path, "r") as f:
            data = f.readlines()
            data = "".join(data)
            data = "\n" + data #+ "\n"
        _input += data
    else:
        #_input += "\n\n"
        pass

    with open(test_case_path, "r") as f:
        data = json.load(f)
    if not data.get("fn_name"):
        _input += "\nUse Standard Input format."#\n"
    else:
        _input += "\nUse Call-Based format."#\n"
    _input += "\nYour answer should be exactly the python 3 code and nothing else.#\n"

    _input += "\nANSWER:\n"

    if args.peeking > 0.0:
        # Need to do some peeking.

        # Read one example solution
        with open(solutions_path, 'r') as f:
            sols = json.load(f)

        # Choose the shortest solution for the model to use.
        # This is so we can conserve tokens (1024 max)
        # sample_sol = min(sols, key=len)

        # # Add args.peeking% of that solution to the prompt
        # sample_sol_token_ids = tokenizer.encode(sample_sol, verbose=False)
        # num_to_keep = int(len(sample_sol_token_ids) * args.peeking)
        # sample_sol_token_ids = sample_sol_token_ids[:num_to_keep]
        # _input += tokenizer.decode(sample_sol_token_ids)

        # Alternatively take a random solution
        sample_sol = random.choice(sols)
        rand_sol = reindent_code(sample_sol)
        rand_sol = tokenizer.encode(rand_sol, verbose=False)
        tokens_taken = int(args.peek_frac * len(rand_sol))
        rand_sol = rand_sol[:tokens_taken]
        _input += tokenizer.decode(rand_sol)
    else:
        sample_sol = None

    return _input, sample_sol

def get_output_str_from_state_for_apps(s):
    """
    Get the code from the transformer output.
    Extracts the content inside triple backticks if present, without using regex.
    """
    # If the model output includes "ANSWER:", strip everything before it
    if "ANSWER:" in s:
        s = s.split("ANSWER:\n", 1)[1]

    # Remove the end-of-text sentinel
    s = s.replace("<|endoftext|>", "")

    # Look for the first opening triple backticks
    start = s.find("```")
    if start != -1:
        # Find the end of the opening fence (skip any language tag)
        newline_after_fence = s.find("\n", start + 3)
        if newline_after_fence != -1:
            # Look for the closing triple backticks after that
            end = s.find("```", newline_after_fence + 1)
            if end != -1:
                # Extract the content between fences
                code_block = s[newline_after_fence + 1 : end]
                return code_block.strip()

    # Fallback to returning the cleaned-up string
    return s.strip()
