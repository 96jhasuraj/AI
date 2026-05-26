from litellm import completion

SYSTEM_PROMPT = """
You are a helpful coding assistant.
Return only the Python code, with no explanation or preamble.
"""

USER_PROMPT = """
Write a Python function that implements the factorial algorithm using recursion.

The name of the function should be `factorial`.

The function should take a single non-negative integer as input.

Include a docstring explaining what the function does.
"""

response = completion(
    model="ollama/deepseek-r1:8b",
    messages=[
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": USER_PROMPT
        }
    ],
    api_base="http://localhost:11434"
)

print(response.choices[0].message.content)