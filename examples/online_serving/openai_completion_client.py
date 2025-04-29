# SPDX-License-Identifier: Apache-2.0

from openai import OpenAI

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8192/v1"


PROMPT = "Question: Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market? Answer:"

def main():
    client = OpenAI(
        # defaults to os.environ.get("OPENAI_API_KEY")
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    # models = client.models.list()
    # model = models.data[0].id

    # Completion API
    stream = False
    completion = client.completions.create(
        model="meta-llama/Llama-3.1-8B-Instruct",
        prompt=PROMPT,
        echo=False,
        max_tokens=100,
        stream=stream)

    print("-" * 50)
    print("Completion results:")
    if stream:
        text = ""
        for c in completion:
            print(c)
            text += c.choices[0].text
        
        print("\n")
        print(text)
    else:
        print(completion)
    print("-" * 50)


if __name__ == "__main__":
    main()
