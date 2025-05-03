# SPDX-License-Identifier: Apache-2.0

from openai import OpenAI

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8193/v1"


def main():
    client = OpenAI(
        # defaults to os.environ.get("OPENAI_API_KEY")
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    # models = client.models.list()
    # model = models.data[0].id
    model = "Qwen/Qwen3-0.6B"

    # Completion API
    stream = False
    completion = client.completions.create(
        model=model,
        prompt=
        "The best part about working on vLLM is that I got to meet so many people across various different organizations like UCB, Google, and Meta which means",  # noqa: E501
        echo=False,
        stream=stream)

    print("-" * 50)
    print("Completion results:")
    if stream:
        for c in completion:
            print(c)
    else:
        print(completion)
    print("-" * 50)


if __name__ == "__main__":
    main()
