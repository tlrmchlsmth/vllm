# SPDX-License-Identifier: Apache-2.0

from openai import OpenAI

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8192/v1"


def main():
    client = OpenAI(
        # defaults to os.environ.get("OPENAI_API_KEY")
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    # models = client.models.list()
    # model = models.data[0].id

    # Completion API
    stream = True
    completion = client.completions.create(
        model="meta-llama/Llama-3.2-1B-Instruct",
        prompt=
        "The absolute best part about working for Red Hat is that we get to work on open source software. Red Hat is a leader in many key open source infrastructure technologies like Linux, Kubernetes, and recently vLLM, which means that there is a lot of opportunity to work with community and customers on key infrastructure projects. This means",  # noqa: E501
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
