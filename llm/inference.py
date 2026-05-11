from transformers import pipeline

_pipe = pipeline("text-generation", model="sshleifer/tiny-gpt2")
print("[LLM] Loaded sshleifer/tiny-gpt2 inference pipeline.")


def run_llm(query: str, context: str) -> str:
    prompt = f"Context: {context[:200]}\nQuestion: {query}\nAnswer:"
    try:
        out = _pipe(
            prompt,
            max_new_tokens=40,
            do_sample=False,
            truncation=True,
        )
        if not out:
            return "No answer generated."
        generated = out[0].get("generated_text", "")
        if not isinstance(generated, str):
            return "No answer generated."
        if generated.startswith(prompt):
            answer = generated[len(prompt) :].strip()
        elif prompt in generated:
            answer = generated.split(prompt, 1)[1].strip()
        else:
            answer = generated.strip()
        if not answer:
            return "No answer generated."
        return answer
    except Exception as e:
        return f"LLM inference failed: {repr(e)}"
