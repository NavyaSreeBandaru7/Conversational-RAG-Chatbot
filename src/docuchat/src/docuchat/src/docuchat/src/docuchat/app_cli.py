from __future__ import annotations
from src.docuchat.rag import build_chat

def main():
    ask = build_chat()
    print("✅ DocuChat ready. Type /exit to quit.\n")

    while True:
        q = input("You: ").strip()
        if not q:
            continue
        if q.lower() in ["/exit", "exit", "quit"]:
            break

        answer, sources = ask(q)
        print("\nAI:", answer)
        print("\nSources:")
        for s in sources:
            print("-", s)
        print()

if __name__ == "__main__":
    main()
